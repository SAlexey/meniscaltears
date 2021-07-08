import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torchvision.models._utils import IntermediateLayerGetter

from .linear import MLP
from hydra.utils import instantiate

# helpers


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def tripple(t):
    return t if isinstance(t, (tuple, list)) else (t, t, t)


# classes


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        attn = self.attend(dots)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim, heads=heads, dim_head=dim_head, dropout=dropout
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        channels=1,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        cls_out: int = 2,
        box_out: int = 12,
    ):
        super().__init__()
        image_depth, image_height, image_width = image_size
        patch_depth, patch_height, patch_width = patch_size

        # for an mri image of size (160, 384, 384)
        # can expect patch of size (16,  32,  32)

        assert all(
            (
                (image_depth % patch_depth == 0),
                (image_height % patch_height == 0),
                (image_width % patch_width == 0),
            )
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (
            (image_depth // patch_depth)
            * (image_height // patch_height)
            * (image_width // patch_width)
        )
        patch_dim = channels * patch_depth * patch_height * patch_width

        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (d p0) (h p1) (w p2) -> b (d h w) (p0 p1 p2 c)",
                p0=patch_depth,
                p1=patch_height,
                p2=patch_width,
            ),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 2, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 2, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_cls = nn.Sequential(
            nn.LayerNorm(dim), MLP(dim, dim, cls_out, num_layers=3, dropout=0.5)
        )
        self.mlp_box = nn.Sequential(
            nn.LayerNorm(dim), MLP(dim, dim, box_out, num_layers=3, dropout=0.5)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)

        # x = x.mean(dim=1) if self.pool == "mean" else x[:, 2]

        cls_token = x[:, 1]
        box_token = x[:, 2]

        x = self.to_latent(x)
        out = {"labels": self.mlp_cls(cls_token), "boxes": self.mlp_box(box_token)}
        return out


class ConViT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        heads,
        mlp_dim,
        cls_tokens=1,
        pool="cls",
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        cls_out=6,
        box_out=12,
    ):
        super().__init__()

        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        self.to_patch_embedding = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=24, kernel_size=3, stride=2),
            nn.BatchNorm3d(24),
            nn.Conv3d(in_channels=24, out_channels=48, kernel_size=3, stride=2),
            nn.BatchNorm3d(48),
            nn.Conv3d(in_channels=48, out_channels=96, kernel_size=3, stride=2),
            nn.BatchNorm3d(96),
            nn.Conv3d(in_channels=96, out_channels=128, kernel_size=3, stride=2),
            nn.BatchNorm3d(128),
            nn.Conv3d(in_channels=128, out_channels=dim, kernel_size=1),
            Rearrange("b n d h w -> b (d h w) n"),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, 4761 + cls_tokens, dim))
        self.cls_token = nn.Parameter(torch.randn(1, cls_tokens, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_cls = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, cls_out))
        self.mlp_box = nn.Sequential(nn.LayerNorm(dim), MLP(dim, 1024, box_out, 3, 0.5))

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, *_ = x.shape

        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, :1]

        x = self.to_latent(x)
        out = {"labels": self.mlp_cls(x), "boxes": self.mlp_box(x)}
        return out
