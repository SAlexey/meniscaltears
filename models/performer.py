from einops.einops import repeat
from performer_pytorch import FastAttention
import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange


class PerfoMeD(nn.Module):

    def __init__(self, hidden_dim, heads=8, dim_head=64, num_encoder_layers=4, num_decoder_layers=4, qkv_bias=False, attn_out_bias=False, dropout=0.0, activation=nn.ReLU()):
        
        dim = dim_head * heads

        self.heads = heads
        self.hidden_dim = hidden_dim

        self.to_k = nn.Linear(dim, hidden_dim, bias=qkv_bias)
        self.to_q = nn.Linear(dim, hidden_dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, hidden_dim, bias=qkv_bias)

        self.to_out = nn.Linear(hidden_dim, dim, bias=attn_out_bias)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        self.encoder = nn.ModuleList([EncoderLayer(dim) for _ in range(num_encoder_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(dim) for _ in range(num_encoder_layers)])


    def forward(self, src, pos, query):
        
        bs, c, d, h, w = src.shape  
        
        src, pos = map(lambda s: rearrange(src, "bs c d h w -> (d h w) bs c"), (src, pos))
        
        query = repeat(query, "num 1 hidden_dim -> num bs hidden_dim", bs=bs, hidden_dim=self.hidden_dim)
        

        tgt = torch.zeros_like(query)

        ctx = self.encoder(src, pos)    # latent context 
        tgt = self.decoder(tgt, ctx, pos, query)

        return tgt, ctx     
        


class EncoderLayer(nn.Module):
    
    def __init__(self, dim, heads, dim_head, dropout=0.0):
        
        self.norm = nn.LayerNorm(dim)
        self.sattn = FastAttention(dim_head)

        self.proj_out = nn.Linear(dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            
        )

        # split to multihead
        self.to_mh = Rearrange("len bs (heads dim_head) -> bs len heads dim_head", heads=heads, dim_head=dim_head)

    def forward(self, src, pos):
        
        
        k = self.to_k(src) + pos
        q = self.to_q(src) + pos
        v = self.to_v(src)

        # split between heads    
        k, q, v = map(self.multi_head_split, (k, q, v))
        
        mh_sattn = self.sattn(k, q, v)

        return self.proj_out(mh_sattn)





class DecoderLayer(EncoderLayer):

    def __init__(self, dim, heads, dim_head):
        super().__init__(dim, heads, dim_head)
        
        # decoder encoder attention
        self.dec_enc_attn = FastAttention(dim_head)
    
    
    def forward(self, src, ctx, pos, query):
        """
        decodes the context
        tgt: target to be populated
        pos: target position embedding (query embedding)
        """

        mh_sattn = self.sattn()