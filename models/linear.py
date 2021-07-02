import torch
from torch import nn
import torch.nn.functional as F

""""
Copy paste from  facebookresearch/detr
"""


class MLP(nn.Module):
    """
    Very simple multi-layer perceptron (also called FFN)
    with dropout
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.0):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = (
                F.dropout(F.relu(layer(x)), p=self.dropout)
                if i < self.num_layers - 1
                else layer(x)
            )
        return x
