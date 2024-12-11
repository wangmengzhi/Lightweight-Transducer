import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class RotaryEmbedding:
    def __init__(self, dim, max_len=2048):
        freqs = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        freqs = torch.arange(max_len, dtype=torch.float32).unsqueeze(-1)*freqs
        freqs = repeat(freqs, '... n -> ... (n r)', r = 2)
        self.cos = freqs.cos()
        self.sin = freqs.sin()

    def __call__(self, x):
        seq_len, device = x.shape[-2], x.device
        cos, sin = self.cos[:seq_len].to(device), self.sin[:seq_len].to(device)
        x2 = torch.stack((-x[...,1::2], x[...,::2]), dim = -1)
        x2 = rearrange(x2, '... d r -> ... (d r)')
        return x * cos + x2 * sin

pos_emb = RotaryEmbedding(dim = 64)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_qkv = 64,
        dropout = 0.,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.heads = dim//dim_qkv
        self.qkv = nn.Linear(dim, dim*3, bias = False)
        self.out = nn.Linear(dim, dim)
        self.dropout_p = dropout
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, x):
        x = self.norm(x)
        q, k, v = self.qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))
        q = pos_emb(q)
        k = pos_emb(k)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_p if self.training else 0)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.out(out)
        return self.dropout(out)
    
class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        dim_ff = 2048,
        dropout = 0.,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim_ff),
            nn.SiLU(),
            nn.Dropout(dropout,inplace=True),
            nn.Linear(dim_ff, dim),
            nn.Dropout(dropout,inplace=True)
        )

    def forward(self, x):
        return self.net(x)

class ConformerConvModule(nn.Module):
    def __init__(
        self,
        dim,
        kernel_size = 15,
        dropout = 0.):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            Rearrange('b n c -> b c n'),
            nn.GLU(dim=1),
            nn.Conv1d(dim, dim, kernel_size, padding=(kernel_size-1)//2, groups = dim),
            Rearrange('b c n -> b n c'),
            nn.LayerNorm(dim),
            nn.SiLU(inplace=True),
            nn.Linear(dim, dim),
            nn.Dropout(dropout,inplace=True)
        )

    def forward(self, x):
        return self.net(x)

class ConformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_qkv = 64,
        dim_ff = 2048,
        kernel_size = 15,
        dropout = 0.
    ):
        super().__init__()
        self.ff1 = FeedForward(dim = dim, dim_ff = dim_ff, dropout = dropout)
        self.ff2 = FeedForward(dim = dim, dim_ff = dim_ff, dropout = dropout)
        self.conv = ConformerConvModule(dim = dim, kernel_size = kernel_size, dropout = dropout)
        self.attn = Attention(dim = dim, dim_qkv=dim_qkv, dropout = dropout)
        self.post_norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        x = self.ff1(x)*0.5+x
        x = self.attn(x)+x
        x = self.conv(x)+x
        x = self.ff2(x)*0.5+x
        x = self.post_norm(x)
        return x
