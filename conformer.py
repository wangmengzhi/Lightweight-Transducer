import torch
from torch import nn, einsum
import torch.nn.functional as F
import math
from math import pi, log

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from init import init_with_xavier_uniform,init_with_lecun_normal,init_with_uniform

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# helper classes

class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()

class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()

class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups = chan_in)
        for n, p in self.named_parameters():
            init_with_xavier_uniform(n, p)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)

# attention, feedforward, and conv module

class Scale(nn.Module):
    def __init__(self, scale, fn):
        super().__init__()
        self.fn = fn
        self.scale = scale

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale
    
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

# classes
    
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_qk = 64,
        dim_v = 64,
        dropout = 0.,
        max_pos_emb = 10,
        mask = None,
    ):
        super().__init__()
        inner_dim = dim_v * heads
        self.inner_dim=inner_dim
        self.heads= heads
        self.scale = dim_qk ** -0.5
        self.to_qk = nn.Linear(dim, dim_qk*heads*2, bias = False)
        self.to_v = nn.Linear(dim, dim, bias = False)
        self.to_out = nn.Linear(dim, dim)

        self.max_pos_emb = max_pos_emb
        self.conv=nn.Parameter(torch.Tensor(heads,2 * max_pos_emb + 1))
        self.dropout_att = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout,inplace=True)
        self.mask=mask
        nn.init.zeros_(self.conv)

    def forward(self, x, mask):
        b, n, device, h, max_pos_emb = x.shape[0], x.shape[-2], x.device, self.heads, self.max_pos_emb
        q,k= self.to_qk(x).chunk(2, dim = -1)
        v= self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        n=k.shape[2]
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        # shaw's relative positional embedding
        seq = torch.arange(n, device = device)
        dist = rearrange(seq, 'j -> () j')-rearrange(seq, 'i -> i ()')
        dist = dist.clip(-max_pos_emb, max_pos_emb) + max_pos_emb
        conv=torch.gather(self.conv.unsqueeze(1).repeat(1,n,1),index=dist.unsqueeze(0).repeat(h,1,1),dim=-1) #head*T*len head*T*T
        dots = dots+conv
        attn = dots.softmax(dim = -1)
        attn=self.dropout_att(attn)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return self.dropout(out)
    
class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        mult = 4,
        dropout = 0.,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            Swish(),
            nn.Dropout(dropout,inplace=True),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout,inplace=True)
        )
        for n, p in self.named_parameters():
            init_with_xavier_uniform(n, p)

    def forward(self, x):
        return self.net(x)

class ConformerConvModule(nn.Module):
    def __init__(
        self,
        dim,
        output_dim,
        lorder = 1,
        rorder = 1,
        dropout = 0.):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, output_dim * 2),
            Rearrange('b n c -> b c n'),
            GLU(dim=1),
            DepthWiseConv1d(output_dim, output_dim, kernel_size = lorder+rorder+1, padding = (lorder,rorder)),
            Rearrange('b c n -> b n c'),
            nn.LayerNorm((output_dim)),
            Swish(),
            nn.Linear(output_dim, output_dim),
            nn.Dropout(dropout,inplace=True)
        )

    def forward(self, x):
        return self.net(x)

# Conformer Block

class ConformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_qk=64,
        dim_v = 64,
        heads = 8,
        ff_mult = 4,
        lorder = 7,
        rorder = 7,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.,
        att_mask=None
    ):
        super().__init__()
        self.ff1 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        self.ff2 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))
        self.conv = ConformerConvModule(dim = dim, output_dim=heads*dim_v, lorder = lorder, rorder = rorder, dropout = conv_dropout)
        self.attn = Attention(dim = dim, dim_qk=dim_qk, dim_v = dim_v, heads = heads, dropout = attn_dropout, mask=att_mask)
        self.attn = PreNorm(dim, self.attn)
        self.post_norm = nn.LayerNorm(dim)
    def forward(self, x):
        x = self.ff1(x) + x
        x = self.attn(x,mask=None)+x
        x = self.conv(x)+x
        x =self.ff2(x)+x
        x = self.post_norm(x)
        return x
