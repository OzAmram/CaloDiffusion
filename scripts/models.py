#some pytorch modules & useful functions
from einops import rearrange
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from inspect import isfunction
from functools import partial

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class CylindricalConv(nn.Module):
    #assumes format of channels,zbin,phi_bin,rbin
    def __init__(self, dim_in, dim_out, kernel_size = 3, stride=1, groups = 1, padding = 0, bias = True):
        super().__init__()
        if(type(padding) != int):
            self.padding_orig = copy.copy(padding)
            padding = list(padding)
            padding[1] = 0
        else:
            padding = [padding]*3
            self.padding_orig = copy.copy(padding)
            padding[1] = 0
        self.kernel_size = kernel_size
        self.conv = nn.Conv3d(dim_in, dim_out, kernel_size=kernel_size, stride = stride, groups = groups, padding = padding, bias = bias)

    def forward(self, x):
        #to achieve 'same' use padding P = ((S-1)*W-S+F)/2, with F = filter size, S = stride, W = input size
        #pad last dim with nothing, 2nd to last dim is circular one
        circ_pad = self.padding_orig[1]
        x = F.pad(x, pad = (0,0, circ_pad, circ_pad, 0, 0), mode = 'circular')
        x = self.conv(x)
        return x

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings  


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8, cylindrical = False):
        super().__init__()
        if(not cylindrical): self.proj = nn.Conv3d(dim, dim_out, kernel_size = 3, padding = 1)
        else:  self.proj = CylindricalConv(dim, dim_out, kernel_size = 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""
    
    def __init__(self, dim, dim_out, *, cond_emb_dim=None, groups=8, cylindrical = False):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(cond_emb_dim, dim_out))
            if exists(cond_emb_dim)
            else None
        )

        conv = CylindricalConv(dim, dim_out, kernel_size = 1) if cylindrical else nn.Conv3d(dim, dim_out, kernel_size = 1)
        self.block1 = Block(dim, dim_out, groups=groups, cylindrical = cylindrical)
        self.block2 = Block(dim_out, dim_out, groups=groups, cylindrical = cylindrical)
        self.res_conv = conv if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)

        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1 1")
            h = h + time_emb

        h = self.block2(h)
        return h + self.res_conv(x)

class ConvNextBlock(nn.Module):
    """https://arxiv.org/abs/2201.03545"""

    def __init__(self, dim, dim_out, *, cond_emb_dim=None, mult=2, norm=True, cylindrical = False):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(cond_emb_dim, dim))
            if exists(cond_emb_dim)
            else None
        )

        if(not cylindrical): conv_op = nn.Conv3d
        else: conv_op = CylindricalConv

        self.ds_conv = conv_op(dim, dim, kernel_size = 7, padding=3, groups=dim)

        self.net = nn.Sequential(
            nn.GroupNorm(1, dim) if norm else nn.Identity(),
            conv_op(dim, dim_out * mult, kernel_size = 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, dim_out * mult),
            conv_op(dim_out * mult, dim_out, kernel_size =3, padding=1),
        )

        self.res_conv = conv_op(dim, dim_out, kernel_size = 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.ds_conv(x)

        if exists(self.mlp) and exists(time_emb):
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, "b c -> b c 1 1")

        h = self.net(h)
        return h + self.res_conv(x)

#TODO Fix to work in 3D
class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)

#TODO Fix to work in 3D
class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), 
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


#up and down sample in 2 dims but keep z dimm

def Upsample(dim, extra_upsample = (0,0,0), cylindrical = False):
    #TODO make cylindrical ? 
    return nn.ConvTranspose3d(dim, dim, kernel_size = (3,4,4), stride = (1,2,2), padding = 1, output_padding = extra_upsample)
    #return nn.Upsample(scale_factor = (1,2,2))

def Downsample(dim, cylindrical = False):
    if(cylindrical): return CylindricalConv(dim, dim, kernel_size = (3,4,4), stride = (1,2,2), padding = 1)
    else: return nn.Conv3d(dim, dim, kernel_size = (3,4,4), stride = (1,2,2), padding = 1)
    #return nn.AvgPool3d(kernel_size = (1,2,2), stride = (1,2,2), padding =0)


class CondUnet(nn.Module):
#Unet with conditional layers
    def __init__(
        self,
        out_dim=1,
        layer_sizes = None,
        channels=1,
        cond_dim = 64,
        resnet_block_groups=8,
        use_convnext=False,
        use_attn = False,
        convnext_mult=2,
        cylindrical = False,
        data_shape = (-1,1,45, 16,9),
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels
        self.use_attn = use_attn



        #dims = [channels, *map(lambda m: dim * m, dim_mults)]
        #layer_sizes.insert(0, channels)
        in_out = list(zip(layer_sizes[:-1], layer_sizes[1:])) 
        #print(in_out)
        
        if(not cylindrical): self.init_conv = nn.Conv3d(channels, layer_sizes[0], kernel_size = 3, padding = 1)
        else: self.init_conv = CylindricalConv(channels, layer_sizes[0], kernel_size = 3, padding = 1)

        if use_convnext:
            block_klass = partial(ConvNextBlock, mult=convnext_mult, cylindrical = cylindrical)
        else:
            block_klass = partial(ResnetBlock, groups=resnet_block_groups, cylindrical = cylindrical)

        # time and energy embeddings
        time_dim = cond_dim // 2


        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim//2),
            nn.Linear(time_dim//2, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.cond_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim//2),
            nn.Linear(time_dim//2, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )


        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        self.extra_upsamples = []
        num_resolutions = len(in_out)

        cur_data_shape = data_shape[-3:]

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            if(not is_last):
                extra_upsample_dim = (0, cur_data_shape[1]%2, cur_data_shape[2]%2)
                cur_data_shape = (cur_data_shape[0], cur_data_shape[1] // 2, cur_data_shape[2] //2)
                self.extra_upsamples.append(extra_upsample_dim)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, cond_emb_dim=cond_dim),
                        block_klass(dim_out, dim_out, cond_emb_dim=cond_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out, cylindrical) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = layer_sizes[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, cond_emb_dim=cond_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, cond_emb_dim=cond_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            if(not is_last): 
                extra_upsample = self.extra_upsamples.pop()


            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in, cond_emb_dim=cond_dim),
                        block_klass(dim_in, dim_in, cond_emb_dim=cond_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in, extra_upsample, cylindrical) if not is_last else nn.Identity(),
                    ]
                )
            )

        if(not cylindrical): final_lay = nn.Conv3d(layer_sizes[0], out_dim, 1)
        else:  final_lay = CylindricalConv(layer_sizes[0], out_dim, 1)
        self.final_conv = nn.Sequential( block_klass(layer_sizes[1], layer_sizes[0]),  final_lay )

    def forward(self, x, cond, time):

        x = self.init_conv(x)

        t = self.time_mlp(time)
        c = self.cond_mlp(cond)
        conditions = torch.cat([t,c], axis = -1)


        h = []

        # downsample
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, conditions)
            x = block2(x, conditions)
            if(self.use_attn): x = attn(x)
            h.append(x)
            x = downsample(x)

        # bottleneck
        x = self.mid_block1(x, conditions)
        if(self.use_attn): x = self.mid_attn(x)
        x = self.mid_block2(x, conditions)

        # upsample
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, conditions)
            x = block2(x, conditions)
            if(self.use_attn): x = attn(x)
            x = upsample(x)

        return self.final_conv(x)
