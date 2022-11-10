import json, yaml
import os
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.ticker as mtick
from einops import rearrange


import torch
import torch.nn as nn
import torch.nn.functional as f

from inspect import isfunction
from functools import partial

def split_data_np(data, frac=0.8):
    np.random.shuffle(data)
    split = int(frac * data.shape[0])
    train_data =data[:split]
    test_data = data[split:]
    return train_data,test_data


def split_data(data,nevts,frac=0.8):
    data = data.shuffle(nevts)
    train_data = data.take(int(frac*nevts)).repeat()
    test_data = data.skip(int(frac*nevts)).repeat()
    return train_data,test_data

line_style = {
    'Geant4':'dotted',
    'CaloScore: VP':'-',
    'CaloScore: VE':'-',
    'CaloScore: subVP':'-',

    'VP 50 steps':'v',
    'VE 50 steps':'v',
    'subVP 50 steps':'v',

    'VP 500 steps':'^',
    'VE 500 steps':'^',
    'subVP 500 steps':'^',

    'VP r=0.0':'v',
    'VE r=0.0':'v',
    'subVP r=0.0':'v',

    'VP r=0.3':'^',
    'VE r=0.3':'^',
    'subVP r=0.3':'^',

    'Autoencoder' : '-',
    'Diffu' : '-',
    'LatentDiffu' : '-',

}

colors = {
    'Geant4':'black',
    'CaloScore: VP':'#7570b3',
    'CaloScore: VE':'#d95f02',
    'CaloScore: subVP':'#1b9e77',

    'VP 50 steps':'#e7298a',
    'VE 50 steps':'#e7298a',
    'subVP 50 steps':'#e7298a',

    'VP 500 steps':'#e7298a',
    'VE 500 steps':'#e7298a',
    'subVP 500 steps':'#e7298a',

    'VP r=0.0':'#e7298a',
    'VE r=0.0':'#e7298a',
    'subVP r=0.0':'#e7298a',

    'VP r=0.3':'#e7298a',
    'VE r=0.3':'#e7298a',
    'subVP r=0.3':'#e7298a',
    
    'Autoencoder': 'purple',
    'Diffu': 'blue',
    'LatentDiffu': 'green',

}

name_translate={

    'AE' : "Autoencoder",
    'Diffu' : "Diffu",
    'LatentDiffu' : "LatentDiffu",


    'VPSDE':'CaloScore: VP',
    'VESDE':'CaloScore: VE',
    'subVPSDE':'CaloScore: subVP',

    '50_VPSDE':'VP 50 steps',
    '50_VESDE':'VE 50 steps',
    '50_subVPSDE':'subVP 50 steps',

    '500_VPSDE':'VP 500 steps',
    '500_VESDE':'VE 500 steps',
    '500_subVPSDE':'subVP 500 steps',

    '0p0_VPSDE':'VP r=0.0',
    '0p0_VESDE':'VE r=0.0',
    '0p0_subVPSDE':'subVP r=0.0',

    '0p3_VPSDE':'VP r=0.3',
    '0p3_VESDE':'VE r=0.3',
    '0p3_subVPSDE':'subVP r=0.3',

    
    }

def SetStyle():
    from matplotlib import rc
    rc('text', usetex=True)

    import matplotlib as mpl
    rc('font', family='serif')
    rc('font', size=22)
    rc('xtick', labelsize=15)
    rc('ytick', labelsize=15)
    rc('legend', fontsize=15)

    # #
    mpl.rcParams.update({'font.size': 19})
    #mpl.rcParams.update({'legend.fontsize': 18})
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams.update({'xtick.labelsize': 18}) 
    mpl.rcParams.update({'ytick.labelsize': 18}) 
    mpl.rcParams.update({'axes.labelsize': 18}) 
    mpl.rcParams.update({'legend.frameon': False}) 
    mpl.rcParams.update({'lines.linewidth': 2})
    
    import matplotlib.pyplot as plt
    #import mplhep as hep
    #hep.set_style(hep.style.CMS)
    #hep.style.use("CMS") 

def SetGrid(ratio=True):
    fig = plt.figure(figsize=(9, 9))
    if ratio:
        gs = gridspec.GridSpec(2, 1, height_ratios=[3,1]) 
        gs.update(wspace=0.025, hspace=0.1)
    else:
        gs = gridspec.GridSpec(1, 1)
    return fig,gs



def PlotRoutine(feed_dict,xlabel='',ylabel='',reference_name='Geant4'):
    assert reference_name in feed_dict.keys(), "ERROR: Don't know the reference distribution"
    
    fig,gs = SetGrid() 
    ax0 = plt.subplot(gs[0])
    plt.xticks(fontsize=0)
    ax1 = plt.subplot(gs[1],sharex=ax0)

    for ip,plot in enumerate(feed_dict.keys()):
        if 'steps' in plot or 'r=' in plot:
            ax0.plot(np.mean(feed_dict[plot],0),label=plot,marker=line_style[plot],color=colors[plot],lw=0)
        else:
            ax0.plot(np.mean(feed_dict[plot],0),label=plot,linestyle=line_style[plot],color=colors[plot])
        if reference_name!=plot:
            eps = 1e-8
            ratio = 100*np.divide(np.mean(feed_dict[reference_name],0)-np.mean(feed_dict[plot],0),np.mean(feed_dict[reference_name],0) + eps)
            #ax1.plot(ratio,color=colors[plot],marker='o',ms=10,lw=0,markerfacecolor='none',markeredgewidth=3)
            if 'steps' in plot or 'r=' in plot:
                ax1.plot(ratio,color=colors[plot],markeredgewidth=1,marker=line_style[plot],lw=0)
            else:
                ax1.plot(ratio,color=colors[plot],linewidth=2,linestyle=line_style[plot])
                
        
    FormatFig(xlabel = "", ylabel = ylabel,ax0=ax0)
    ax0.legend(loc='best',fontsize=16,ncol=1)

    plt.ylabel('Difference. (%)')
    plt.xlabel(xlabel)
    plt.axhline(y=0.0, color='r', linestyle='--',linewidth=1)
    plt.axhline(y=10, color='r', linestyle='--',linewidth=1)
    plt.axhline(y=-10, color='r', linestyle='--',linewidth=1)
    plt.ylim([-50,50])

    return fig,ax0

class ScalarFormatterClass(mtick.ScalarFormatter):
    #https://www.tutorialspoint.com/show-decimal-places-and-scientific-notation-on-the-axis-of-a-matplotlib-plot
    def _set_format(self):
        self.format = "%1.2f"


def FormatFig(xlabel,ylabel,ax0):
    #Limit number of digits in ticks
    # y_loc, _ = plt.yticks()
    # y_update = ['%.1f' % y for y in y_loc]
    # plt.yticks(y_loc, y_update) 
    ax0.set_xlabel(xlabel,fontsize=20)
    ax0.set_ylabel(ylabel)
        

    # xposition = 0.9
    # yposition=1.03
    # text = 'H1'
    # WriteText(xposition,yposition,text,ax0)


def WriteText(xpos,ypos,text,ax0):

    plt.text(xpos, ypos,text,
             horizontalalignment='center',
             verticalalignment='center',
             transform = ax0.transAxes, fontsize=25, fontweight='bold')


def HistRoutine(feed_dict,xlabel='',ylabel='',reference_name='Geant4',logy=False,binning=None,label_loc='best'):
    assert reference_name in feed_dict.keys(), "ERROR: Don't know the reference distribution"
    
    fig,gs = SetGrid() 
    ax0 = plt.subplot(gs[0])
    plt.xticks(fontsize=0)
    ax1 = plt.subplot(gs[1],sharex=ax0)

    
    if binning is None:
        binning = np.linspace(np.quantile(feed_dict[reference_name],0.0),np.quantile(feed_dict[reference_name],1),10)
        
    xaxis = [(binning[i] + binning[i+1])/2.0 for i in range(len(binning)-1)]
    reference_hist,_ = np.histogram(feed_dict[reference_name],bins=binning,density=True)
    
    for ip,plot in enumerate(feed_dict.keys()):
        if 'steps' in plot or 'r=' in plot:
            dist,_ = np.histogram(feed_dict[plot],bins=binning,density=True)
            ax0.plot(xaxis,dist,color=colors[plot],marker=line_style[plot],ms=10,lw=0,markeredgewidth=3,label=plot)
            #dist,_,_=ax0.hist(feed_dict[plot],bins=binning,label=plot,marker=line_style[plot],color=colors[plot],density=True,histtype="step")
        else:
            dist,_,_=ax0.hist(feed_dict[plot],bins=binning,label=plot,linestyle=line_style[plot],color=colors[plot],density=True,histtype="step")
            
        if reference_name!=plot:
            eps = 1e-8
            ratio = 100*np.divide(reference_hist-dist,reference_hist + eps)
            if 'steps' in plot or 'r=' in plot:
                ax1.plot(xaxis,ratio,color=colors[plot],marker=line_style[plot],ms=10,lw=0,markeredgewidth=3)
            else:
                ax1.plot(xaxis,ratio,color=colors[plot],marker='o',ms=10,lw=0,markerfacecolor='none',markeredgewidth=3)
        
    ax0.legend(loc=label_loc,fontsize=16,ncol=1)        
    FormatFig(xlabel = "", ylabel = ylabel,ax0=ax0) 

    if logy:
        ax0.set_yscale('log')
    
    plt.ylabel('Difference. (%)')
    plt.xlabel(xlabel)
    plt.axhline(y=0.0, color='r', linestyle='-',linewidth=1)
    plt.axhline(y=10, color='r', linestyle='--',linewidth=1)
    plt.axhline(y=-10, color='r', linestyle='--',linewidth=1)
    plt.ylim([-50,50])

    return fig,ax0



def EnergyLoader(file_name,nevts,emax,emin,rank=0,logE=True):
    
    with h5.File(file_name,"r") as h5f:
        e = h5f['incident_energies'][rank*int(nevts):(rank+1)*int(nevts)].astype(np.float32)/1000.0
        shower = h5f['showers'][rank*int(nevts):(rank+1)*int(nevts)].astype(np.float32)/1000.0

    if logE:        
        return np.log10(e/emin)/np.log10(emax/emin)
    else:
        return (e-emin)/(emax-emin)
        



def DataLoader(file_name,shape,nevts,emax,emin,max_deposit=2,logE=True,norm_data=False, showerMap = 'log'):
    #rank = hvd.rank()
    #size = hvd.size()
    rank = 0
    size = 1

    with h5.File(file_name,"r") as h5f:
        e = h5f['incident_energies'][rank:int(nevts):size].astype(np.float32)/1000.0
        shower = h5f['showers'][rank:int(nevts):size].astype(np.float32)/1000.0

        
    shower = np.reshape(shower,(shower.shape[0],-1))
    shower = shower/(max_deposit*e)

    if norm_data:
        #Normalize voxels to 1 and learn the normalization factor as a dimension
        def NormLayer(shower,shape):
            shower_padded = np.zeros([shower.shape[0],shape[1]],dtype=np.float32)
            deposited_energy = np.sum(shower,-1,keepdims=True)
            shower = np.ma.divide(shower,np.sum(shower,-1,keepdims=True)).filled(0)
            shower = np.concatenate((shower,deposited_energy),-1)        
            shower_padded[:,:shower.shape[1]] += shower
            return shower_padded
        
        shower = NormLayer(shower,shape)
    shower = shower.reshape(shape)
    
    if(showerMap == 'log'):
        alpha = 1e-6
        x = alpha + (1 - 2*alpha)*shower
        shower = np.ma.log(x/(1-x)).filled(0)    
    elif(showerMap == 'sqrt'):
        shower = np.sqrt(shower)

    if logE:        
        return shower,np.log10(e/emin)/np.log10(emax/emin)
    else:
        return shower,(e-emin)/(emax-emin)
        

def LoadJson(file_name):
    import json,yaml
    JSONPATH = os.path.join(file_name)
    return yaml.safe_load(open(JSONPATH))


def ReverseNorm(voxels,e,shape,emax,emin,max_deposit,logE=True,norm_data=False, showerMap ='log'):
    '''Revert the transformations applied to the training set'''
    #shape=voxels.shape
    alpha = 1e-6
    if logE:
        energy = emin*(emax/emin)**e
    else:
        energy = emin + (emax-emin)*e
        
    if(showerMap == 'log'):
        exp = np.exp(voxels)    
        x = exp/(1+exp)
        data = (x-alpha)/(1 - 2*alpha)
    elif(showerMap == 'sqrt'):
        data = np.square(voxels)

    if norm_data:
        def ApplyNorm(data):
            energies = data[:,shape[1]:shape[1]+1]
            shower = data[:,:shape[1]]
            shower = shower/np.sum(shower,-2,keepdims=True)
            shower *=energies
            return shower

        data = ApplyNorm(data)
    data = data.reshape(voxels.shape[0],-1)*max_deposit*energy.reshape(-1,1)
    
    return data,energy
    

def polar_to_cart(polar_data,nr=9,nalpha=16,nx=12,ny=12):
    cart_img = np.zeros((nx,ny))
    ntotal = 0
    nfilled = 0

    if not hasattr(polar_to_cart, "x_interval"):
        polar_to_cart.x_interval = np.linspace(-1,1,nx)
        polar_to_cart.y_interval = np.linspace(-1,1,ny)
        polar_to_cart.alpha_pos = np.linspace(1e-5,1,nalpha)
        polar_to_cart.r_pos = np.linspace(1e-5,1,nr)

    for alpha in range(nalpha):
        for r in range(nr):
            if(polar_data[alpha,r] > 0):
                x = polar_to_cart.r_pos[r] * np.cos(polar_to_cart.alpha_pos[alpha]*np.pi*2)
                y = polar_to_cart.r_pos[r] * np.sin(polar_to_cart.alpha_pos[alpha]*np.pi*2)
                binx = np.argmax(polar_to_cart.x_interval>x)
                biny = np.argmax(polar_to_cart.y_interval>y)
                ntotal+=1
                if cart_img[binx,biny] >0:
                    nfilled+=1
                cart_img[binx,biny]+=polar_data[alpha,r]
    return cart_img

#some pytorch modules & useful functions
def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False



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
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, kernel_size = 3, padding = 1)
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
    
    def __init__(self, dim, dim_out, *, cond_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(cond_emb_dim, dim_out))
            if exists(cond_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv3d(dim, dim_out, kernel_size = 1) if dim != dim_out else nn.Identity()

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

    def __init__(self, dim, dim_out, *, cond_emb_dim=None, mult=2, norm=True):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(cond_emb_dim, dim))
            if exists(cond_emb_dim)
            else None
        )

        self.ds_conv = nn.Conv3d(dim, dim, kernel_size = 7, padding=3, groups=dim)

        self.net = nn.Sequential(
            nn.GroupNorm(1, dim) if norm else nn.Identity(),
            nn.Conv3d(dim, dim_out * mult, kernel_size = 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, dim_out * mult),
            nn.Conv3d(dim_out * mult, dim_out, kernel_size =3, padding=1),
        )

        self.res_conv = nn.Conv3d(dim, dim_out, kernel_size = 1) if dim != dim_out else nn.Identity()

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


#downsample in two dimmensions but keep same dimm in first
def Upsample(dim):
    #return nn.ConvTranspose3d(dim, dim, kernel_size = (4,4,4), stride = (1,2,2), padding = (0,1,1))
    return nn.Upsample(scale_factor = (1,2,2))

def Downsample(dim):
    #return nn.Conv3d(dim, dim, kernel_size = (4,4,4), stride = (1,2,2), padding = (0,1,1))
    return nn.AvgPool3d(kernel_size = (1,2,2), stride = (1,2,2), padding =0)


class CondUnet(nn.Module):
#Unet with conditional layers
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=1,
        resnet_block_groups=8,
        use_convnext=False,
        use_attn = False,
        convnext_mult=2,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels
        self.use_attn = use_attn

        #init_dim = default(init_dim, dim // 3 * 2)
        #self.init_conv = nn.Conv3d(channels, init_dim, kernel_size =7, padding=3)

        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:])) 
        
        if use_convnext:
            block_klass = partial(ConvNextBlock, mult=convnext_mult)
        else:
            block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        time_dim = dim * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.cond_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        cond_dim = time_dim*2

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)



        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, cond_emb_dim=cond_dim),
                        block_klass(dim_out, dim_out, cond_emb_dim=cond_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, cond_emb_dim=cond_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, cond_emb_dim=cond_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in, cond_emb_dim=cond_dim),
                        block_klass(dim_in, dim_in, cond_emb_dim=cond_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim, dim), nn.Conv3d(dim, out_dim, 1)
        )

    def forward(self, x, cond, time):

        #x = self.init_conv(x)

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

def SetFig(xlabel,ylabel):
    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(1, 1) 
    ax0 = plt.subplot(gs[0])
    ax0.yaxis.set_ticks_position('both')
    ax0.xaxis.set_ticks_position('both')
    ax0.tick_params(direction="in",which="both")    
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel(xlabel,fontsize=20)
    plt.ylabel(ylabel,fontsize=20)

    ax0.minorticks_on()
    return fig, ax0


if __name__ == "__main__":
    #Preprocessing of the input files: conversion to cartesian coordinates + zero-padded mask generation
    file_path = '/wclustre/cms/denoise/CaloChallenge/dataset_2_2.hdf5'
    #with h5.File(file_path,"r") as h5f:
    #    e = h5f['incident_energies'][:]
    #    showers = h5f['showers'][:]
    ##shape = [-1,45,50,18,1]
    ##nx=32
    ##ny=32
    #
    #shape = [-1,45,16,9,1]

    #nx = 12
    #ny = 12

    #showers = showers.reshape((-1,shape[2],shape[3]))
    #cart_data = []
    #for ish,shower in enumerate(showers):
    #    if ish%(10000)==0:print(ish)
    #    cart_data.append(polar_to_cart(shower,nr=shape[3],nalpha=shape[2],nx=nx,ny=ny))

    #cart_data = np.reshape(cart_data,(-1,45,nx,ny))
    #with h5.File('/wclustre/cms/denoise/CaloChallenge/dataset_2_2_cart.hdf5',"w") as h5f:
    #    dset = h5f.create_dataset("showers", data=cart_data)
    #    dset = h5f.create_dataset("incident_energies", data=e)


    file_path = '/wclustre/cms/denoise/CaloChallenge/dataset_2_1_cart.hdf5'
    ##file_path='/wclustre/cms/denoise/CaloChallenge/dataset_1_photons_1.hdf5'
    with h5.File(file_path,"r") as h5f:
        showers = h5f['showers'][:]
        energies = h5f['incident_energies'][:]
    mask = np.sum(showers,0)==0
    mask_file = file_path.replace('.hdf5','_mask.hdf5')
    print("Creating mask file %s " % mask_file)
    with h5.File(mask_file,"w") as h5f:
        dset = h5f.create_dataset("mask", data=mask)
    
