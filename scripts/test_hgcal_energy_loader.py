import sys
sys.path.append("../..")
sys.path.append("..")
from utils import *

if(torch.cuda.is_available()): device = torch.device('cuda')
else: device = torch.device('cpu')

def save_fig(fname, fig, ax0):
    plt_exts = [".png", ".pdf", "_logy.png", "_logy.pdf"]
    for plt_ext in plt_exts:
        if('logy') in plt_ext: ax0.set_yscale("log")
        else: ax0.set_yscale("linear")
        fig.savefig(fname + plt_ext)

def RadialEnergyHGCal(data_dict, NN_embed, plot_folder, label=''):

    def GetWidth(mean,mean2):
        width = np.ma.sqrt(mean2-mean**2).filled(0)
        return width

    r_vals = NN_embed.geom.ring_map[:, :NN_embed.geom.max_ncell]

     

    feed_dict = {}
    for key in data_dict:
        d = np.squeeze(data_dict[key])
        nrings = NN_embed.geom.nrings
        r_bins = np.zeros((d.shape[0], nrings))
        for i in range(nrings):
            mask = (r_vals == i)
            r_bins[:,i] = np.sum(d * mask,axis = (1,2))

        feed_dict[key] = r_bins


    fig,ax0 = PlotRoutine(feed_dict, xlabel='R-bin', ylabel= 'Avg. Energy', logy=True)
    save_fig('{}/EnergyR{}'.format(plot_folder, label), fig, ax0)

    return feed_dict


hgcal = True

dataset = "../data/HGCal_showers_william_v2/HGCal_showers1.h5"

shape_orig = [-1,28, 1988]
#dataset_num = 101
#shape_pad = [-1,1, 28,1988]
pre_embed = True
dataset_num = 111
shape_pad = [-1,1, 28,12,21]

shape_final = [-1,1, 28,12,21]
max_ncell = 1988
max_dep = 1.0
ecut = 0.0
orig_shape = False
logE = False
e_key = 'gen_info'
emax = [100, 2.01, 1.572],
emin = [50, 1.99, 1.57],
norm_fac = 200.

geom_file = '/home/oamram/CaloDiffusion/HGCalShowers/geom_william.pkl'

showerMap = 'layer-logit-norm'
nevts = 10000

plot_folder = "../plots/hgcal_energy_loader_pre_embed_test/"

NN_embed = HGCalConverter(bins = shape_final, geom_file = geom_file, device = device ).to(device = device)
NN_embed.init()




with h5.File(dataset,"r") as h5f:
    gen = h5f[e_key][0:int(nevts)].astype(np.float32)
    gen[:,0] *= norm_fac
    raw_shower = h5f['showers'][0:int(nevts)].astype(np.float32) * norm_fac
    raw_shower = raw_shower[:,:,:max_ncell]

data_,e_,layers = DataLoader( dataset, shape_pad, emax = emax,emin = emin, nevts = nevts,
    max_deposit=max_dep, 
    logE=logE,
    showerMap = showerMap,
    dataset_num = dataset_num,
    ecut = ecut,
    orig_shape = orig_shape,
    hgcal= hgcal,
    max_cells = max_ncell,
    embed = pre_embed,
    NN_embed = NN_embed,
)

print(data_.shape)
print(layers.shape)


print(shape_pad)
data = np.reshape(data_,shape_pad)

if(not pre_embed):
    tdata = torch.tensor(data)
    
    data_enc = apply_in_batches(NN_embed.enc, tdata, device=device)
    data_dec  = apply_in_batches(NN_embed.dec, data_enc, device=device).detach().cpu().numpy()
    data_enc = data_enc.detach().cpu().numpy()

else:
    data_enc = data_dec = data



data_rev, e_rev = ReverseNorm( data_dec, e_, shape_pad, layerE = layers, emax = emax,emin = emin, 
    max_deposit=max_dep, #noise can generate more deposited energy than generated
    logE=logE,
    showerMap = showerMap,
    dataset_num = dataset_num,
    ecut = ecut,
    orig_shape = orig_shape,
    hgcal = hgcal,

    embed = pre_embed,
    NN_embed = NN_embed,
)




data_rev[data_rev < ecut] = 0.

print("ShowerMap %s" % showerMap)
print("RAW: \n")
#print(raw_shower[0,0,10])
print("PREPROCESSED: \n")
#print(data_[0,0,10])
mean = np.mean(data_enc)
std = np.std(data_enc)
maxE = np.amax(data_enc)
minE = np.amin(data_enc)

maxEn = np.amax(e_)
minEn = np.amin(e_)
print("MEAN: %.4f STD : %.5f"  % (mean, std))
print("MAX: %.4f MIN : %.5f"  % (maxE, minE))
print("maxE %.2f minE %.2f" % (maxEn, minEn))

if(layers is not None):
    totalE,layers_ = layers[:,0], layers[:,1:]
    print("TotalE MEAN %.4f, STD: %.5f" % (np.mean(totalE), np.std(totalE)))
    print("LAYERS MEAN %.4f, STD: %.5f" % (np.mean(layers_), np.std(layers_)))
print("REVERSED: \n")
print("AVG DIFF: ", np.mean( data_rev[:100] - raw_shower[:100]))

if(not orig_shape):
    data_rev = np.reshape(data_rev, shape_orig)
    raw_shower = np.reshape(raw_shower, shape_orig)
    layer_rev = np.sum(data_rev, (2))
    raw_layer = np.sum(raw_shower, (2))
    print(data_rev.shape)
    print("AVG LAYER DIFF: ", np.mean( layer_rev[:100] - raw_layer[:100]))

print("REVERSED: \n")
#print(data_rev[0,0,10])
print("AVG DIFF: ", torch.mean(torch.tensor(data_rev[:1000]) - raw_shower[:1000]))



data_dict = {}
data_dict['Geant4'] = raw_shower
data_dict['Embed- Pre-process - ReverseNorm - Decode'] = data_rev
RadialEnergyHGCal(data_dict,NN_embed, plot_folder )

layers = [3, 10, 24]
geo = NN_embed.geom

avg_shower_before = np.squeeze(np.mean(raw_shower, axis = 0))
avg_shower_after = np.squeeze(np.mean(data_rev, axis = 0))

avg_shower_ratio = avg_shower_after / avg_shower_before

for ilay in layers:
    print(ilay)

    ncells = int(round(geo.ncells[ilay]))
    plot_shower_hex(geo.xmap[ilay][:ncells], geo.ymap[ilay][:ncells], avg_shower_before[ilay][:ncells] , log_scale=False, nrings = geo.nrings, fout = plot_folder + "avg_shower_lay%i_before.png" % (ilay))
    plot_shower_hex(geo.xmap[ilay][:ncells], geo.ymap[ilay][:ncells], avg_shower_after[ilay][:ncells] , log_scale=False, nrings = geo.nrings, fout = plot_folder + "avg_shower_lay%i_after.png" % (ilay))
    plot_shower_hex(geo.xmap[ilay][:ncells], geo.ymap[ilay][:ncells], avg_shower_ratio[ilay][:ncells] , log_scale=False, nrings = geo.nrings, fout = plot_folder + "avg_shower_lay%i_ratio.png" % (ilay))


