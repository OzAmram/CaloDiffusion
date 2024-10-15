
import sys
sys.path.append("../..")
sys.path.append("..")
from utils import *

hgcal = True

dataset = "../data/HGCal_showers_william_v2/HGCal_showers1.h5"

#dataset_num = 101
shape_pad = [-1,1, 28,1988]
dataset_num = 111
shape_final = [-1,1, 28,12,21]
embed = True
max_ncell = 1988
max_dep = 1.0
ecut = 0.0000001
orig_shape = False
logE = False
e_key = 'gen_info'
emax = [100, 2.01, 1.572],
emin = [50, 1.99, 1.57],
norm_fac = 100.

geom_file = '/home/oamram/CaloDiffusion/HGCalShowers/geom_william.pkl'

showerMap = 'layer-logit-norm'
nevts = 1000



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
)

print(data_.shape)
print(layers.shape)

if(torch.cuda.is_available()): device = torch.device('cuda')
else: device = torch.device('cpu')

print(shape_pad)
data = np.reshape(data_,shape_pad)

if(embed):
    NN_embed = HGCalConverter(bins = shape_final, geom_file = geom_file, device = device ).to(device = device)
    NN_embed.init()

    tdata = torch.tensor(data)
    
    data_enc = apply_in_batches(NN_embed.enc, tdata, device=device)
    data_dec  = apply_in_batches(NN_embed.dec, data_enc, device=device).detach().cpu().numpy()
    data_enc = data_enc.detach().cpu().numpy()

else:
    data_dec = data



data_rev, e_rev = ReverseNorm( data_dec, e_, shape_pad, layerE = layers, emax = emax,emin = emin, 
    max_deposit=max_dep, #noise can generate more deposited energy than generated
    logE=logE,
    showerMap = showerMap,
    dataset_num = dataset_num,
    ecut = ecut,
    orig_shape = orig_shape,
    hgcal = hgcal,
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
    data_rev = np.reshape(data_rev, shape_pad)
    raw_shower = np.reshape(raw_shower, shape_pad)
    layer_rev = np.sum(data_rev, (3))
    raw_layer = np.sum(raw_shower, (3))
    print(data_rev.shape)
    print("AVG LAYER DIFF: ", np.mean( layer_rev[:100] - raw_layer[:100]))

print("REVERSED: \n")
#print(data_rev[0,0,10])
print("AVG DIFF: ", torch.mean(torch.tensor(data_rev[:1000]) - raw_shower[:1000]))
