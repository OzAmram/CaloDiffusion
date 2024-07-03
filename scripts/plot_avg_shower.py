from utils import *
import argparse
import os
import h5py
import numpy as np


parser = argparse.ArgumentParser()

parser.add_argument('-g', '--geom_file', default='/home/oamram/CaloDiffusion/HGCalShowers/geom.pkl', help='Geom file')
parser.add_argument('-c', '--config', default='', help='Geom file')
parser.add_argument('-i', '--fin', default='', help='File with showers to plot')
parser.add_argument('-n', '--nShowers', default=1, type = int, help='How many showers to plot')
parser.add_argument('-o', '--outdir', default='../plots/showers/', help='Where to put output plots')
flags = parser.parse_args()

dataset_config = LoadJson(flags.config)
hgcal = dataset_config.get('HGCAL', False)

if (not os.path.exists(flags.outdir)): os.system("mkdir " + flags.outdir)
f = h5py.File(flags.fin)

nShowers = max(flags.nShowers, 10000)
showers = f['showers'][:nShowers]


if(hgcal):
    print("Embedding HGCal")
    vmin = 1e-4
    shape_plot = dataset_config['SHAPE_FINAL']
    showers *= 200
    NN_embed = HGCalConverter(bins = shape_plot, geom_file = dataset_config['BIN_FILE'])
    showers = showers[:,:, :NN_embed.geom.max_ncell]
    showers =torch.from_numpy(showers.astype(np.float32)).reshape(dataset_config['SHAPE_PAD'])
    norm_before = torch.sum(showers)
    showers = np.squeeze(NN_embed.enc(showers).detach().numpy())
    norm_after = np.sum(showers)
    print("norm before %f, after %f" % (norm_before, norm_after))

    plt.figure(figsize=(5,5))
    plt.matshow(NN_embed.enc_mat[0])
    plt.savefig(flags.outdir + "mat.png")

    plt.figure(figsize=(5,5))
    plt.matshow(NN_embed.enc_mat[0][:, :10])
    plt.savefig(flags.outdir + "mat_zoom.png")

else:
    vmin = 1e-2
    showers = np.squeeze(np.reshape(showers, dataset_config['SHAPE_PAD']))


#avg showers
avg_shower = np.mean(showers, axis=0)
print("Avg shower shape" , avg_shower.shape)

vmax = np.amax(avg_shower)

print("avg shower")
for ilay in range(avg_shower.shape[0]):
    lay = avg_shower[ilay]
    plot_shower_layer(lay, vmin = vmin, vmax = vmax, title = "Layer %i" % ilay, fname = flags.outdir + "avg_shower_lay%i.png" % ilay)




