from utils import *
import argparse
import os
import h5py
import numpy as np


parser = argparse.ArgumentParser()

parser.add_argument('-c', '--config', default='', help='Geom file')
parser.add_argument('-i', '--fin', default='', help='File with showers to plot')
parser.add_argument('-n', '--nShowers', default=1, type = int, help='How many showers to plot')
parser.add_argument('-o', '--outdir', default='../plots/showers/', help='Where to put output plots')
parser.add_argument('--EMin', type = float, default=-1.0, help='Voxel min energy')
flags = parser.parse_args()

dataset_config = LoadJson(flags.config)
hgcal = dataset_config.get('HGCAL', False)

if (not os.path.exists(flags.outdir)): os.system("mkdir " + flags.outdir)
f = h5py.File(flags.fin)

scale_fac = 200. if hgcal else (1/1000.)
nShowers = max(flags.nShowers, 10000)

showers = f['showers'][:nShowers] * scale_fac


if(flags.EMin > 0.):
    mask = showers < flags.EMin
    showers[mask] = 0.



if(hgcal):
    print("Embedding HGCal")
    vmin = 1e-4
    shape_plot = dataset_config['SHAPE_FINAL']
    NN_embed = HGCalConverter(bins = shape_plot, geom_file = dataset_config['BIN_FILE'])
    trainable = dataset_config.get('TRAINABLE_EMBED', False)
    if(not trainable): NN_embed.init()


    showers = showers[:,:, :NN_embed.geom.max_ncell]
    showers =torch.from_numpy(showers.astype(np.float32)).reshape(dataset_config['SHAPE_PAD'])
    norm_before = torch.sum(showers)
    showers = torch.squeeze(NN_embed.enc(showers)).detach().numpy()
    norm_after = np.sum(showers)
    print("norm before %f, after %f" % (norm_before, norm_after))

    enc_mat = NN_embed.enc_mat.detach().numpy()
    print('enc map', NN_embed.enc_mat.shape)
    num_alpha_bins = shape_plot[-2]
    num_r_bins = shape_plot[-1]
    enc_mat_reshape = np.reshape(enc_mat, (enc_mat.shape[0], num_alpha_bins, num_r_bins, -1))
    print(' in r-alpha space', enc_mat_reshape.shape)
    print(enc_mat_reshape[0,:,0,0])
    print(enc_mat_reshape[0,:,1,1])

    eps = 1e-4
    for i in range(enc_mat.shape[2]):
        if( abs(np.sum(enc_mat[0,:,i]) - 1.0) > eps):
            print(i)



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




