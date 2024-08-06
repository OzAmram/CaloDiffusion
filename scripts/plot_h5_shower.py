from HGCal_utils import *
import argparse
import os
import h5py
import numpy as np


parser = argparse.ArgumentParser()

parser.add_argument('-g', '--geom_file', default='/home/oamram/CaloDiffusion/HGCalShowers/geom_william.pkl', help='Geom file')
parser.add_argument('-i', '--fin', default='', help='File with showers to plot')
parser.add_argument('-n', '--nShowers', default=1, type = int, help='How many showers to plot')
parser.add_argument('-o', '--outdir', default='../plots/showers/', help='Where to put output plots')
parser.add_argument('--EMin', type = float, default=-1.0, help='Voxel min energy')
flags = parser.parse_args()

geom_file = open(flags.geom_file, 'rb')
geo = pickle_load(geom_file)

if (not os.path.exists(flags.outdir)): os.system("mkdir " + flags.outdir)
f = h5py.File(flags.fin)
showers = f['showers'][:]
showers *= 200.

if(flags.EMin > 0.):
    mask = showers < flags.EMin
    showers[mask] = 0.

print(geo.xmap.shape)

for i in range(flags.nShowers):
    print("Shower %i" % i)
    shower = showers[i]
    for ilay in range(geo.nlayers):

        #print(geo.xmap[ilay][:5], geo.ymap[ilay][:5])
        ncells = int(round(geo.ncells[ilay]))
        plot_shower_hex(geo.xmap[ilay][:ncells], geo.ymap[ilay][:ncells], shower[ilay][:ncells], nrings = geo.nrings, fout = flags.outdir + "shower%i_lay%i.png" % (i, ilay))

#avg showers
avg_shower = np.mean(showers, axis=0)
std_shower = np.std(showers, axis=0)

print("avg shower")
for ilay in range(geo.nlayers):

    ncells = int(round(geo.ncells[ilay]))
    plot_shower_hex(geo.xmap[ilay][:ncells], geo.ymap[ilay][:ncells], avg_shower[ilay][:ncells], nrings = geo.nrings, fout = flags.outdir + "avg_shower_lay%i.png" % (ilay))
    plot_shower_hex(geo.xmap[ilay][:ncells], geo.ymap[ilay][:ncells], std_shower[ilay][:ncells], nrings = geo.nrings, fout = flags.outdir + "stddev_shower_lay%i.png" % (ilay))





