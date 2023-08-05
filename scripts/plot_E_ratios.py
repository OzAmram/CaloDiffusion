import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import gridspec
import argparse
import h5py as h5
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import time, sys, copy
import utils
import torch
import torch.utils.data as torchdata
from CaloAE import *
from CaloDiffu import *
import h5py


plt_exts = ["png", "pdf"]
#plt_ext = "pdf"
rank = 0
size = 1

utils.SetStyle()


parser = argparse.ArgumentParser()

parser.add_argument('--data_folder', default='/wclustre/cms_mlsim/denoise/CaloChallenge/', help='Folder containing data and MC files')
parser.add_argument('--plot_folder', default='../plots', help='Folder to save results')
parser.add_argument('--generated', '-g', default='', help='Generated showers')
parser.add_argument('--model_loc', default='test', help='Location of model')
parser.add_argument('--config', default='config_dataset2.json', help='Training parameters')
parser.add_argument('--nevts', type=int,default=-1, help='Number of events to load')
parser.add_argument('--batch_size', type=int, default=100, help='Batch size for generation')
parser.add_argument('--model', default='Diffu', help='Diffusion model to load. Options are: Diffu, AE, all')
parser.add_argument('--plot_label', default='', help='Add to plot')
parser.add_argument('--sample', action='store_true', default=False,help='Sample from learned model')
parser.add_argument('--sample_steps', default = -1, type = int, help='How many steps for sampling (override config)')
parser.add_argument('--sample_offset', default = 0, type = int, help='Skip some iterations in the sampling (noisiest iters most unstable)')
parser.add_argument('--sample_algo', default = 'ddpm', help = 'What sampling algorithm (ddpm, ddim, cold, cold2)')
parser.add_argument('--job_idx', default = -1, type = int, help = 'Split generation among different jobs')
parser.add_argument('--debug', action='store_true', default=False,help='Debugging options')
parser.add_argument('--from_end', action='store_true', default = False, help='Use events from end of file (usually holdout set)')

flags = parser.parse_args()
dataset_config = utils.LoadJson(flags.config)
emax = dataset_config['EMAX']
emin = dataset_config['EMIN']
training_obj = dataset_config.get('TRAINING_OBJ', 'noise_pred')
dataset_num = dataset_config.get('DATASET_NUM', 2)

sample_steps = dataset_config["NSTEPS"] if flags.sample_steps < 0 else flags.sample_steps
evt_start = 0


def LoadSamples(fname):
    with h5.File(fname,"r") as h5f:
        generated = h5f['showers'][:flags.nevts]/1000.
        energies = h5f['incident_energies'][:flags.nevts]/1000.
    energies = np.reshape(energies,(-1,1))
    if(dataset_num <= 1):
        generated = geom_conv.convert(geom_conv.reshape(generated)).detach().numpy()
    generated = np.reshape(generated,dataset_config['SHAPE'])

    return generated,energies

def HistERatio(data_dict,true_energies):
    

    ratios =  []
    feed_dict = {}
    for key in data_dict:
        dep = np.sum(data_dict[key].reshape(data_dict[key].shape[0],-1),-1)
        if('Geant' in key): feed_dict[key] = dep / true_energies.reshape(-1)
        else: feed_dict[key] = dep / model_energies.reshape(-1)

    binning = np.linspace(0.5, 1.5, 51)

    fig,ax0 = utils.HistRoutine(feed_dict,xlabel='Dep. energy / Gen. energy', logy=False,binning=binning, ratio = True, plot_label = flags.plot_label, leg_font = 16)
    for plt_ext in plt_exts: fig.savefig('{}/ERatio_multi.{}'.format(flags.plot_folder, plt_ext))
    return feed_dict


geom_conv = None
if(dataset_num <= 1):
    bins = XMLHandler(dataset_config['PART_TYPE'], dataset_config['BIN_FILE'])
    geom_conv = GeomConverter(bins)




fnames = [
          "../models/dataset2_june15_attn_compressZ_Diffu/generated_dataset2_june15_attn_compressZ_Diffu_n50.h5",
          "../models/dataset2_june15_attn_compressZ_Diffu/generated_dataset2_june15_attn_compressZ_Diffu_n100.h5",
          "../models/dataset2_june15_attn_compressZ_Diffu/generated_dataset2_june15_attn_compressZ_Diffu_n200.h5",
        "../models/dataset2_june15_attn_compressZ_Diffu/generated_dataset2_june15_attn_compressZ_Diffu_n400.h5",
          ]
labels = ["CaloDiffusion 50 Steps",
          "CaloDiffusion 100 Steps",
          "CaloDiffusion 200 Steps",
          "CaloDiffusion 400 Steps",]


energies = []
data_dict = {}
for i,fname in enumerate(fnames):
    if np.size(energies) == 0:
        data,energies = LoadSamples(fname)
        data_dict[labels[i]]=data
    else:
        data_dict[labels[i]]=LoadSamples(fname)[0]

total_evts = energies.shape[0]


data = []
true_energies = []
for dataset in dataset_config['EVAL']:
    with h5.File(os.path.join(flags.data_folder,dataset),"r") as h5f:
        if(flags.from_end):
            start = -int(total_evts)
            end = None
        else: 
            start = evt_start
            end = start + total_evts
        show = h5f['showers'][start:end]/1000.
        if(dataset_num <=1 ):
            show = geom_conv.convert(geom_conv.reshape(show)).detach().numpy()
        data.append(show)
        true_energies.append(h5f['incident_energies'][start:end]/1000.)
        if(data[-1].shape[0] == total_evts): break


data_dict['Geant4']=np.reshape(data,dataset_config['SHAPE'])
true_energies = np.reshape(true_energies,(-1,1))
model_energies = np.reshape(energies,(-1,1))
#assert(np.allclose(data_energies, model_energies))


plot_routines = {
     'Energy Ratio split':HistERatio,
}


print("Saving plots to "  + os.path.abspath(flags.plot_folder) )
for plot in plot_routines:
    if '2D' in plot and flags.model == 'all':continue #skip scatter plots superimposed
    print(plot)
    if 'split' in plot:
        plot_routines[plot](data_dict,energies)
    else:
        high_level.append(plot_routines[plot](data_dict))
