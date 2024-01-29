import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import argparse
import h5py as h5
import torch.optim as optim
import torch.utils.data as torchdata

from utils import *

if __name__ == '__main__':

    if(torch.cuda.is_available()): device = torch.device('cuda')
    else: device = torch.device('cpu')
        
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_folder', default='../data/', help='Folder containing data and MC files')
    parser.add_argument('-c', '--config', default='configs/test.json', help='Config file with training parameters')
    parser.add_argument('--output', "-o",  default='avg_showers.h5', help='Location for output file')
    parser.add_argument('--nevts', type=float,default=-1, help='Number of events to load')
    parser.add_argument('--plot_dir', default="", help='Put plots of average showers')
    flags = parser.parse_args()

    dataset_config = LoadJson(flags.config)
    data = []
    energies = []

    print("TRAINING OPTIONS")
    print(dataset_config, flush = True)

    nholdout  = dataset_config.get('HOLDOUT', 0)

    batch_size = dataset_config['BATCH']
    num_epochs = dataset_config['MAXEPOCH']
    early_stop = dataset_config['EARLYSTOP']
    dataset_num = dataset_config['DATASET_NUM']

    shower_embed = dataset_config.get('SHOWER_EMBED', '')
    orig_shape = ('orig' in shower_embed)
    layer_norm = 'layer' in dataset_config['SHOWERMAP']


    nFiles = len(dataset_config['FILES'])
    for i, dataset in enumerate(dataset_config['FILES']):
        data_,e_,layers_ = DataLoader(
            os.path.join(flags.data_folder,dataset),
            dataset_config['SHAPE_PAD'],
            emax = dataset_config['EMAX'],emin = dataset_config['EMIN'],
            nevts = flags.nevts,
            max_deposit=dataset_config['MAXDEP'], #noise can generate more deposited energy than generated
            logE=dataset_config['logE'],
            showerMap = dataset_config['SHOWERMAP'],

            nholdout = nholdout if (i == len(dataset_config['FILES']) -1 ) else 0,
            dataset_num  = dataset_num,
            orig_shape = orig_shape,
        )

        if(i ==0): 
            data = data_
            energies = e_
        else:
            data = np.concatenate((data, data_))
            energies = np.concatenate((energies, e_))


    plt.figure(figsize = (8,6))
    plt.hist(energies, bins = 100, density = True)
    plt.xlabel("log(Incident Energy)")
    plt.savefig("log_energies.png")
    plt.figure(figsize = (8,6))
    #plt.hist(energies_raw, bins = 100, density = True)
    #plt.xlabel("Incident Energy")
    #plt.savefig("energies.png")

    energies = energies.reshape(-1)

    #after log pre-processing energies are uniform from 0 to 1
    startE = -0.001
    stopE = 1.
    nbins = 100 if dataset_num >= 2 else 15
    E_bins = np.linspace(startE, stopE, nbins)
    bins = np.digitize(energies, E_bins) - 1 #Numpy indexes bins starting at 1...
    print(np.amin(bins), np.amax(bins))

    avg_shower_shape = list(data.shape)
    avg_shower_shape[0] = nbins
    avg_showers = np.zeros(avg_shower_shape)
    std_showers = np.zeros(avg_shower_shape)

    #vectorize ? 
    for i in range(nbins):
        print("Starting bin %i, %i showers" % (i, np.sum(bins == i)))
        avg_showers[i] = np.mean(data[bins == i], axis = 0)
        std_showers[i] = np.std(data[bins == i], axis = 0)

    print("Writing out to %s" % flags.output)
    with h5.File(flags.output, "w") as fout:
        fout.create_dataset("avg_showers", data = avg_showers)
        fout.create_dataset("std_showers", data = std_showers)
        fout.create_dataset("E_bins", data = E_bins)

    print("Done!")

    if(flags.plot_dir != ""):
        print("Making plots of showers, output to %s" % flags.plot_dir)
        for draw_bin in range(nbins):
            E_center = (E_bins[draw_bin] + E_bins[draw_bin + 1]) /2.
            shower = avg_showers[draw_bin]

            raw_shower, raw_E = ReverseNorm(shower,E_center,
                    dataset_config['SHAPE_PAD'],
                    emax = dataset_config['EMAX'],emin = dataset_config['EMIN'],
                    max_deposit=dataset_config['MAXDEP'],
                    logE=dataset_config['logE'],
                    showerMap = dataset_config['SHOWERMAP'],
            )

            HLF = HighLevelFeatures("electron", "/work1/cms_mlsim/CaloDiffusion/CaloChallenge/code/binning_dataset_2.xml")
            HLF._DrawShower(raw_shower.reshape(-1), flags.plot_dir + "avg_shower_bin%i.png" % draw_bin, "Avg shower Bin %i (Avg E %.3f GeV)" % (draw_bin , raw_E))




