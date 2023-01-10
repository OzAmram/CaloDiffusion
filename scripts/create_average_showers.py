import numpy as np
import os
import argparse
import h5py as h5
import matplotlib as plt
import h5py

from utils import *

if __name__ == '__main__':

    if(torch.cuda.is_available()): device = torch.device('cuda')
    else: device = torch.device('cpu')
        
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_folder', default='/wclustre/cms_mlsim/denoise/CaloChallenge/', help='Folder containing data and MC files')
    parser.add_argument('--config', default='config_dataset2.json', help='Training parameters')
    parser.add_argument('--output', "-o",  default='avg_showers.h5', help='Location for output file')
    parser.add_argument('--nevts', type=float,default=1e5, help='Number of events to load')
    flags = parser.parse_args()

    dataset_config = LoadJson(flags.config)
    data = []
    energies = []

    print("TRAINING OPTIONS")
    print(dataset_config, flush = True)

    batch_size = dataset_config['BATCH']
    num_epochs = dataset_config['MAXEPOCH']
    early_stop = dataset_config['EARLYSTOP']


    nFiles = len(dataset_config['FILES'])
    for i, dataset in enumerate(dataset_config['FILES']):
        data_,e_ = DataLoader(
            os.path.join(flags.data_folder,dataset),
            dataset_config['SHAPE_PAD'],
            emax = dataset_config['EMAX'],emin = dataset_config['EMIN'],
            max_deposit=dataset_config['MAXDEP'], #noise can generate more deposited energy than generated
            logE=dataset_config['logE'],
            showerMap = dataset_config['SHOWERMAP'],
            nholdout = 5000 if (i == nFiles -1 ) else 0,
            nevts = flags.nevts / nFiles,
        )

        if(i ==0): 
            data = data_
            energies = e_
        else:
            data = np.concatenate((data, data_))
            energies = np.concatenate((energies, e_))


    data_raw ,energies_raw =  ReverseNorm(data, energies,
                                           shape=dataset_config['SHAPE'],
                                           logE=dataset_config['logE'],
                                           max_deposit=dataset_config['MAXDEP'],
                                           emax = dataset_config['EMAX'],
                                           emin = dataset_config['EMIN'],
                                           showerMap = dataset_config['SHOWERMAP'])

    #plt.figure(figsize = (8,6))
    #plt.hist(energies, bins = 100, density = True)
    #plt.xlabel("log(Incident Energy)")
    #plt.savefig("log_energies.png")
    #plt.figure(figsize = (8,6))
    #plt.hist(energies_raw, bins = 100, density = True)
    #plt.xlabel("Incident Energy")
    #plt.savefig("energies.png")

    energies = energies.reshape(-1)

    print(np.amin(energies), np.amax(energies))

    #after log pre-processing energies are uniform from 0 to 1
    startE = 0.
    stopE = 1.
    nbins = 100
    bins = np.linspace(startE, stopE, nbins)
    E_bins = np.digitize(energies, bins)

    avg_shower_shape = list(data_raw.shape)
    avg_shower_shape[0] = nbins
    avg_showers = np.zeros(avg_shower_shape)


    #vectorize ? 
    for i in range(nbins):
        avg_showers[i] = np.mean(data_raw[E_bins == i], axis = 0)

    print("Writing out to %s" % flags.output)
    with h5py.File(flags.output, "w") as fout:
        fout.create_dataset("avg_showers", data = avg_showers)
        fout.create_dataset("E_bins", data = E_bins)

    print("Done!")




