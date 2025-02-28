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

if(torch.cuda.is_available()): device = torch.device('cuda')
else: device = torch.device('cpu')

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

nevts = int(flags.nevts)
dataset_config = utils.LoadJson(flags.config)
emax = dataset_config['EMAX']
emin = dataset_config['EMIN']
cold_diffu = dataset_config.get('COLD_DIFFU', False)
cold_noise_scale = dataset_config.get("COLD_NOISE", 1.0)
training_obj = dataset_config.get('TRAINING_OBJ', 'noise_pred')
dataset_num = dataset_config.get('DATASET_NUM', 2)

sample_steps = dataset_config["NSTEPS"] if flags.sample_steps < 0 else flags.sample_steps

batch_size = flags.batch_size
shower_embed = dataset_config.get('SHOWER_EMBED', '')
orig_shape = ('orig' in shower_embed)
do_NN_embed = ('NN' in shower_embed)

if(not os.path.exists(flags.plot_folder)): 
    print("Creating plot directory " + flags.plot_folder)
    os.system("mkdir " + flags.plot_folder)

evt_start = 0
job_label = ""
if(flags.job_idx >= 0):
    if(flags.nevts <= 0):
        print("Must include number of events for split jobs")
        sys.exit(1)
    evt_start = flags.job_idx * flags.nevts
    job_label = "_job%i" % flags.job_idx




if flags.sample:
    checkpoint_folder = '../models/{}_{}/'.format(dataset_config['CHECKPOINT_NAME'],flags.model)
    energies = None
    data = None
    for i, dataset in enumerate(dataset_config['EVAL']):
        n_dataset = h5py.File(os.path.join(flags.data_folder,dataset))['showers'].shape[0]
        if(evt_start >= n_dataset):
            evt_start -= n_dataset
            continue

        data_,e_ = utils.DataLoader(
            os.path.join(flags.data_folder,dataset),
            dataset_config['SHAPE_PAD'],
            emax = dataset_config['EMAX'],emin = dataset_config['EMIN'],
            nevts = flags.nevts,
            max_deposit=dataset_config['MAXDEP'], #noise can generate more deposited energy than generated
            logE=dataset_config['logE'],
            showerMap = dataset_config['SHOWERMAP'],
            from_end = flags.from_end,
            dataset_num  = dataset_num,
            orig_shape = orig_shape,
            evt_start = evt_start
        )
        
        if(data is None): 
            data = data_
            energies = e_
        else:
            data = np.concatenate((data, data_))
            energies = np.concatenate((energies, e_))
        if(flags.nevts > 0 and data_.shape[0] == flags.nevts): break

    energies = np.reshape(energies,(-1,))
    if(not orig_shape): data = np.reshape(data,dataset_config['SHAPE_PAD'])
    else: data = np.reshape(data, (len(data), -1))

    torch_data_tensor = torch.from_numpy(data)
    torch_E_tensor = torch.from_numpy(energies)

    print("DATA mean, std", torch.mean(torch_data_tensor), torch.std(torch_data_tensor))

    torch_dataset  = torchdata.TensorDataset(torch_E_tensor, torch_data_tensor)
    data_loader = torchdata.DataLoader(torch_dataset, batch_size = batch_size, shuffle = False)

    avg_showers = std_showers = E_bins = None

    NN_embed = None
    if('NN' in shower_embed):
        if(dataset_num == 1):
            binning_file = "../CaloChallenge/code/binning_dataset_1_photons.xml"
            bins = XMLHandler("photon", binning_file)
        else: 
            binning_file = "../CaloChallenge/code/binning_dataset_1_pions.xml"
            bins = XMLHandler("pion", binning_file)

        NN_embed = NNConverter(bins = bins).to(device = device)
        


    if(flags.model == "Diffu"):
        print("Loading Diffu model from " + flags.model_loc)

        shape = dataset_config['SHAPE_PAD'][1:] if (not orig_shape) else dataset_config['SHAPE_ORIG'][1:]
        model = CaloDiffu(shape, config=dataset_config , training_obj = training_obj,NN_embed = NN_embed, nsteps = sample_steps,
                cold_diffu = cold_diffu, avg_showers = avg_showers, std_showers = std_showers, E_bins = E_bins ).to(device = device)

        saved_model = torch.load(flags.model_loc, map_location = device)
        if('model_state_dict' in saved_model.keys()): model.load_state_dict(saved_model['model_state_dict'])
        elif(len(saved_model.keys()) > 1): model.load_state_dict(saved_model)

        generated = []
        start_time = time.time()
        flags.debug = True
        for i,(E,d_batch) in enumerate(data_loader):
            if(E.shape[0] == 0): continue
            E = E.to(device=device)
            d_batch = d_batch.to(device=device)

            out = model.Sample(E, num_steps = sample_steps, cold_noise_scale = cold_noise_scale, sample_algo = flags.sample_algo,
                    debug = flags.debug, sample_offset = flags.sample_offset)
            gen, all_gen, x0s = out

            all_gen = np.array(all_gen)

        
            if(i == 0): 
                generated = gen
                all_steps = all_gen
            else: 
                generated = np.concatenate((generated, gen))
                all_steps = np.concatenate((all_steps, all_gen), axis = 1)
            del E, d_batch

        end_time = time.time()
        print("Total sampling time %.3f seconds" % (end_time - start_time))

    print("GENERATED", np.mean(generated), np.std(generated), np.amax(generated), np.amin(generated))


    if(not orig_shape): generated = generated.reshape(dataset_config["SHAPE"])

    for iStep in range(len(all_steps)):

        showers,energies_norm = utils.ReverseNorm(all_steps[iStep], energies[:nevts],
                                               shape=dataset_config['SHAPE'],
                                               logE=dataset_config['logE'],
                                               max_deposit=dataset_config['MAXDEP'],
                                               emax = dataset_config['EMAX'],
                                               emin = dataset_config['EMIN'],
                                               showerMap = dataset_config['SHOWERMAP'],
                                               dataset_num  = dataset_num,
                                               orig_shape = orig_shape,
                                               ecut = dataset_config['ECUT'],
                                               )

        energies_norm = np.reshape(energies_norm,(-1,1))
        if(dataset_num > 1):
            #mask for voxels that are always empty
            mask_file = os.path.join(flags.data_folder,dataset_config['EVAL'][0].replace('.hdf5','_mask.hdf5'))
            if(not os.path.exists(mask_file)):
                print("Creating mask based on data batch")
                mask = np.sum(data,0)==0

            else:
                with h5.File(mask_file,"r") as h5f:
                    mask = h5f['mask'][:]
            showers = showers*(np.reshape(mask,(1,-1))==0)
        fout = os.path.join(checkpoint_folder,'generated_step%i.h5' % iStep)
        #print("Creating " + fout)
        with h5.File(fout,"w") as h5f:
            dset = h5f.create_dataset("showers", data=1000*np.reshape(showers,(showers.shape[0],-1)), compression = 'gzip')
            dset = h5f.create_dataset("incident_energies", data=1000*energies, compression = 'gzip')
