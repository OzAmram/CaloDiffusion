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
from ControlNet import *
from plot import *
import h5py

if(torch.cuda.is_available()): device = torch.device('cuda')
else: device = torch.device('cpu')



parser = argparse.ArgumentParser()

parser.add_argument('--data_folder', default='../data/', help='Folder containing data and MC files')
parser.add_argument('--plot_folder', default='../plots/plots/', help='Folder to save results')
parser.add_argument('--plot_reshape', default=False, action='store_true', help='Plots in emebedded space of HGCal')
parser.add_argument('--generated', '-g', default='', help='Generated showers')
parser.add_argument('--model_loc', default='test', help='Location of model')
parser.add_argument('--config', '-c', default='config_dataset2.json', help='Training parameters')
parser.add_argument('--nevts', type=int,default=-1, help='Number of events to load')
parser.add_argument('--batch_size', type=int, default=100, help='Batch size for generation')
parser.add_argument('--model', default='Diffu', help='Diffusion model to load. Options are: Diffu, AE, all')
parser.add_argument('--plot_label', default='', help='Add to plot')
parser.add_argument('--sample_steps', default = -1, type = int, help='How many steps for sampling (override config)')
parser.add_argument('--sample_offset', default = 0, type = int, help='Skip some iterations in the sampling (noisiest iters most unstable)')
parser.add_argument('--sample_algo', default = 'ddpm', help = 'What sampling algorithm (ddpm, ddim, cold, cold2)')
parser.add_argument('--controlnet', default = False, action='store_true', help = 'Does the model have a controlnet')

parser.add_argument('--layer_only', default=False, action= 'store_true', help='Only sample layer energies')
parser.add_argument('--layer_model', default='', help='Location of model for layer energies')
parser.add_argument('--layer_sample_algo', default = 'ddim', help = 'What sampling algorithm for layer model(ddpm, ddim, cold, cold2)')
parser.add_argument('--layer_sample_steps', default = 200, type = int, help='How many steps for sampling layer model (override config)')


parser.add_argument('--geant_only', action='store_true', default=False,help='Plots with just geant')

parser.add_argument('--job_idx', default = -1, type = int, help = 'Split generation among different jobs')
parser.add_argument('--debug', action='store_true', default=False,help='Debugging options')
parser.add_argument('--from_end', action='store_true', default = False, help='Use events from end of file (usually holdout set)')
parser.add_argument('--trainset', action='store_true', default = False, help='Generate from training set energies')

flags = parser.parse_args()

nevts = int(flags.nevts)
dataset_config = utils.LoadJson(flags.config)
emax = dataset_config['EMAX']
emin = dataset_config['EMIN']
cold_diffu = dataset_config.get('COLD_DIFFU', False)
cold_noise_scale = dataset_config.get("COLD_NOISE", 1.0)
training_obj = dataset_config.get('TRAINING_OBJ', 'noise_pred')
geom_file = dataset_config.get('BIN_FILE', '')
dataset_num = dataset_config.get('DATASET_NUM', 2)
layer_norm = 'layer' in dataset_config['SHOWERMAP']
hgcal = dataset_config.get('HGCAL', False)
max_cells = dataset_config.get('MAX_CELLS', None)

model_nsteps = dataset_config["NSTEPS"] if flags.sample_steps < 0 else flags.sample_steps
if('consis' in flags.sample_algo): model_nsteps = dataset_config['CONSIS_NSTEPS']

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




checkpoint_folder = '../models/{}_{}/'.format(dataset_config['CHECKPOINT_NAME'],flags.model)
energies = None
data = None

dset = dataset_config['EVAL'] if not flags.trainset else dataset_config['FILES']


for i, dataset in enumerate(dset):
    n_dataset = h5py.File(os.path.join(flags.data_folder,dataset))['showers'].shape[0]
    if(evt_start >= n_dataset):
        evt_start -= n_dataset
        continue

    data_,e_,layers_ = DataLoader(
        os.path.join(flags.data_folder,dataset),
        dataset_config['SHAPE_PAD'],
        emax = dataset_config['EMAX'],emin = dataset_config['EMIN'],
        hgcal = hgcal,
        nevts = flags.nevts,
        max_deposit=dataset_config['MAXDEP'], #noise can generate more deposited energy than generated
        logE=dataset_config['logE'],
        showerMap = dataset_config['SHOWERMAP'],
        max_cells = max_cells,

        dataset_num  = dataset_num,
        orig_shape = orig_shape,
    )
    
    if(data is None): 
        data = data_
        energies = e_
        layers = layers_
    else:
        energies = np.concatenate((energies, e_))
        if(layer_norm): layers = np.concatenate((layers, layers_))
    if(flags.nevts > 0 and energies.shape[0] >= flags.nevts): break

if(layer_norm): layers = np.reshape(layers, (layers.shape[0], -1))
else: data = np.reshape(data, (len(data), -1))

torch_E_tensor = torch.from_numpy(energies.astype(np.float32))
torch_layer_tensor =  torch.from_numpy(layers.astype(np.float32)) if layer_norm else torch.zeros_like(torch_E_tensor)

torch_dataset  = torchdata.TensorDataset(torch_E_tensor, torch_layer_tensor)
data_loader = torchdata.DataLoader(torch_dataset, batch_size = batch_size, shuffle = False)

avg_showers = std_showers = E_bins = None
if(cold_diffu or flags.model == 'Avg' or flags.controlnet):
    f_avg_shower = h5.File(flags.data_folder + dataset_config["AVG_SHOWER_LOC"])
    #Already pre-processed
    avg_showers = torch.from_numpy(f_avg_shower["avg_showers"][()].astype(np.float32)).to(device = device)
    std_showers = torch.from_numpy(f_avg_shower["std_showers"][()].astype(np.float32)).to(device = device)
    E_bins = torch.from_numpy(f_avg_shower["E_bins"][()].astype(np.float32)).to(device = device)
    print("AVG showers", avg_showers.shape)

NN_embed = None
if('NN' in shower_embed and not hgcal):
    if(dataset_num == 1):
        bins = XMLHandler("photon", geom_file)
    else: 
        bins = XMLHandler("pion", geom_file)

    NN_embed = NNConverter(bins = bins).to(device = device)
elif(hgcal):
    trainable = dataset_config.get('TRAINABLE_EMBED', False)
    NN_embed = HGCalConverter(bins = dataset_config['SHAPE_FINAL'], geom_file = geom_file, device = device, trainable = trainable).to(device = device)
    if(not trainable):
        NN_embed.init()
    


if(flags.model == "AE"):
    print("Loading AE from " + flags.model_loc)
    """
    model = CaloAE(dataset_config['SHAPE_PAD'][1:], batch_size, config=dataset_config).to(device=device)

    saved_model = torch.load(flags.model_loc, map_location = device)
    if('model_state_dict' in saved_model.keys()): model.load_state_dict(saved_model['model_state_dict'])
    elif(len(saved_model.keys()) > 1): model.load_state_dict(saved_model)
    #model.load_state_dict(torch.load(flags.model_loc, map_location=device))

    generated = []
    for i,(E,d_batch) in enumerate(data_loader):
        E = E.to(device=device)
        d_batch = d_batch.to(device=device)
    
        gen = model(d_batch).detach().cpu().numpy()
        if(i == 0): generated = gen
        else: generated = np.concatenate((generated, gen))
        del E, d_batch
        """

elif(flags.model == "Diffu"):
    print("Loading Diffu model from " + flags.model_loc)

    shape = dataset_config['SHAPE_PAD'][1:] if (not orig_shape) else dataset_config['SHAPE_ORIG'][1:]


    model = CaloDiffu(shape, config=dataset_config , training_obj = training_obj,NN_embed = NN_embed, nsteps = model_nsteps,
            cold_diffu = cold_diffu, avg_showers = avg_showers, std_showers = std_showers, E_bins = E_bins).to(device = device)

    saved_model = torch.load(flags.model_loc, map_location = device)

    if(flags.controlnet):
        #init controlnet model 
        controlnet_model = copy.deepcopy(model)
        #Upsampling layers not used
        controlnet_model.model.ups = None


        model = ControlledUNet(model, controlnet_model)
        #Freeze UNet part

    print(list(saved_model.keys()))

    if('model_state_dict' in saved_model.keys()): model.load_state_dict(saved_model['model_state_dict'])
    else: model.load_state_dict(saved_model)


    layer_model = None
    if(flags.layer_model != "" and "dummy" not in flags.layer_model):
        print("Loading Diffu model from " + flags.layer_model)

        cond_size =1
        if(hgcal): cond_size += 2

        layer_model = ResNet(dim_in = dataset_config['SHAPE_PAD'][2] + 1, num_layers = 5, cond_size = cond_size).to(device = device)
        saved_layer = torch.load(flags.layer_model, map_location = device)
        if('model_state_dict' in saved_layer.keys()): layer_model.load_state_dict(saved_layer['model_state_dict'])
        else: layer_model.load_state_dict(saved_layer)

    gen_layers = []
    generated = []
    gen_layers_ = None
    start_time = time.time()

    print("Sample Algo : %s, %i steps" % (flags.sample_algo, flags.sample_steps))
    if(layer_model is not None):
        print("Layer Sample Algo : %s, %i steps" % (flags.layer_sample_algo, flags.layer_sample_steps))

    for i,(E,layers_) in enumerate(data_loader):
        batch_start = time.time()
        if(E.shape[0] == 0): continue
        E = E.to(device=device)

        layer_shape = (E.shape[0], dataset_config['SHAPE_PAD'][2]+1)
        if(layer_model is not None):
            gen_layers_ = model.Sample(E, num_steps = flags.layer_sample_steps, sample_algo = flags.layer_sample_algo,
                                       model = layer_model, gen_shape = layer_shape, layer_sample = True)
            cond_layers = torch.Tensor(gen_layers_).to(device = device)
        else:
            cond_layers = layers_.to(device = device)

        if(flags.layer_only):#one shot generation
            shower_steps, shower_algo = 1, 'consis'
        else:
            shower_steps, shower_algo = flags.sample_steps, flags.sample_algo

        out = model.Sample(E, layers = cond_layers, num_steps = shower_steps, cold_noise_scale = cold_noise_scale, sample_algo = shower_algo,
                debug = flags.debug, sample_offset = flags.sample_offset)


        if(flags.debug):
            gen, all_gen, x0s = out
            for j in [0,len(all_gen)//4, len(all_gen)//2, 3*len(all_gen)//4, 9*len(all_gen)//10, len(all_gen)-10, len(all_gen)-5,len(all_gen)-1]:
                fout_ex = '{}/{}_{}_norm_voxels_gen_step{}.{}'.format(flags.plot_folder,dataset_config['CHECKPOINT_NAME'],flags.model, j, plt_exts[0])
                make_histogram([all_gen[j].cpu().reshape(-1), data.reshape(-1)], ['Diffu', 'Geant4'], ['blue', 'black'], xaxis_label = 'Normalized Voxel Energy', 
                                num_bins = 40, normalize = True, fname = fout_ex)

                fout_ex = '{}/{}_{}_norm_voxels_x0_step{}.{}'.format(flags.plot_folder,dataset_config['CHECKPOINT_NAME'],flags.model, j, plt_exts[0])
                make_histogram([x0s[j].cpu().reshape(-1), data.reshape(-1)], ['Diffu', 'Geant4'], ['blue', 'black'], xaxis_label = 'Normalized Voxel Energy', 
                                num_bins = 40, normalize = True, fname = fout_ex)
        else: gen = out

    
        if(i == 0): 
            generated = gen
            gen_layers = gen_layers_
        else: 
            generated = np.concatenate((generated, gen))
            if(gen_layers_ is not None): gen_layers = np.concatenate((gen_layers, gen_layers_))

        batch_end = time.time()
        print("Time to sample %i events is %.3f seconds" % (E.shape[0], batch_end - batch_start))
        del E, layers_
    end_time = time.time()
    print("Total sampling time %.3f seconds" % (end_time - start_time))
elif(flags.model == "Avg"):
    #define model just for useful fns
    model = CaloDiffu(dataset_config['SHAPE_PAD'][1:], nevts,config=dataset_config, avg_showers = avg_showers, std_showers = std_showers, E_bins = E_bins ).to(device = device)

    generated = model.gen_cold_image(torch_E_tensor, cold_noise_scale).numpy()


if(not orig_shape): generated = generated.reshape(dataset_config["SHAPE_ORIG"])

if(flags.debug):
    fout_ex = '{}/{}_{}_norm_voxels.{}'.format(flags.plot_folder,dataset_config['CHECKPOINT_NAME'],flags.model, plt_exts[0])
    make_histogram([generated.reshape(-1), data.reshape(-1)], ['Diffu', 'Geant4'], ['blue', 'black'], xaxis_label = 'Normalized Voxel Energy', 
                    num_bins = 40, normalize = True, fname = fout_ex)

out_layers = layers if (gen_layers is None) else gen_layers
generated,energies = utils.ReverseNorm(generated,energies, layerE = out_layers,
                                       shape=dataset_config['SHAPE_FINAL'],
                                       logE=dataset_config['logE'],
                                       max_deposit=dataset_config['MAXDEP'],
                                       emax = dataset_config['EMAX'],
                                       emin = dataset_config['EMIN'],
                                       showerMap = dataset_config['SHOWERMAP'],
                                       dataset_num  = dataset_num,
                                       orig_shape = orig_shape,
                                       ecut = dataset_config['ECUT'],
                                       hgcal = hgcal,
                                       )

energies = np.reshape(energies,(energies.shape[0],-1))
do_mask = False

if(do_mask  and dataset_num > 1):
    #mask for voxels that are always empty
    mask_file = os.path.join(flags.data_folder,dataset_config['EVAL'][0].replace('.hdf5','_mask.hdf5'))
    if(not os.path.exists(mask_file)):
        print("Creating mask based on data batch")
        mask = np.sum(data,0)==0

    else:
        with h5.File(mask_file,"r") as h5f:
            mask = h5f['mask'][:]
    generated = generated*(np.reshape(mask,(1,-1))==0)

if(flags.generated == ""):
    fout = os.path.join(checkpoint_folder,'generated_{}_{}{}.h5'.format(dataset_config['CHECKPOINT_NAME'],flags.sample_algo + str(flags.sample_steps), job_label))
else:
    fout = flags.generated

flags.generated = fout

print("Creating " + fout)
if(not hgcal):
    with h5.File(fout,"w") as h5f:
        dset = h5f.create_dataset("showers", data=1000*np.reshape(generated, dataset_config["SHAPE_ORIG"]), compression = 'gzip')
        dset = h5f.create_dataset("incident_energies", data=1000*energies, compression = 'gzip')
else:
    with h5.File(fout,"w") as h5f:
        print(generated.shape)
        dset = h5f.create_dataset("showers", data=(1./100.)* np.reshape(generated, dataset_config["SHAPE_ORIG"]), compression = 'gzip')
        dset = h5f.create_dataset("gen_info", data=energies, compression = 'gzip')

#if(flags.job_idx < 0): make_plots(flags)
