import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import gridspec
import argparse
import h5py as h5
import os
import torch.utils.data as torchdata
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import time, sys, copy
import utils
from HGCal_utils import *
import h5py
from SparsityClassifier import SparsityClassifier
plt_exts = [".png", ".pdf", "_logy.png", "_logy.pdf"]
#plt_ext = "pdf"

def save_fig(fname, fig, ax0):
    for plt_ext in plt_exts:
        if('logy') in plt_ext: ax0.set_yscale("log")
        else: ax0.set_yscale("linear")
        fig.savefig(fname + plt_ext)

def apply_mask_conserveE(generated, mask):
        #Preserve layer energies after applying a mask
        generated[generated < 0] = 0 
        d_masked = np.where(mask, generated, 0.)
        lostE = np.sum(d_masked, axis = -1, keepdims=True)
        ELayer = np.sum(generated, axis = -1, keepdims=True)
        eps = 1e-10
        rescale = (ELayer + eps)/(ELayer - lostE +eps)
        generated[mask] = 0.
        generated *= rescale

        return generated

def apply_sparsity_classifier(showers, gen_info, fname, dataset_config, plot_dir = ""):

    print("Applying sparsity classifier from %s" % fname)

    geom_file = dataset_config.get('BIN_FILE', '')
    trainable = dataset_config.get('TRAINABLE_EMBED', False)
    shape = dataset_config['SHAPE_FINAL'][1:] 
    max_deposit=dataset_config['MAXDEP'] #noise can generate more deposited energy than generated
    showerMap = dataset_config['SHOWERMAP'],
    dataset_num = dataset_config.get('DATASET_NUM', 2)
    shower_embed = dataset_config.get('SHOWER_EMBED', '')
    orig_shape = ('orig' in shower_embed)
    ecut = dataset_config['ECUT']

    gen_max = np.array(dataset_config['EMAX'])
    gen_min = np.array(dataset_config['EMIN'])

    if(torch.cuda.is_available()): device = torch.device('cuda')
    else: device = torch.device('cpu')

    NN_embed = HGCalConverter(bins = dataset_config['SHAPE_FINAL'], geom_file = geom_file, device = device, trainable = trainable).to(device = device)
    NN_embed.init()

    model = SparsityClassifier(shape, config=dataset_config, NN_embed = NN_embed ).to(device = device)
    checkpoint = torch.load(fname, map_location = device)

    if('model_state_dict' in checkpoint.keys()): model.load_state_dict(checkpoint['model_state_dict'], strict = False)
    elif(len(checkpoint.keys()) > 1): model.load_state_dict(checkpoint, strict = False)

    shower_preprocessed, layerE_preprocessed = preprocess_hgcal_shower(showers, gen_info[:,0], shape, showerMap, dataset_num = dataset_num, orig_shape = orig_shape, ecut = ecut, max_deposit=max_deposit)
    gen_preprocessed = (gen_info-gen_min)/(gen_max-gen_min)

    showers_torch = torch.from_numpy(shower_preprocessed.astype(np.float32))
    gen_torch = torch.from_numpy(gen_preprocessed.astype(np.float32))

    #Evaluate voxel probabailites 
    torch_dataset  = torchdata.TensorDataset(showers_torch, gen_torch)
    batch_size = 256
    data_loader = torchdata.DataLoader(torch_dataset, batch_size = batch_size, shuffle = False)

    voxel_probs = None
    for i,(shower_batch, gen_batch) in enumerate(data_loader):
        shower_batch = shower_batch.to(device)
        gen_batch = gen_batch.to(device)

        voxel_p = model.pred(shower_batch, gen_batch, None).detach().cpu().numpy()
        if(i ==0): voxel_probs = voxel_p
        else: voxel_probs = np.concatenate((voxel_probs, voxel_p))

    plt.figure(figsize=(10,10))
    plt.hist(voxel_probs.reshape(-1), bins = 100, density = True)
    plt.yscale('log')

    if(len(plot_dir) > 0): plt.savefig(plot_dir + "voxel_probs.png")

    rand_sample = np.random.rand(*voxel_probs.shape)

    mask = voxel_probs < 0.1
    #mask = (rand_sample < voxel_probs) & (voxel_probs < 0.4)
    #mask = voxel_probs < 0.5 & (rand_sample > voxel_probs)
    #mask = (voxel_probs < 0.1) | ((voxel_probs < 0.45) & (rand_sample > voxel_probs))
    #mask = (voxel_probs < 0.45) & (rand_sample > voxel_probs)
    print('mask mean', np.mean(mask))

    #showers[mask] = 0.
    print("before", np.mean(showers))
    showers_new = apply_mask_conserveE(showers, mask)
    print("after", np.mean(showers_new))


    return showers_new



def make_plots(flags):

    utils.SetStyle()
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


    batch_size = flags.batch_size
    shower_embed = dataset_config.get('SHOWER_EMBED', '')
    orig_shape = ('orig' in shower_embed)
    do_NN_embed = ('NN' in shower_embed)


    if((not hgcal) or flags.plot_reshape): shape_plot = dataset_config['SHAPE_FINAL']
    else: shape_plot = dataset_config['SHAPE_PAD']
    print("Data shape", shape_plot)

    if(not os.path.exists(flags.plot_folder)): os.system("mkdir %s" % flags.plot_folder)

    if(hgcal):
        NN_embed = HGCalConverter(bins = shape_plot, geom_file = dataset_config['BIN_FILE'])
        if(flags.plot_reshape): NN_embed.init()
        else: print("HGCal in original shape")

    geom_conv = None
    def LoadSamples(fname, EMin = -1.0, nevts = -1, diffu_sample = False):
        print("Load %s" % fname)
        end = None if nevts < 0 else nevts
        #scale_fac = 200. if hgcal else (1/1000.)
        scale_fac = 100. if hgcal else (1/1000.)
        with h5.File(fname,"r") as h5f:
            if(hgcal): 
                generated = h5f['showers'][:end,:,:dataset_config['MAX_CELLS']] * scale_fac
                energies = h5f['gen_info'][:end,0] 
                gen_info = np.array(h5f['gen_info'][:end,:])
            else: 
                generated = h5f['showers'][:end] * scale_fac
                energies = h5f['incident_energies'][:end] * scale_fac
        energies = np.reshape(energies,(-1,1))
        if(flags.plot_reshape):
            if(dataset_num <= 1):
                bins = XMLHandler(dataset_config['PART_TYPE'], dataset_config['BIN_FILE'])
                geom_conv = GeomConverter(bins)
                generated = geom_conv.convert(geom_conv.reshape(generated)).detach().numpy()
            elif(hgcal):
                generated =torch.from_numpy(generated.astype(np.float32)).reshape(dataset_config['SHAPE_PAD'])
                generated = NN_embed.enc(generated).detach().numpy()

        generated = np.reshape(generated,shape_plot)

        if(diffu_sample and len(flags.sparsity_classifier) > 0):
            generated = apply_sparsity_classifier(generated, gen_info, flags.sparsity_classifier, dataset_config, flags.plot_folder)

        if(EMin > 0.):
            mask = generated < EMin
            generated = apply_mask_conserveE(generated, mask)


        return generated,energies


    geant_key= 'Geant4 (CMSSW)'
    model = flags.model

    energies = []
    data_dict = {}

    checkpoint_folder = '../models/{}_{}/'.format(dataset_config['CHECKPOINT_NAME'], model)
    

    data = []
    true_energies = []

    if(not flags.geant_only):
        if(flags.generated == ""):
            print("Missing data file to plot!")
            exit(1)
        f_sample = flags.generated

        showers,energies = LoadSamples( f_sample, flags.EMin, flags.nevts, diffu_sample = True )
        data_dict[utils.name_translate[model]]=showers
        total_evts = energies.shape[0]


    for dataset in dataset_config['EVAL']:
        showers, energies = LoadSamples("../data/" + dataset, flags.EMin, total_evts)
        data.append(showers)
        true_energies.append(energies)
        if(data[-1].shape[0] == total_evts): break

    data_dict[geant_key]=np.reshape(data,shape_plot)

    if(not flags.geant_only):
        print("Geant Avg", np.mean(data_dict[geant_key]))
        print("Generated Avg", np.mean(data_dict[utils.name_translate[model]]))
        print("Geant Range: ", np.min(data_dict[geant_key] [data_dict[geant_key] > 0.]), np.max(data_dict[geant_key]))
        print("Generated Range: ", np.min(data_dict[utils.name_translate[model]]), np.max(data_dict[utils.name_translate[model]]))
    else: 
        energies = copy.copy(true_energies)
        total_evts = data_dict[geant_key].shape[0]



    true_energies = np.reshape(true_energies,(-1,1))
    model_energies = np.reshape(energies,(-1,1))
    #assert(np.allclose(data_energies, model_energies))

    #Plot high level distributions and compare with real values
    #assert np.allclose(true_energies,energies), 'ERROR: Energies between samples dont match'


    def IncidentE(data_dict,es):
        
        ratios =  []
        feed_dict = {}
        for key in data_dict:
            if('Geant' in key): feed_dict[key] = true_energies.reshape(-1)
            else: feed_dict[key] = model_energies.reshape(-1)

        binning = np.linspace(0.5, 1.5, 51)

        fig,ax0 = utils.HistRoutine(feed_dict,xlabel='Gen. energy', logy=False,binning=binning, ratio = True, plot_label = flags.plot_label, cms_style = flags.cms)
        save_fig('{}/EInc_{}_{}'.format(flags.plot_folder,dataset_config['CHECKPOINT_NAME'],flags.model), fig, ax0)
        return feed_dict


    def HistERatio(data_dict,es):
        

        ratios =  []
        feed_dict = {}
        for key in data_dict:
            dep = np.sum(data_dict[key].reshape(data_dict[key].shape[0],-1),-1)
            if('Geant' in key): feed_dict[key] = dep / true_energies.reshape(-1)
            else: feed_dict[key] = dep / model_energies.reshape(-1)

        #Energy scale is arbitrary, scale so dist centered at 1 for geant

        norm = np.mean(feed_dict[geant_key])
        for key in data_dict:
            if('Geant' in key): feed_dict[key] /= norm
            else: feed_dict[key] /= norm

        #binning = np.linspace(0.5, 1.5, 51)
        binning = np.linspace(0.7, 1.3, 30)

        fig,ax0 = utils.HistRoutine(feed_dict,xlabel='Dep. energy / Gen. energy', logy=False,binning=binning, ratio = True, plot_label = flags.plot_label, cms_style = flags.cms)
        save_fig('{}/ERatio_{}_{}'.format(flags.plot_folder,dataset_config['CHECKPOINT_NAME'],flags.model), fig, ax0)
        return feed_dict



    def ScatterESplit(data_dict,es):
        

        fig,ax = utils.SetFig("Gen. energy [GeV]","Dep. energy [GeV]")

        if(flags.cms):
            hep.style.use(hep.style.CMS)
            hep.cms.text(ax=ax, text="Simulation Preliminary")

        for key in data_dict:
            x = true_energies[0:500] if 'Geant' in key else model_energies[0:500]
            y = np.sum(data_dict[key].reshape(data_dict[key].shape[0],-1),-1)[0:500]

            ax.scatter(x, y, label=key)


        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.legend(loc='best',fontsize=16,ncol=1)
        plt.tight_layout()
        if(len(flags.plot_label) > 0): ax.set_title(flags.plot_label, fontsize = 20, loc = 'right', style = 'italic')
        save_fig('{}/Scatter_{}_{}'.format(flags.plot_folder,dataset_config['CHECKPOINT_NAME'],flags.model), fig, ax)



    def AverageShowerWidth(data_dict):

        def GetMatrix(sizex, minval=-1,maxval=1, binning = None):
            nbins = sizex
            if(binning is None): binning = np.linspace(minval,maxval,nbins+1)
            coord = [(binning[i] + binning[i+1])/2.0 for i in range(len(binning)-1)]
            matrix = np.array(coord)
            return matrix

        
        #TODO : Use radial bins
        #r_bins = [0,4.65,9.3,13.95,18.6,23.25,27.9,32.55,37.2,41.85]

        phi_matrix = GetMatrix(shape_plot[3], minval = -math.pi, maxval = math.pi)
        phi_matrix = np.reshape(phi_matrix,(1,1, phi_matrix.shape[0]))
        
        
        r_matrix = GetMatrix(shape_plot[4], minval = 0, maxval = shape_plot[4])
        r_matrix = np.reshape(r_matrix,(1,1, r_matrix.shape[0]))


        def GetCenter(matrix,energies,power=1):
            ec = np.sum(energies*np.power(matrix,power),-1)
            sum_energies = np.sum(np.reshape(energies,(energies.shape[0],energies.shape[1],-1)),-1)
            ec = np.ma.divide(ec,sum_energies).filled(0)
            return ec



        def GetWidth(mean,mean2):
            width = np.ma.sqrt(mean2-mean**2).filled(0)
            return width
        
        feed_dict_phi = {}
        feed_dict_phi2 = {}
        feed_dict_r = {}
        feed_dict_r2 = {}
        
        for key in data_dict:

            data = data_dict[key]
            phi_preprocessed = np.reshape(data,(data.shape[0],shape_plot[2], shape_plot[3],-1))
            phi_proj = preprocessed = np.sum(phi_preprocessed, axis = -1)

            r_preprocessed = np.reshape(data,(data.shape[0],shape_plot[2], shape_plot[4],-1))
            r_proj = preprocessed = np.sum(r_preprocessed, axis = -1)

            feed_dict_phi[key], feed_dict_phi2[key] = utils.ang_center_spread(phi_matrix, phi_proj)
            feed_dict_r[key] = GetCenter(r_matrix, r_proj)
            feed_dict_r2[key] = GetWidth(feed_dict_r[key],GetCenter(r_matrix,r_proj,2))


        if(dataset_config['cartesian_plot']): 
            xlabel1 = 'x'
            f_str1 = "Eta"
            xlabel2 = 'y'
            f_str2 = "Phi"
        else: 
            xlabel1 = 'r'
            f_str1 = "R"
            xlabel2 = 'alpha'
            f_str2 = "Alpha"
        fig,ax0 = utils.PlotRoutine(feed_dict_r,xlabel='Layer number', ylabel= '%s-center of energy' % xlabel1, plot_label = flags.plot_label, cms_style = flags.cms)
        save_fig('{}/{}EC_{}_{}'.format(flags.plot_folder,f_str1,dataset_config['CHECKPOINT_NAME'],flags.model), fig, ax0)
        fig,ax0 = utils.PlotRoutine(feed_dict_phi,xlabel='Layer number', ylabel= '%s-center of energy' % xlabel2, plot_label = flags.plot_label, cms_style = flags.cms)
        save_fig('{}/{}EC_{}_{}'.format(flags.plot_folder,f_str2,dataset_config['CHECKPOINT_NAME'],flags.model), fig, ax0)
        fig,ax0 = utils.PlotRoutine(feed_dict_r2,xlabel='Layer number', ylabel= '%s-width' % xlabel1, plot_label = flags.plot_label, cms_style = flags.cms)
        save_fig('{}/{}W_{}_{}'.format(flags.plot_folder,f_str1,dataset_config['CHECKPOINT_NAME'],flags.model), fig, ax0)
        fig,ax0 = utils.PlotRoutine(feed_dict_phi2,xlabel='Layer number', ylabel= '%s-width (radians)' % xlabel2, plot_label = flags.plot_label, cms_style = flags.cms)
        save_fig('{}/{}W_{}_{}'.format(flags.plot_folder,f_str2,dataset_config['CHECKPOINT_NAME'],flags.model), fig, ax0)

        return feed_dict_r2


    def ELayer(data_dict):
        
        def _preprocess(data):
            preprocessed = np.reshape(data,(data.shape[0],shape_plot[2],-1))
            layer_sum = np.sum(preprocessed, -1)
            totalE = np.sum(preprocessed, -1)
            layer_mean = np.mean(layer_sum,0)
            layer_std = np.std(layer_sum,0) / layer_mean
            layer_nonzero = layer_sum > (1e-6 * totalE)
            #preprocessed = np.mean(preprocessed,0)
            return layer_mean, layer_std, layer_nonzero
        
        feed_dict_avg = {}
        feed_dict_std = {}
        feed_dict_nonzero = {}
        for key in data_dict:
            feed_dict_avg[key], feed_dict_std[key], feed_dict_nonzero[key] = _preprocess(data_dict[key])

        fig,ax0 = utils.PlotRoutine(feed_dict_avg,xlabel='Layer number', ylabel= 'Mean dep. energy [GeV]', plot_label = flags.plot_label, no_mean = True, cms_style = flags.cms)
        save_fig('{}/EnergyZ_{}_{}'.format(flags.plot_folder,dataset_config['CHECKPOINT_NAME'],flags.model), fig, ax0)

        fig,ax0 = utils.PlotRoutine(feed_dict_std,xlabel='Layer number', ylabel= 'Std. dev. / Mean of energy [GeV]', plot_label = flags.plot_label, no_mean = True, cms_style = flags.cms)
        save_fig('{}/StdEnergyZ_{}_{}'.format(flags.plot_folder,dataset_config['CHECKPOINT_NAME'],flags.model),fig, ax0)

        fig,ax0 = utils.PlotRoutine(feed_dict_nonzero,xlabel='Layer number', ylabel= 'Freq. > $10^{-6}$ Total Energy', plot_label = flags.plot_label, cms_style = flags.cms)
        save_fig('{}/NonZeroEnergyZ_{}_{}'.format(flags.plot_folder,dataset_config['CHECKPOINT_NAME'],flags.model), fig, ax0)

        return feed_dict_avg


    def SparsityLayer(data_dict):
        
        def _preprocess(data):
            eps = 1e-6
            preprocessed = np.reshape(data,(data.shape[0],shape_plot[2],-1))
            layer_sparsity = np.sum(preprocessed > eps, axis = -1) / preprocessed.shape[2]
            mean_sparsity = np.mean(layer_sparsity, axis = 0)
            std_sparsity = np.std(layer_sparsity, axis = 0)
            #preprocessed = np.mean(preprocessed,0)
            return mean_sparsity, std_sparsity
        
        feed_dict_avg = {}
        feed_dict_std = {}
        feed_dict_nonzero = {}
        for key in data_dict:
            feed_dict_avg[key], feed_dict_std[key] = _preprocess(data_dict[key])

        fig,ax0 = utils.PlotRoutine(feed_dict_avg,xlabel='Layer number', ylabel= 'Mean sparsity', plot_label = flags.plot_label, no_mean = True, cms_style = flags.cms)
        save_fig('{}/SparsityZ_{}_{}'.format(flags.plot_folder,dataset_config['CHECKPOINT_NAME'],flags.model), fig, ax0)

        fig,ax0 = utils.PlotRoutine(feed_dict_std,xlabel='Layer number', ylabel= 'Std. dev. sparsity', plot_label = flags.plot_label, no_mean = True, cms_style = flags.cms)
        save_fig('{}/StdSparsityZ_{}_{}'.format(flags.plot_folder,dataset_config['CHECKPOINT_NAME'],flags.model), fig, ax0)


        return feed_dict_avg


    def RadialEnergyHGCal(data_dict):

        def GetWidth(mean,mean2):
            width = np.ma.sqrt(mean2-mean**2).filled(0)
            return width

        r_vals = NN_embed.geom.ring_map[:, :NN_embed.geom.max_ncell]

         

        feed_dict = {}
        for key in data_dict:
            nrings = NN_embed.geom.nrings
            r_bins = np.zeros((data_dict[key].shape[0], nrings))
            for i in range(nrings):
                mask = (r_vals == i)
                r_bins[:,i] = np.sum(data_dict[key] * mask,axis = (1,2,3))

            feed_dict[key] = r_bins


        fig,ax0 = utils.PlotRoutine(feed_dict, xlabel='R-bin', ylabel= 'Avg. Energy', logy=True, plot_label = flags.plot_label, cms_style = flags.cms)
        save_fig('{}/EnergyR_{}_{}'.format(flags.plot_folder, dataset_config['CHECKPOINT_NAME'],flags.model), fig, ax0)

        return feed_dict



    def RCenterHGCal(data_dict):

        def GetWidth(mean,mean2):
            width = np.ma.sqrt(mean2-mean**2).filled(0)
            return width

        r_vals = NN_embed.geom.ring_map[:, :NN_embed.geom.max_ncell]

        feed_dict_C_hist = {}
        feed_dict_C_avg = {}
        feed_dict_W_hist = {}
        feed_dict_W_avg = {}
        for key in data_dict:
            #center
            r_centers = utils.WeightedMean(r_vals, np.squeeze(data_dict[key]))
            r2_centers = utils.WeightedMean(r_vals, np.squeeze(data_dict[key]), power=2)
            feed_dict_C_hist[key] = np.reshape(r_centers,(-1))
            feed_dict_C_avg[key] = np.mean(r_centers, axis = 0)

            #width
            r_widths = GetWidth(r_centers, r2_centers)
            feed_dict_W_hist[key] = np.reshape(r_widths,(-1))
            feed_dict_W_avg[key] = np.mean(r_widths, axis = 0)

        xlabel = 'R-bin'

        fig,ax0 = utils.HistRoutine(feed_dict_C_hist, xlabel='Shower R Center', ylabel= 'Arbitrary units', plot_label = flags.plot_label, normalize = True, cms_style = flags.cms)
        save_fig('{}/RCenter_{}_{}'.format(flags.plot_folder, dataset_config['CHECKPOINT_NAME'],flags.model), fig, ax0)

        fig,ax0 = utils.PlotRoutine(feed_dict_C_avg, ylabel='Avg. Shower R Center', xlabel= 'Layer', plot_label = flags.plot_label, no_mean = True, cms_style = flags.cms)
        save_fig('{}/RCenterLayer_{}_{}'.format(flags.plot_folder, dataset_config['CHECKPOINT_NAME'],flags.model), fig, ax0)


        fig,ax0 = utils.HistRoutine(feed_dict_W_hist, xlabel='Shower R Width', ylabel= 'Arbitrary units', plot_label = flags.plot_label, normalize = True, cms_style = flags.cms)
        save_fig('{}/RWidth_{}_{}'.format(flags.plot_folder, dataset_config['CHECKPOINT_NAME'],flags.model), fig, ax0)

        fig,ax0 = utils.PlotRoutine(feed_dict_W_avg, ylabel='Avg. Shower R Width', xlabel= 'Layer', plot_label = flags.plot_label, no_mean = True, cms_style = flags.cms)
        save_fig('{}/RWidthLayer_{}_{}'.format(flags.plot_folder, dataset_config['CHECKPOINT_NAME'],flags.model), fig, ax0)
        return feed_dict_C_hist

    def PhiCenterHGCal(data_dict):

        phi_vals = NN_embed.geom.theta_map[:, :NN_embed.geom.max_ncell]

        feed_dict_C_hist = {}
        feed_dict_C_avg = {}
        feed_dict_W_hist = {}
        feed_dict_W_avg = {}
        for key in data_dict:
            #center
            phi_centers, phi_widths = utils.ang_center_spread(phi_vals, np.squeeze(data_dict[key]))
            feed_dict_C_hist[key] = np.reshape(phi_centers,(-1))
            feed_dict_C_avg[key] = np.mean(phi_centers, axis = 0)

            #width
            feed_dict_W_hist[key] = np.reshape(phi_widths,(-1))
            feed_dict_W_avg[key] = np.mean(phi_widths, axis = 0)

        xlabel = 'Phi'

        fig,ax0 = utils.HistRoutine(feed_dict_C_hist, xlabel='Shower Phi Center', ylabel= 'Arbitrary units', plot_label = flags.plot_label, normalize = True, cms_style = flags.cms)
        save_fig('{}/PhiCenter_{}_{}'.format(flags.plot_folder, dataset_config['CHECKPOINT_NAME'],flags.model), fig, ax0)

        fig,ax0 = utils.PlotRoutine(feed_dict_C_avg, ylabel='Avg. Shower Phi Center', xlabel= 'Layer', plot_label = flags.plot_label, no_mean = True, cms_style = flags.cms)
        save_fig('{}/PhiCenterLayer_{}_{}'.format(flags.plot_folder, dataset_config['CHECKPOINT_NAME'],flags.model), fig, ax0)


        fig,ax0 = utils.HistRoutine(feed_dict_W_hist, xlabel='Shower Phi Width', ylabel= 'Arbitrary units', plot_label = flags.plot_label, normalize = True, cms_style = flags.cms)
        save_fig('{}/PhiWidth_{}_{}'.format(flags.plot_folder, dataset_config['CHECKPOINT_NAME'],flags.model), fig, ax0)

        fig,ax0 = utils.PlotRoutine(feed_dict_W_avg, ylabel='Avg. Shower Phi Width', xlabel= 'Layer', plot_label = flags.plot_label, no_mean = True, cms_style = flags.cms)
        save_fig('{}/PhiWidthLayer_{}_{}'.format(flags.plot_folder, dataset_config['CHECKPOINT_NAME'],flags.model), fig, ax0)
        return feed_dict_C_hist



    def AverageER(data_dict):

        def _preprocess(data):
            preprocessed = np.transpose(data,(0,4,1,2,3))
            preprocessed = np.reshape(preprocessed,(data.shape[0],shape_plot[4],-1))
            preprocessed = np.sum(preprocessed,-1)
            return preprocessed
            
        feed_dict = {}
        for key in data_dict:
            feed_dict[key] = _preprocess(data_dict[key])

        if(dataset_config['cartesian_plot']): 
            xlabel = 'x-bin'
            f_str = "X"
        else: 
            xlabel = 'R-bin'
            f_str = "R"

        fig,ax0 = utils.PlotRoutine(feed_dict,xlabel=xlabel, ylabel= 'Mean Energy [GeV]', plot_label = flags.plot_label, cms_style = flags.cms)
        save_fig('{}/Energy{}_{}_{}'.format(flags.plot_folder,f_str, dataset_config['CHECKPOINT_NAME'],flags.model), fig, ax0)
        return feed_dict
        
    def AverageEPhi(data_dict):

        def _preprocess(data):
            preprocessed = np.transpose(data,(0,3,1,2,4))
            preprocessed = np.reshape(preprocessed,(data.shape[0],shape_plot[3],-1))
            preprocessed = np.sum(preprocessed,-1)
            return preprocessed

        feed_dict = {}
        for key in data_dict:
            feed_dict[key] = _preprocess(data_dict[key])

        if(dataset_config['cartesian_plot']): 
            xlabel = 'y-bin'
            f_str = "Y"
        else: 
            xlabel = 'alpha-bin'
            f_str = "Alpha"


        fig,ax0 = utils.PlotRoutine(feed_dict,xlabel=xlabel, ylabel= 'Mean Energy [GeV]', plot_label = flags.plot_label, cms_style = flags.cms)
        save_fig('{}/Energy{}_{}_{}'.format(flags.plot_folder, f_str, dataset_config['CHECKPOINT_NAME'],flags.model), fig, ax0)
        return feed_dict

    def HistEtot(data_dict):
        def _preprocess(data):
            preprocessed = np.reshape(data,(data.shape[0],-1))
            return np.sum(preprocessed,-1)

        feed_dict = {}
        for key in data_dict:
            feed_dict[key] = _preprocess(data_dict[key])

            
        #binning = np.geomspace(np.quantile(feed_dict[geant_key],0.01),np.quantile(feed_dict[geant_key],1.0),20)
        binning = np.geomspace(40.0, np.amax(feed_dict[geant_key]),20)
        fig,ax0 = utils.HistRoutine(feed_dict,xlabel='Deposited energy [GeV]', logy=True,binning=binning, plot_label = flags.plot_label, cms_style = flags.cms)
        save_fig('{}/TotalE_{}_{}'.format(flags.plot_folder,dataset_config['CHECKPOINT_NAME'],flags.model), fig, ax0)
        return feed_dict
        
    def HistNhits(data_dict):

        def _preprocess(data):
            min_voxel = 1e-3
            preprocessed = np.reshape(data,(data.shape[0],-1))
            return np.sum(preprocessed>min_voxel,-1)
        
        feed_dict = {}
        vMax = 0.
        for key in data_dict:
            feed_dict[key] = _preprocess(data_dict[key])
            vMax = max(np.max(feed_dict[key]), vMax)
            
        binning = np.linspace(np.min(feed_dict[geant_key]), vMax, 20)
        fig,ax0 = utils.HistRoutine(feed_dict,xlabel='Number of hits (> 1 MeV)', label_loc='upper right', binning = binning, ratio = True, plot_label = flags.plot_label, cms_style = flags.cms )
        yScalarFormatter = utils.ScalarFormatterClass(useMathText=True)
        yScalarFormatter.set_powerlimits((0,0))
        ax0.yaxis.set_major_formatter(yScalarFormatter)
        save_fig('{}/Nhits_{}_{}'.format(flags.plot_folder,dataset_config['CHECKPOINT_NAME'],flags.model), fig, ax0)
        return feed_dict

    def HistVoxelE(data_dict):

        def _preprocess(data, nShowers):
            nShowers = min(nShowers, data.shape[0])
            return np.reshape(data[:nShowers], (-1))
        
        feed_dict = {}
        nShowers = 1000
        vMax = 0.
        for key in data_dict:
            feed_dict[key] = _preprocess(data_dict[key], nShowers)
            vMax = max(np.max(feed_dict[key]), vMax)
            
        vMin= 1e-4
        binning = np.geomspace(vMin,vMax, 50)
        fig,ax0 = utils.HistRoutine(feed_dict,xlabel='Voxel Energy [GeV]', logy= True, binning = binning, ratio = True, normalize = False,  plot_label = flags.plot_label, cms_style = flags.cms)
        ax0.set_xscale("log")
        save_fig('{}/VoxelE_{}_{}'.format(flags.plot_folder,dataset_config['CHECKPOINT_NAME'],flags.model), fig, ax0)
        return feed_dict


    def HistMaxELayer(data_dict):

        def _preprocess(data):
            preprocessed = np.reshape(data,(data.shape[0],shape_plot[2],-1))
            preprocessed = np.ma.divide(np.max(preprocessed,-1),np.sum(preprocessed,-1)).filled(0)
            return preprocessed


        feed_dict = {}
        for key in data_dict:
            feed_dict[key] = _preprocess(data_dict[key])

        fig,ax0 = utils.PlotRoutine(feed_dict,xlabel='Layer number', ylabel= 'Max voxel/Dep. energy', plot_label = flags.plot_label, cms_style = flags.cms)
        save_fig('{}/MaxEnergyZ_{}_{}'.format(flags.plot_folder,dataset_config['CHECKPOINT_NAME'],flags.model), fig, ax0)
        return feed_dict

    def HistMaxE(data_dict):

        def _preprocess(data):
            preprocessed = np.reshape(data,(data.shape[0],-1))
            preprocessed = np.ma.divide(np.max(preprocessed,-1),np.sum(preprocessed,-1)).filled(0)
            return preprocessed


        feed_dict = {}
        for key in data_dict:
            feed_dict[key] = _preprocess(data_dict[key])

        binning = np.linspace(0,1,10)
        fig,ax0 = utils.HistRoutine(feed_dict,xlabel= 'Max. voxel/Dep. energy',binning=binning,logy=True, plot_label = flags.plot_label, cms_style = flags.cms)
        save_fig('{}/MaxEnergy_{}_{}'.format(flags.plot_folder,dataset_config['CHECKPOINT_NAME'],flags.model), fig, ax0)
        return feed_dict
        



    def plot_shower(shower, fout = "", title = "", vmax = 0, vmin = 0):
        #cmap = plt.get_cmap('PiYG')
        cmap = copy.copy(plt.get_cmap('viridis'))
        cmap.set_bad("white")

        shower[shower==0]=np.nan

        fig,ax = utils.SetFig("x-bin","y-bin")
        if vmax==0:
            vmax = np.nanmax(shower[:,:,0])
            vmin = np.nanmin(shower[:,:,0])
            #print(vmin,vmax)
        im = ax.pcolormesh(range(shower.shape[0]), range(shower.shape[1]), shower[:,:,0], cmap=cmap,vmin=vmin,vmax=vmax)

        yScalarFormatter = utils.ScalarFormatterClass(useMathText=True)
        yScalarFormatter.set_powerlimits((0,0))
        #cbar.ax.set_major_formatter(yScalarFormatter)

        cbar=fig.colorbar(im, ax=ax,label='Dep. energy [GeV]',format=yScalarFormatter)
        
        
        bar = ax.set_title(title,fontsize=15)

        if(len(fout) > 0): fig.savefig(fout)
        return vmax, vmin



    def Plot_Shower_2D(data_dict):
        plt.rcParams['pcolor.shading'] ='nearest'
        layer_number = [10,44]



        for layer in layer_number:
            
            def _preprocess(data):
                preprocessed = data[:,layer,:]
                preprocessed = np.mean(preprocessed,0)
                preprocessed[preprocessed==0]=np.nan
                return preprocessed

            vmin=vmax=0
            nShowers = 5
            for ik,key in enumerate([geant_key,utils.name_translate[flags.model]]):
                average = _preprocess(data_dict[key])

                fout_avg = '{}/{}2D_{}_{}_{}.{}'.format(flags.plot_folder,key,layer,dataset_config['CHECKPOINT_NAME'],flags.model, plt_exts[0])
                title = "{}, layer number {}".format(key,layer)
                plot_shower(average, fout = fout_avg, title = title)

                for i in range(nShowers):
                    shower = data_dict[key][i,layer]
                    fout_ex = '{}/{}2D_{}_{}_{}_shower{}.{}'.format(flags.plot_folder,key,layer,dataset_config['CHECKPOINT_NAME'],flags.model, i, plt_exts[0])
                    title = "{} Shower {}, layer number {}".format(key, i, layer)
                    vmax, vmin = plot_shower(shower, fout = fout_ex, title = title, vmax = vmax, vmin = vmin)


            

    #do_cart_plots = (not dataset_config['CYLINDRICAL']) and dataset_config['SHAPE_PAD'][-1] == dataset_config['SHAPE_PAD'][-2]
    do_cart_plots = False
    dataset_config['cartesian_plot'] = do_cart_plots
    print("Do cartesian plots " + str(do_cart_plots))
    high_level = []
    plot_routines = {
         'Energy per layer':ELayer,
         'Energy':HistEtot,
         '2D Energy scatter split':ScatterESplit,
         'Energy Ratio split':HistERatio,
         'Layer Sparsity' : SparsityLayer,
    }
    if(flags.geant_only):
        plot_routines['IncidentE split'] = IncidentE


    if(hgcal and not flags.plot_reshape):
        plot_routines['Energy R']=RadialEnergyHGCal
        plot_routines['Energy R Center']=RCenterHGCal
        plot_routines['Energy Phi Center']=PhiCenterHGCal
        plot_routines['Nhits'] = HistNhits
        plot_routines['Max voxel']=HistMaxELayer
        plot_routines['VoxelE'] = HistVoxelE

    elif(not flags.layer_only):
        plot_routines['Energy per radius']=AverageER
        plot_routines['Energy per phi']=AverageEPhi
        plot_routines['Nhits'] = HistNhits
        plot_routines['Shower width']=AverageShowerWidth
        plot_routines['Max voxel']=HistMaxELayer
        plot_routines['VoxelE'] = HistVoxelE

    if(do_cart_plots):
        plot_routines['2D average shower']=Plot_Shower_2D

    if(flags.cms): hep.style.use(hep.style.CMS)

    print("Saving plots to "  + os.path.abspath(flags.plot_folder) )
    for plot in plot_routines:
        if '2D' in plot and flags.model == 'all':continue #skip scatter plots superimposed
        print(plot)
        if 'split' in plot:
            plot_routines[plot](data_dict,energies)
        else:
            high_level.append(plot_routines[plot](data_dict))

if(__name__ == "__main__"):

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_folder', default='../data/', help='Folder containing data and MC files')
    parser.add_argument('--plot_folder', default='../plots', help='Folder to save results')
    parser.add_argument('--generated', '-g', default='', help='Generated showers')
    parser.add_argument('--model_loc', default='test', help='Location of model')
    parser.add_argument('--config', '-c', default='config_dataset2.json', help='Training parameters')
    parser.add_argument('-n', '--nevts', type=int,default=-1, help='Number of events to load')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for generation')
    parser.add_argument('--model', default='Diffu', help='Diffusion model to load. Options are: Diffu, AE, all')
    parser.add_argument('--plot_label', default='', help='Add to plot')
    parser.add_argument('--EMin', type = float, default=-1.0, help='Voxel min energy')

    parser.add_argument('-s', '--sparsity_classifier', default='',  help='Sparsity classifier')
    parser.add_argument('--layer_only', default=False, action= 'store_true', help='Only sample layer energies')
    parser.add_argument('--layer_model', default='', help='Location of model for layer energies')
    parser.add_argument('--plot_reshape', default=False, action='store_true', help='Plots in emebedded space of HGCal')

    parser.add_argument('--cms', default=False, action='store_true', help='Plots in CMS style')

    parser.add_argument('--geant_only', action='store_true', default=False,help='Plots with just geant')

    flags = parser.parse_args()
    make_plots(flags)

        
