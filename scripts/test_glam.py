
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import argparse
import h5py as h5
import torch.optim as optim
import torch.utils.data as torchdata


from utils import *
from HGCal_utils import *
from Dataset import Dataset
from CaloDiffu import *
from models import *


#plt_ext = "pdf"

def save_fig(fname, fig, ax0):
    plt_exts = [".png", ".pdf", "_logy.png", "_logy.pdf"]
    for plt_ext in plt_exts:
        if('logy') in plt_ext: ax0.set_yscale("log")
        else: ax0.set_yscale("linear")
        fig.savefig(fname + plt_ext)

def AverageER(data_dict, flags):

    def _preprocess(data):
        preprocessed = np.transpose(data,(0,4,1,2,3))
        preprocessed = np.reshape(preprocessed,(data.shape[0],data.shape[4],-1))
        num_voxels_per_radius = preprocessed.shape[2]
        preprocessed = np.sum(preprocessed,-1)/num_voxels_per_radius
        return preprocessed
        
    feed_dict = {}
    for key in data_dict:
        feed_dict[key] = _preprocess(data_dict[key])
        print(feed_dict[key].shape)
        print(np.mean(feed_dict[key]))

    xlabel = 'R-bin'
    f_str = "R"

    fig,ax0 = PlotRoutine(feed_dict,xlabel=xlabel, ylabel= 'Mean Energy [GeV]', plot_label = flags.plot_label)
    save_fig('{}/Energy{}'.format(flags.plot_folder,f_str), fig, ax0)
    return feed_dict



if __name__ == '__main__':
    print("TRAIN DIFFU")

    if(torch.cuda.is_available()): device = torch.device('cuda')
    else: device = torch.device('cpu')
        
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_folder', default='../data/', help='Folder containing data and MC files')
    parser.add_argument('--plot_folder', default='', help='Folder containing data and MC files')
    parser.add_argument('--label', default='', help='Folder containing data and MC files')
    parser.add_argument('--model', default='Diffu', help='Diffusion model to train. Options are: VPSDE, VESDE and subVPSDE')
    parser.add_argument('-c', '--config', default='configs/test.json', help='Config file with training parameters')
    parser.add_argument('-n', '--nevts', type=int,default=-1, help='Number of events to load')
    parser.add_argument('--frac', type=float,default=0.85, help='Fraction of total events used for training')
    parser.add_argument('--load', action='store_true', default=False,help='Load pretrained weights to continue the training')
    parser.add_argument('--seed', type=int, default=1234,help='Pytorch seed')
    parser.add_argument('--num_workers', type=int, default=0,help='Num pytorch workers')
    parser.add_argument('--reclean', action='store_true', default=False,help='Reclean data')
    parser.add_argument('--reset_training', action='store_true', default=False,help='Retrain')
    flags = parser.parse_args()

    dataset_config = LoadJson(flags.config)

    print("TRAINING OPTIONS")
    print(dataset_config, flush = True)

    torch.manual_seed(flags.seed)

    cold_diffu = dataset_config.get('COLD_DIFFU', False)
    cold_noise_scale = dataset_config.get('COLD_NOISE', 1.0)

    nholdout  = dataset_config.get('HOLDOUT', 0)

    batch_size = dataset_config['BATCH']
    num_epochs = dataset_config['MAXEPOCH']
    early_stop = dataset_config['EARLYSTOP']
    training_obj = dataset_config.get('TRAINING_OBJ', 'noise_pred')
    loss_type = dataset_config.get("LOSS_TYPE", "l2")
    dataset_num = dataset_config.get('DATASET_NUM', 2)
    shower_embed = dataset_config.get('SHOWER_EMBED', '')

    hgcal = dataset_config.get('HGCAL', False)
    geom_file = dataset_config.get('BIN_FILE', '')
    orig_shape = ('orig' in shower_embed)
    layer_norm = 'layer' in dataset_config['SHOWERMAP']
    max_cells = dataset_config.get('MAX_CELLS', None)
    shape_plot = dataset_config['SHAPE_FINAL']

    train_files = []
    val_files = []


    dataset = dataset_config['FILES'][0]

    showers,E,layers = DataLoader(
        os.path.join(flags.data_folder,dataset),
        dataset_config['SHAPE_PAD'],
        emax = dataset_config['EMAX'],emin = dataset_config['EMIN'],
        hgcal = hgcal,
        nevts = 250,
        max_deposit=dataset_config['MAXDEP'], #noise can generate more deposited energy than generated
        logE=dataset_config['logE'],
        showerMap = dataset_config['SHOWERMAP'],
        max_cells = max_cells,

        dataset_num  = dataset_num,
        orig_shape = orig_shape,
    )

    layers = np.reshape(layers, (layers.shape[0], -1))
    if(orig_shape): showers = np.reshape(showers, dataset_config['SHAPE_ORIG'])
    else : showers = np.reshape(showers, dataset_config['SHAPE_PAD'])

    print("Pre-processed shower mean %.2f std dev %.2f" % (np.mean(showers), np.std(showers)))



    """
    print("Simple Geo")
    geom_f = open(geom_file, 'rb')
    geom = pickle_load(geom_f)
    geom.theta_map = np.arctan2(geom.xmap, geom.ymap) % (2. *np.pi)
    print(geom.ring_map[0,:23])
    print(geom.theta_map[0,:23])
    #first two layers
    geom.ncells = [23]
    geom.max_ncell = 23

    map_test, mask_test = init_map(12, 3, geom, 0)
    for i in range(geom.max_ncell):
        print(torch.sum(map_test[:,:,i]))

    print(map_test)
    print(mask_test)
    """




    if(hgcal):
        trainable = dataset_config.get('TRAINABLE_EMBED', False)
        print("Trainable", trainable)
        NN_embed = HGCalConverter(bins = dataset_config['SHAPE_FINAL'], geom_file = geom_file, device = device, trainable = trainable).to(device = device)
        NN_embed.init(norm = False, dataset_num = dataset_num)

    else:
        if(dataset_num == 1):
            bins = XMLHandler("photon", geom_file)
        else: 
            bins = XMLHandler("pion", geom_file)
            NN_embed = NNConverter(bins = bins).to(device = device)


    flags.plot_folder = "../plots/test_glam2/"
    flags.plot_label = ""
    showers = torch.Tensor(showers).to(device)

    print("OG")
    print(showers[0].shape)
    enc0 = NN_embed.enc(showers)

    print("Embed mean and std", torch.mean(enc0), torch.std(enc0))

    data_dict = {}
    data_dict['Geant4 (CMSSW)'] = enc0.detach().cpu().numpy()


    AverageER(data_dict, flags)

    plt.figure(figsize=(10,10))
    plt.hist(data_dict['Geant4 (CMSSW)'].reshape(-1), bins=100)
    plt.savefig(flags.plot_folder + "emb_voxel_dist.png")




    print("ENC")
    print(enc0[0].shape)

    shower_dec = NN_embed.dec(enc0)
    diff = torch.mean(showers[:,:,:,:1000] - shower_dec[:,:,:,:1000])
    print("Avg. Diff " + str(diff))


    dummy_showers = torch.zeros_like(showers)
    dummy_showers[:, :, :, 1660] = 1.0
    enc_dummy = NN_embed.enc(dummy_showers)
    dec_dummy = NN_embed.dec(enc_dummy)

    eps = 1e-6
    print("test with dummy")
    print(dummy_showers.shape)
    print(dec_dummy[0,0,10,1660])
    print(torch.sum(dec_dummy[0,0,10,:]))
    print(torch.sum(dec_dummy[0,0,10,:] > eps ))



    enc_mat = NN_embed.enc_mat.detach().cpu().numpy()
    print('enc map', NN_embed.enc_mat.shape)
    num_alpha_bins = shape_plot[-2]
    num_r_bins = shape_plot[-1]
    enc_mat_reshape = np.reshape(enc_mat, (enc_mat.shape[0], num_alpha_bins, num_r_bins, -1))
    print(enc_mat_reshape.shape)
    print(np.sum(enc_mat_reshape[10,:,19,:]))
    print(np.sum(enc_mat_reshape[10,0,20] > 0))
    non_zeros = np.argwhere(enc_mat_reshape[10,0,20] > 0)
    print(non_zeros)
    print(np.sum(enc_mat_reshape[10,:,:, non_zeros[0]] > 0))
    print(np.sum(enc_mat_reshape[10,:,:, non_zeros[5]] > 0))

    dec_mat = NN_embed.dec_mat.detach().cpu().numpy()
    print('dec map', NN_embed.dec_mat.shape)
    dec_mat_reshape = np.reshape(dec_mat, (enc_mat.shape[0], -1, num_alpha_bins, num_r_bins))
    print(dec_mat_reshape.shape)
    print(np.sum(dec_mat_reshape[10,0,:2,: ]))
    print(np.sum(dec_mat_reshape[10,0]))
    print(np.sum(dec_mat_reshape[10,200]))
    print(np.sum(dec_mat_reshape[10,:,0,20]))
    print(np.sum(dec_mat_reshape[10,:,0,20] > eps))
    print(np.sum(dec_mat_reshape[10,non_zeros[0]] > eps))
    print(np.sum(dec_mat_reshape[10,non_zeros[5]] > eps))
    #dec_nonzeros = np.argwhere(dec_mat_reshape[10,non_zeros[0]] > 0)
    dec_nonzeros = np.argwhere(dec_mat[10,non_zeros[0]] > eps)
    print(dec_nonzeros[:10])

    plt.figure(figsize=(10,10))
    vals = dec_mat_reshape[10,non_zeros[0]].reshape(-1)
    plt.hist(vals, bins = 100)
    plt.yscale("log")
    plt.savefig(flags.plot_folder + "cell_hist.png")


    plt.figure(figsize=(10,10))
    plt.matshow(dec_mat[10])
    plt.savefig(flags.plot_folder + "dec_mat.png")


    plt.figure(figsize=(10,10))
    plt.yscale("log")
    plt.hist(dec_mat.reshape(-1), bins=100)
    plt.savefig(flags.plot_folder + "dec_mat_hist.png")

    plt.figure(figsize=(10,10))
    plt.yscale("log")
    plt.hist(enc_mat.reshape(-1), bins=100)
    plt.savefig(flags.plot_folder + "enc_mat_hist.png")


