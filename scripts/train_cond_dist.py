import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import argparse
import h5py as h5
import torch.optim as optim
import torch.utils.data as torchdata
from ema_pytorch import EMA

from utils import *
from ControlNet import *

def compute_cond_dist_loss(x, E, t = None, layers = None,  model = None, ema_model = None):   

    noise = torch.randn_like(x)
    if(t is None): t = torch.randint(1, nsteps, (x.size()[0],), device=x.device).long()

    sigma = extract(model.UNet.sqrt_one_minus_alphas_cumprod, t, x.shape) / extract(model.UNet.sqrt_alphas_cumprod, t, x.shape)
    sigma_prev = extract(model.UNet.sqrt_one_minus_alphas_cumprod, t-1, x.shape) / extract(model.UNet.sqrt_alphas_cumprod, t-1, x.shape)

    avg_showers, std_showers =model.UNet.lookup_avg_std_shower(E)
    c_x = avg_showers

    x_noisy = x + sigma * noise
    sigma2 = sigma**2

    #predict x0 and noise using model
    x0 = model.denoise(x_noisy, c_x = c_x, E =E, sigma=sigma, layers = layers)
    noise_pred = (x_noisy - x0)/sigma

    #Use true noise plus estimated x0 to construct new sample
    x_noisy_new = x0 + sigma_prev * noise


    #predict noise using ema model constructed from pred. x0 
    x0_ema = ema_model.model.denoise(x_noisy_new, c_x, E, sigma_prev, layers = layers)

    noise_pred_ema = (x_noisy_new - x0_ema) / sigma_prev


    #combine two terms for loss
    loss = torch.nn.functional.mse_loss(x, x0) + torch.nn.functional.mse_loss(noise_pred_ema, noise_pred)
    return loss



if __name__ == '__main__':
    print("TRAIN COND DIST")

    if(torch.cuda.is_available()): device = torch.device('cuda')
    else: device = torch.device('cpu')
        
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_folder', default='../data/', help='Folder containing data and MC files')
    parser.add_argument('--model', default='Diffu', help='Diffusion model to train. Options are: VPSDE, VESDE and subVPSDE')
    parser.add_argument('--sample_algo', default='ddim', help='Sampling algorithm for consistency training')
    parser.add_argument('-c', '--config', default='configs/test.json', help='Config file with training parameters')
    parser.add_argument('--nevts', type=float,default=-1, help='Number of events to load')
    parser.add_argument('--frac', type=float,default=0.85, help='Fraction of total events used for training')
    parser.add_argument('--load', action='store_true', default=False,help='Load pretrained weights to continue the training')
    parser.add_argument('--freeze_baseline_model', action='store_true', default=False,help='Only optimize control net, leaving base diffusion model frozen')
    parser.add_argument('--control_only', action='store_true', default=False,help='Only optimize control net, leaving base diffusion model frozen')
    parser.add_argument('--seed', type=int, default=123,help='Pytorch seed')
    parser.add_argument('--reset_training', action='store_true', default=False,help='Retrain')
    flags = parser.parse_args()

    dataset_config = LoadJson(flags.config)
    data = []
    energies = []

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
    orig_shape = ('orig' in shower_embed)
    layer_norm = 'layer' in dataset_config['SHOWERMAP']

    sample_steps = dataset_config.get('CONSIS_NSTEPS', 100)


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
            if(layer_norm): layers = layers_
        else:
            data = np.concatenate((data, data_))
            energies = np.concatenate((energies, e_))
            if(layer_norm): layers = np.concatenate((layers, layers_))

    NN_embed = None
    if('NN' in shower_embed):
        if(dataset_num == 1):
            binning_file = "../CaloChallenge/code/binning_dataset_1_photons.xml"
            bins = XMLHandler("photon", binning_file)
        else: 
            binning_file = "../CaloChallenge/code/binning_dataset_1_pions.xml"
            bins = XMLHandler("pion", binning_file)

        NN_embed = NNConverter(bins = bins).to(device = device)

        
    dshape = dataset_config['SHAPE_PAD']
    if(layer_norm): layers = np.reshape(layers, (layers.shape[0], -1))
    if(not orig_shape): data = np.reshape(data,dshape)
    else: data = np.reshape(data, (len(data), -1))

    num_data = data.shape[0]
    #print("Pre-processed shower mean %.2f std dev %.2f" % (np.mean(data), np.std(data)))
    torch_data_tensor = torch.from_numpy(data)
    torch_E_tensor = torch.from_numpy(energies)
    torch_layer_tensor =  torch.from_numpy(layers) if layer_norm else torch.zeros_like(torch_E_tensor)
    del data
    #train_data, val_data = utils.split_data_np(data,flags.frac)

    torch_dataset  = torchdata.TensorDataset(torch_E_tensor, torch_layer_tensor, torch_data_tensor)
    nTrain = int(round(flags.frac * num_data))
    nVal = num_data - nTrain
    train_dataset, val_dataset = torch.utils.data.random_split(torch_dataset, [nTrain, nVal])

    loader_train = torchdata.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    loader_val = torchdata.DataLoader(val_dataset, batch_size = batch_size, shuffle = True)

    del torch_data_tensor, torch_E_tensor, train_dataset, val_dataset



    f_avg_shower = h5.File(flags.data_folder + dataset_config["AVG_SHOWER_LOC"])
    #Already pre-processed
    avg_showers = torch.from_numpy(f_avg_shower["avg_showers"][()].astype(np.float32)).to(device = device)
    std_showers = torch.from_numpy(f_avg_shower["std_showers"][()].astype(np.float32)).to(device = device)
    E_bins = torch.from_numpy(f_avg_shower["E_bins"][()].astype(np.float32)).to(device = device)




    checkpoint_folder = '../models/{}_{}/'.format(dataset_config['CHECKPOINT_NAME'],flags.model)

    #load teacher diffusion model
    diffu_checkpoint_path = os.path.join(checkpoint_folder, "checkpoint.pth")
    print("Loading teacher diffu model from %s" % diffu_checkpoint_path)
    diffu_checkpoint = torch.load(diffu_checkpoint_path, map_location = device)

    shape = dataset_config['SHAPE_PAD'][1:] if (not orig_shape) else dataset_config['SHAPE_ORIG'][1:]
    diffu_model = CaloDiffu(shape, config=dataset_config, training_obj = training_obj, NN_embed = NN_embed, nsteps = sample_steps, 
            E_bins = E_bins, avg_showers = avg_showers, std_showers = std_showers).to(device = device)
    diffu_model.eval()

    if('model_state_dict' in diffu_checkpoint.keys()): diffu_model.load_state_dict(diffu_checkpoint['model_state_dict'])
    elif(len(diffu_checkpoint.keys()) > 1): diffu_model.load_state_dict(diffu_checkpoint)

    #init controlnet model 
    controlnet_model = copy.deepcopy(diffu_model)
    #Upsampling layers not used
    controlnet_model.model.ups = None


    model = ControlledUNet(diffu_model, controlnet_model)
    model.train()
    #Freeze UNet part
    if(flags.freeze_baseline_model): model.UNet.model.eval()


    checkpoint = dict()
    checkpoint_path = os.path.join(checkpoint_folder, "controlnet_checkpoint.pth")
    if(flags.load and os.path.exists(checkpoint_path)): 
        print("Loading controlnet model from %s" % checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location = device)
        model.load_state_dict(checkpoint['model_state_dict'])


    if(not flags.control_only):
        ema_model = EMA(model, 
                    beta = 0.95,              # exponential moving average factor
                    update_after_step = 10,    # only after this number of .update() calls will it start updating
                    update_every = 1,          # how often to actually update
                   )
        ema_model.eval()
    else: ema_model = None




    early_stopper = EarlyStopper(patience = dataset_config['EARLYSTOP'], mode = 'diff', min_delta = 1e-5)
    if('early_stop_dict' in checkpoint.keys() and not flags.reset_training): early_stopper.__dict__ = checkpoint['early_stop_dict']
    print(early_stopper.__dict__)
    
    #optimizer on only controlnet or full model (not EMA)
    if(flags.freeze_baseline_model): optimizer = optim.Adam(controlnet_model.model.parameters(), lr = float(dataset_config["LR"]))
    else: optimizer = optim.Adam(model.parameters(), lr = float(dataset_config["LR"]))


    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer = optimizer, factor = 0.1, patience = 15, verbose = True) 
    if('optimizer_state_dict' in checkpoint.keys() and not flags.reset_training): optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if('scheduler_state_dict' in checkpoint.keys() and not flags.reset_training): scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    training_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    start_epoch = 0
    min_validation_loss = 99999.
    if('train_loss_hist' in checkpoint.keys()): 
        training_losses = checkpoint['train_loss_hist']
        val_losses = checkpoint['val_loss_hist']
        start_epoch = checkpoint['epoch'] + 1

    print("Using %i steps for cond_dist training" % sample_steps)
    #training loop
    for epoch in range(start_epoch, num_epochs):
        print("Beginning epoch %i" % epoch, flush=True)
        train_loss = 0

        if(flags.freeze_baseline_model): model.ControlNet.model.train()
        else: model.train()
        for i, (E,layers, data) in tqdm(enumerate(loader_train, 0), unit="batch", total=len(loader_train)):
            model.zero_grad()
            optimizer.zero_grad()

            data = data.to(device = device)
            E = E.to(device = device)
            layers = layers.to(device = device)

            if(flags.control_only):
                batch_loss = model.compute_loss(data, E, layers = layers, loss_type = loss_type )
            else:
                t = torch.repeat_interleave(torch.randint(1, sample_steps, (1,), device=device).long(), data.size()[0])
                batch_loss = compute_cond_dist_loss(data, E, t=t, layers = layers, model = model, ema_model = ema_model)




            batch_loss.backward()
            optimizer.step()

            if(ema_model is not None): ema_model.update()

            train_loss+=batch_loss.item()

            del data, E, batch_loss

        train_loss = train_loss/len(loader_train)
        training_losses[epoch] = train_loss
        print("loss: "+ str(train_loss))

        val_loss = 0
        if(flags.freeze_baseline_model): model.ControlNet.model.eval()
        else: model.eval()
        for i, (vE, vlayers, vdata) in tqdm(enumerate(loader_val, 0), unit="batch", total=len(loader_val)):
            vdata = vdata.to(device=device)
            vE = vE.to(device = device)
            vlayers = vlayers.to(device = device)

            if(flags.control_only):
                batch_loss = model.compute_loss(vdata, vE, layers = vlayers, loss_type = loss_type )
            else:
                t = torch.repeat_interleave(torch.randint(1, sample_steps, (1,), device=device).long(), data.size()[0])
                batch_loss = compute_cond_dist_loss(vdata, vE, t=t, layers = vlayers, model = model, ema_model = ema_model)


            val_loss+=batch_loss.item()
            del vdata,vE, batch_loss

        val_loss = val_loss/len(loader_val)
        #scheduler.step(torch.tensor([val_loss]))
        val_losses[epoch] = val_loss
        print("val_loss: "+ str(val_loss), flush = True)

        scheduler.step(torch.tensor([train_loss]))


        if(val_loss < min_validation_loss):
            val_path = os.path.join(checkpoint_folder, 'controlnet_best_val.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict() if flags.control_only else ema_model.ema_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss_hist': training_losses,
                'val_loss_hist': val_losses,
                'early_stop_dict': early_stopper.__dict__,
                }, val_path)

            min_validation_loss = val_loss

        if(early_stopper.early_stop(val_loss - train_loss)):
            print("Early stopping!")
            break
        print(early_stopper.__dict__)

        # save the model
        model.eval()
        print("SAVING")
        #torch.save(model.state_dict(), checkpoint_path)
        
        #save full training state so can be resumed
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict() if flags.control_only else ema_model.ema_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss_hist': training_losses,
            'val_loss_hist': val_losses,
            'early_stop_dict': early_stopper.__dict__,
            }, checkpoint_path)

        with open(checkpoint_folder + "/training_losses.txt","w") as tfileout:
            tfileout.write("\n".join("{}".format(tl) for tl in training_losses)+"\n")
        with open(checkpoint_folder + "/validation_losses.txt","w") as vfileout:
            vfileout.write("\n".join("{}".format(vl) for vl in val_losses)+"\n")


    print("Saving to %s" % checkpoint_folder, flush=True)

    final_path = os.path.join(checkpoint_folder, 'controlnet_final.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict() if flags.control_only else ema_model.ema_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss_hist': training_losses,
        'val_loss_hist': val_losses,
        'early_stop_dict': early_stopper.__dict__,
        }, final_path)

    with open(checkpoint_folder + "/training_losses.txt","w") as tfileout:
        tfileout.write("\n".join("{}".format(tl) for tl in training_losses)+"\n")
    with open(checkpoint_folder + "/validation_losses.txt","w") as vfileout:
        vfileout.write("\n".join("{}".format(vl) for vl in val_losses)+"\n")

