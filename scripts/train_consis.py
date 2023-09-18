import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import argparse
import h5py as h5
import torch.optim as optim
import torch.utils.data as torchdata

from utils import *
from ConsistencyModel import *
from CaloDiffu import *
from models import *

def compute_consis_loss(x, E, t = None, teacher_model = None, model = None, ema_model = None, nsteps = 400, sample_algo = 'ddim'):   

    noise = torch.randn_like(x)
    if(t is None): t = torch.randint(1, nsteps, (x.size()[0],), device=x.device).long()

    sigma = extract(teacher_model.sqrt_one_minus_alphas_cumprod, t, x.shape) / extract(teacher_model.sqrt_alphas_cumprod, t, x.shape)
    sigma_prev = extract(teacher_model.sqrt_one_minus_alphas_cumprod, t-1, x.shape) / extract(teacher_model.sqrt_alphas_cumprod, t-1, x.shape)

    #else:
    #    rnd_normal = torch.randn((x.size()[0],), device=x.device)
    #    sigma = (rnd_normal * model.P_std + model.P_mean).exp().reshape(x.shape[0], 1,1,1,1)

    x_noisy = x + sigma * noise
    sigma2 = sigma**2


    with torch.no_grad():
        #denoise 1 step using fixed diffusion model
        x_prev = teacher_model.p_sample(x_noisy, E, t, sample_algo = sample_algo)

        #predict using ema model on one-step denoised x
        x0_ema = model.denoise(x_prev, E, sigma_prev, model = ema_model.model.model)

    #predict using model on noisy x
    x0 = model.denoise(x_noisy, E,sigma, model = model.model)

    loss = torch.nn.functional.mse_loss(x0_ema, x0)
    return loss



if __name__ == '__main__':
    print("TRAIN CONSIS")

    if(torch.cuda.is_available()): device = torch.device('cuda')
    else: device = torch.device('cpu')
        
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_folder', default='../data/', help='Folder containing data and MC files')
    parser.add_argument('--model', default='Diffu', help='Diffusion model to train. Options are: VPSDE, VESDE and subVPSDE')
    parser.add_argument('--sample_algo', default='ddim', help='Sampling algorithm for consistency training')
    parser.add_argument('--sample_steps', type=int, default=200,help='How many discrete steps for sampler of consitency training')
    parser.add_argument('-c', '--config', default='configs/test.json', help='Config file with training parameters')
    parser.add_argument('--nevts', type=float,default=-1, help='Number of events to load')
    parser.add_argument('--frac', type=float,default=0.85, help='Fraction of total events used for training')
    parser.add_argument('--load', action='store_true', default=False,help='Load pretrained weights to continue the training')
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


    for i, dataset in enumerate(dataset_config['FILES']):
        data_,e_ = DataLoader(
            os.path.join(flags.data_folder,dataset),
            dataset_config['SHAPE_PAD'],
            emax = dataset_config['EMAX'],emin = dataset_config['EMIN'],
            nevts = flags.nevts,
            max_deposit=dataset_config['MAXDEP'], #noise can generate more deposited energy than generated
            logE=dataset_config['logE'],
            showerMap = dataset_config['SHOWERMAP'],
            nholdout = nholdout if (i == len(dataset_config['FILES']) -1 ) else 0,
            dataset_num  = dataset_num,
        )


        if(i ==0): 
            data = data_
            energies = e_
        else:
            data = np.concatenate((data, data_))
            energies = np.concatenate((energies, e_))
        
    dshape = dataset_config['SHAPE_PAD']
    energies = np.reshape(energies,(-1))    
    data = np.reshape(data,dshape)
    num_data = data.shape[0]
    print("Data Shape " + str(data.shape))
    data_size = data.shape[0]
    #print("Pre-processed shower mean %.2f std dev %.2f" % (np.mean(data), np.std(data)))
    torch_data_tensor = torch.from_numpy(data)
    torch_E_tensor = torch.from_numpy(energies)
    del data
    #train_data, val_data = utils.split_data_np(data,flags.frac)

    torch_dataset  = torchdata.TensorDataset(torch_E_tensor, torch_data_tensor)
    nTrain = int(round(flags.frac * num_data))
    nVal = num_data - nTrain
    train_dataset, val_dataset = torch.utils.data.random_split(torch_dataset, [nTrain, nVal])

    loader_train = torchdata.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    loader_val = torchdata.DataLoader(val_dataset, batch_size = batch_size, shuffle = True)

    del torch_data_tensor, torch_E_tensor, train_dataset, val_dataset

    NN_embed = None
    if('NN' in shower_embed):
        if(dataset_num == 1):
            binning_file = "../CaloChallenge/code/binning_dataset_1_photons.xml"
            bins = XMLHandler("photon", binning_file)
        else: 
            binning_file = "../CaloChallenge/code/binning_dataset_1_pions.xml"
            bins = XMLHandler("pion", binning_file)

        NN_embed = NNConverter(bins = bins).to(device = device)




    checkpoint_folder = '../models/{}_{}/'.format(dataset_config['CHECKPOINT_NAME'],flags.model)

    #load teacher diffusion model
    diffu_checkpoint_path = os.path.join(checkpoint_folder, "checkpoint.pth")
    print("Loading teacher diffu model from %s" % diffu_checkpoint_path)
    diffu_checkpoint = torch.load(diffu_checkpoint_path, map_location = device)

    shape = dataset_config['SHAPE_PAD'][1:] if (not orig_shape) else dataset_config['SHAPE_ORIG'][1:]
    diffu_model = CaloDiffu(shape, config=dataset_config, training_obj = training_obj, NN_embed = NN_embed, nsteps = dataset_config['NSTEPS'],).to(device = device)
    diffu_model.eval()

    if('model_state_dict' in diffu_checkpoint.keys()): diffu_model.load_state_dict(diffu_checkpoint['model_state_dict'])
    elif(len(diffu_checkpoint.keys()) > 1): diffu_model.load_state_dict(diffu_checkpoint)

    #init consistency model
    consis_model = copy.deepcopy(diffu_model)

    checkpoint = dict()
    if(flags.load):
        checkpoint_path = os.path.join(checkpoint_folder, "consis_checkpoint.pth")
        print("Loading consis model from %s" % checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location = device)
        consis_model.load_state_dict(checkpoint['model_state_dict'])


    ema_model = EMA(consis_model, 
                beta = 0.95,              # exponential moving average factor
                update_after_step = 10,    # only after this number of .update() calls will it start updating
                update_every = 1,          # how often to actually update
               )




    checkpoint_path = os.path.join(checkpoint_folder, "consis_checkpoint.pth")

    early_stopper = EarlyStopper(patience = dataset_config['EARLYSTOP'], mode = 'diff', min_delta = 1e-5)
    if('early_stop_dict' in checkpoint.keys() and not flags.reset_training): early_stopper.__dict__ = checkpoint['early_stop_dict']
    print(early_stopper.__dict__)
    

    criterion = nn.MSELoss().to(device = device)

    #optimizer only on moving model (not EMA)
    optimizer = optim.Adam(consis_model.model.parameters(), lr = float(dataset_config["LR"]))
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

    #training loop
    for epoch in range(start_epoch, num_epochs):
        print("Beginning epoch %i" % epoch, flush=True)
        train_loss = 0

        consis_model.model.train()
        ema_model.eval()
        for i, (E,data) in tqdm(enumerate(loader_train, 0), unit="batch", total=len(loader_train)):
            consis_model.model.zero_grad()
            optimizer.zero_grad()

            data = data.to(device = device)
            E = E.to(device = device)

            t = torch.randint(1, consis_model.nsteps, (data.size()[0],), device=device).long()

            batch_loss = compute_consis_loss(data, E, t=t, model = consis_model, ema_model = ema_model, teacher_model = diffu_model, nsteps = flags.sample_steps, sample_algo = flags.sample_algo)
            batch_loss.backward()

            optimizer.step()
            ema_model.update()

            train_loss+=batch_loss.item()

            del data, E, batch_loss

        train_loss = train_loss/len(loader_train)
        training_losses[epoch] = train_loss
        print("loss: "+ str(train_loss))

        val_loss = 0
        consis_model.eval()
        for i, (vE, vdata) in tqdm(enumerate(loader_val, 0), unit="batch", total=len(loader_val)):
            vdata = vdata.to(device=device)
            vE = vE.to(device = device)

            t = torch.randint(1, consis_model.nsteps, (vdata.size()[0],), device=device).long()

            batch_loss = compute_consis_loss(vdata, vE, t=t, model = consis_model, ema_model = ema_model, teacher_model = diffu_model, nsteps = flags.sample_steps, sample_algo = flags.sample_algo)

            val_loss+=batch_loss.item()
            del vdata,vE, batch_loss

        val_loss = val_loss/len(loader_val)
        #scheduler.step(torch.tensor([val_loss]))
        val_losses[epoch] = val_loss
        print("val_loss: "+ str(val_loss), flush = True)

        scheduler.step(torch.tensor([train_loss]))

        if(val_loss < min_validation_loss):
            torch.save(ema_model.state_dict(), os.path.join(checkpoint_folder, 'consis_best_val.pth'))
            min_validation_loss = val_loss

        if(early_stopper.early_stop(val_loss - train_loss)):
            print("Early stopping!")
            break
        print(early_stopper.__dict__)

        # save the model
        consis_model.eval()
        print("SAVING")
        #torch.save(model.state_dict(), checkpoint_path)
        
        #save full training state so can be resumed
        torch.save({
            'epoch': epoch,
            'model_state_dict': ema_model.ema_model.state_dict(),
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
    torch.save(ema_model.state_dict(), os.path.join(checkpoint_folder, 'final.pth'))

    with open(checkpoint_folder + "/training_losses.txt","w") as tfileout:
        tfileout.write("\n".join("{}".format(tl) for tl in training_losses)+"\n")
    with open(checkpoint_folder + "/validation_losses.txt","w") as vfileout:
        vfileout.write("\n".join("{}".format(vl) for vl in val_losses)+"\n")

