import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import argparse
import h5py as h5
import torch.optim as optim
import torch.utils.data as torchdata

from utils import *
from CaloDiffu import *
from models import *


if __name__ == '__main__':
    print("TRAIN DIFFU")

    if(torch.cuda.is_available()): device = torch.device('cuda')
    else: device = torch.device('cpu')
        
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_folder', default='../data/', help='Folder containing data and MC files')
    parser.add_argument('--model', default='Diffu', help='Diffusion model to train. Options are: VPSDE, VESDE and subVPSDE')
    parser.add_argument('-c', '--config', default='configs/test.json', help='Config file with training parameters')
    parser.add_argument('--nevts', type=int,default=-1, help='Number of events to load')
    parser.add_argument('--frac', type=float,default=0.85, help='Fraction of total events used for training')
    parser.add_argument('--load', action='store_true', default=False,help='Load pretrained weights to continue the training')
    parser.add_argument('--seed', type=int, default=1234,help='Pytorch seed')
    parser.add_argument('--reset_training', action='store_true', default=False,help='Retrain')
    flags = parser.parse_args()

    dataset_config = LoadJson(flags.config)

    print("TRAINING OPTIONS")
    print(dataset_config, flush = True)

    torch.manual_seed(flags.seed)

    nholdout  = dataset_config.get('HOLDOUT', 0)

    batch_size = dataset_config['BATCH']
    #train layers for longer
    num_epochs = 2*dataset_config['MAXEPOCH']
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

    for i, dataset in enumerate(dataset_config['FILES']):
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

            nholdout = nholdout if (i == len(dataset_config['FILES']) -1 ) else 0,
            dataset_num  = dataset_num,
            orig_shape = orig_shape,
        )


        if(i ==0): 
            data = data_
            energies = e_
            layers = layers_
        else:
            data = np.concatenate((data, data_))
            energies = np.concatenate((energies, e_))
            layers = np.concatenate((layers, layers_))

        

    dshape = dataset_config['SHAPE_PAD']
    if(layer_norm): layers = np.reshape(layers, (layers.shape[0], -1))
    if(not orig_shape): data = np.reshape(data,dshape)
    else: data = np.reshape(data, (len(data), -1))

    num_data = data.shape[0]
    print("Data Shape " + str(data.shape))
    data_size = data.shape[0]
    print("Pre-processed data mean %.2f std dev %.2f" % (np.mean(data), np.std(data)))
    torch_E_tensor = torch.from_numpy(energies.astype(np.float32))
    torch_layer_tensor =  torch.from_numpy(layers.astype(np.float32))

    del data
    #train_data, val_data = utils.split_data_np(data,flags.frac)

    torch_dataset  = torchdata.TensorDataset(torch_E_tensor, torch_layer_tensor)
    nTrain = int(round(flags.frac * num_data))
    nVal = num_data - nTrain
    train_dataset, val_dataset = torch.utils.data.random_split(torch_dataset, [nTrain, nVal])

    loader_train = torchdata.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    loader_val = torchdata.DataLoader(val_dataset, batch_size = batch_size, shuffle = True)

    del torch_E_tensor, train_dataset, val_dataset
    checkpoint_folder = '../models/{}_{}/'.format(dataset_config['CHECKPOINT_NAME'],flags.model)
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    checkpoint = dict()
    checkpoint_path = os.path.join(checkpoint_folder, "layer_checkpoint.pth")
    if(flags.load and os.path.exists(checkpoint_path)): 
        print("Loading training checkpoint from %s" % checkpoint_path, flush = True)
        checkpoint = torch.load(checkpoint_path, map_location = device)
        print(checkpoint.keys())

    shape = dataset_config['SHAPE_PAD'][1:] if (not orig_shape) else dataset_config['SHAPE_ORIG'][1:]

    cond_size =1
    if(hgcal): cond_size += 2
    layer_model = ResNet(dim_in = dataset_config['SHAPE_PAD'][2] + 1, num_layers = 5, cond_size = cond_size).to(device = device)

    summary_shape = [[1, dataset_config['SHAPE_PAD'][2] +1], [[1,cond_size]], [1]]

    summary(layer_model, summary_shape)

    #sometimes save only weights, sometimes save other info
    if('model_state_dict' in checkpoint.keys()): layer_model.load_state_dict(checkpoint['model_state_dict'])
    elif(len(checkpoint.keys()) > 1): layer_model.load_state_dict(checkpoint)


    model = CaloDiffu(shape, config=dataset_config, layer_model = layer_model, training_obj = training_obj, nsteps = dataset_config['NSTEPS'],).to(device = device)







    os.system('cp {} {}/config.json'.format(flags.config,checkpoint_folder)) # bkp of config file

    early_stopper = EarlyStopper(patience = dataset_config['EARLYSTOP'], mode = 'diff', min_delta = 1e-5)
    if('early_stop_dict' in checkpoint.keys() and not flags.reset_training): early_stopper.__dict__ = checkpoint['early_stop_dict']
    print(early_stopper.__dict__)
    

    criterion = nn.MSELoss().to(device = device)

    optimizer = optim.Adam(model.parameters(), lr = float(dataset_config["LR"]))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer = optimizer, factor = 0.1, patience = 15, verbose = True) 
    if('optimizer_state_dict' in checkpoint.keys() and not flags.reset_training): optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if('scheduler_state_dict' in checkpoint.keys() and not flags.reset_training): scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    training_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    start_epoch = 0
    min_validation_loss = 99999.
    if('train_loss_hist' in checkpoint.keys() and not flags.reset_training): 
        training_losses = checkpoint['train_loss_hist']
        val_losses = checkpoint['val_loss_hist']
        start_epoch = checkpoint['epoch'] + 1
        if(len(training_losses) < num_epochs): 
            training_losses = np.concatenate((training_losses, [0]*(num_epochs - len(training_losses))))
            val_losses = np.concatenate((val_losses, [0]*(num_epochs - len(val_losses))))

    #training loop
    for epoch in range(start_epoch, num_epochs):
        print("Beginning epoch %i" % epoch, flush=True)
        train_loss = 0

        model.train()
        for i, (E,layers) in tqdm(enumerate(loader_train, 0), unit="batch", total=len(loader_train)):
            model.zero_grad()
            optimizer.zero_grad()

            E = E.to(device = device)
            layers = layers.to(device = device)


            t = torch.randint(0, model.nsteps, (layers.size()[0],), device=device).long()
            noise = torch.randn_like(layers)


            batch_loss = model.compute_loss(layers, E, noise = noise, t = t, loss_type = loss_type, layer_loss = True)
            batch_loss.backward()

            optimizer.step()
            train_loss+=batch_loss.item()

            del layers, E, noise, batch_loss

        train_loss = train_loss/len(loader_train)
        training_losses[epoch] = train_loss
        print("loss: "+ str(train_loss))

        val_loss = 0
        model.eval()
        for i, (vE, vlayers) in tqdm(enumerate(loader_val, 0), unit="batch", total=len(loader_val)):
            vE = vE.to(device = device)
            vlayers = vlayers.to(device = device)

            t = torch.randint(0, model.nsteps, (vlayers.size()[0],), device=device).long()
            noise = torch.randn_like(vlayers)

            batch_loss = model.compute_loss(vlayers, vE, noise = noise, t = t, loss_type = loss_type, layer_loss = True)

            val_loss+=batch_loss.item()
            del vlayers ,vE, noise, batch_loss

        val_loss = val_loss/len(loader_val)
        #scheduler.step(torch.tensor([val_loss]))
        val_losses[epoch] = val_loss
        print("val_loss: "+ str(val_loss), flush = True)

        scheduler.step(torch.tensor([train_loss]))

        if(val_loss < min_validation_loss):
            torch.save(layer_model.state_dict(), os.path.join(checkpoint_folder, 'layer_best_val.pth'))
            min_validation_loss = val_loss

        if(early_stopper.early_stop(val_loss - train_loss)):
            print("Early stopping!")
            break

        # save the model
        model.eval()
        print("SAVING")
        #torch.save(model.state_dict(), checkpoint_path)
        
        #save full training state so can be resumed
        torch.save({
            'epoch': epoch,
            'model_state_dict': layer_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss_hist': training_losses,
            'val_loss_hist': val_losses,
            'early_stop_dict': early_stopper.__dict__,
            }, checkpoint_path)

        with open(checkpoint_folder + "/layer_training_losses.txt","w") as tfileout:
            tfileout.write("\n".join("{}".format(tl) for tl in training_losses)+"\n")
        with open(checkpoint_folder + "/layer_validation_losses.txt","w") as vfileout:
            vfileout.write("\n".join("{}".format(vl) for vl in val_losses)+"\n")


    print("Saving to %s" % checkpoint_folder, flush=True)
    torch.save(layer_model.state_dict(), os.path.join(checkpoint_folder, 'layer_final.pth'))

    with open(checkpoint_folder + "/training_losses.txt","w") as tfileout:
        tfileout.write("\n".join("{}".format(tl) for tl in training_losses)+"\n")
    with open(checkpoint_folder + "/validation_losses.txt","w") as vfileout:
        vfileout.write("\n".join("{}".format(vl) for vl in val_losses)+"\n")

