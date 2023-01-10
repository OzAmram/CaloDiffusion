import numpy as np
import os
import argparse
import h5py as h5
import torch.optim as optim
import torch.utils.data as torchdata

from utils import *
from CaloAE import *
from CaloDiffu import *
from models import *


if __name__ == '__main__':
    print("TRAIN DIFFU")

    if(torch.cuda.is_available()): device = torch.device('cuda')
    else: device = torch.device('cpu')
        
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_folder', default='/wclustre/cms_mlsim/denoise/CaloChallenge/', help='Folder containing data and MC files')
    parser.add_argument('--model', default='Diffu', help='Diffusion model to train. Options are: VPSDE, VESDE and subVPSDE')
    parser.add_argument('-c', '--config', default='configs/test.json', help='Config file with training parameters')
    parser.add_argument('--nevts', type=float,default=-1, help='Number of events to load')
    parser.add_argument('--frac', type=float,default=0.8, help='Fraction of total events used for training')
    parser.add_argument('--load', action='store_true', default=False,help='Load pretrained weights to continue the training')
    flags = parser.parse_args()

    dataset_config = utils.LoadJson(flags.config)
    data = []
    energies = []

    print("TRAINING OPTIONS")
    print(dataset_config, flush = True)

    batch_size = dataset_config['BATCH']
    num_epochs = dataset_config['MAXEPOCH']
    early_stop = dataset_config['EARLYSTOP']


    for i, dataset in enumerate(dataset_config['FILES']):
        data_,e_ = utils.DataLoader(
            os.path.join(flags.data_folder,dataset),
            dataset_config['SHAPE_PAD'],
            emax = dataset_config['EMAX'],emin = dataset_config['EMIN'],
            nevts = flags.nevts,
            max_deposit=dataset_config['MAXDEP'], #noise can generate more deposited energy than generated
            logE=dataset_config['logE'],
            showerMap = dataset_config['SHOWERMAP'],
            nholdout = 5000 if (i == len(dataset_config['FILES']) -1 ) else 0,
        )


        if(i ==0): 
            data = data_
            energies = e_
        else:
            data = np.concatenate((data, data_))
            energies = np.concatenate((energies, e_))
        

    energies = np.reshape(energies,(-1))    
    data = np.reshape(data,dataset_config['SHAPE_PAD'])
    print(data.shape)
    data_size = data.shape[0]
    print("Pre-processed shower mean %.2f std dev %.2f" % (np.mean(data), np.std(data)))
    torch_data_tensor = torch.from_numpy(data)
    torch_E_tensor = torch.from_numpy(energies)
    #train_data, val_data = utils.split_data_np(data,flags.frac)

    torch_dataset  = torchdata.TensorDataset(torch_E_tensor, torch_data_tensor)
    nTrain = int(round(flags.frac * data.shape[0]))
    nVal = data.shape[0] - nTrain
    train_dataset, val_dataset = torch.utils.data.random_split(torch_dataset, [nTrain, nVal])

    loader_train = torchdata.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    loader_val = torchdata.DataLoader(val_dataset, batch_size = batch_size, shuffle = True)


    del data,torch_data_tensor, torch_E_tensor, train_dataset, val_dataset
    checkpoint_folder = '../models/{}_{}/'.format(dataset_config['CHECKPOINT_NAME'],flags.model)
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    if(flags.model == "Diffu"):
        model = CaloDiffu(dataset_config['SHAPE_PAD'][1:], batch_size, config=dataset_config).to(device = device)
        if(flags.load and os.path.exists(checkpoint_folder + "checkpoint.pth")): 
            load_path = checkpoint_folder + "checkpoint.pth"
            print("Loading saved weights from %s" % load_path, flush = True)
            model.load_state_dict(torch.load(load_path, map_location = device))

    elif(flags.model == "LatentDiffu"):
        AE = CaloAE(dataset_config['SHAPE_PAD'][1:], batch_size, config=dataset_config).to(device = device)
        AE.load_state_dict(torch.load(dataset_config['AE'], map_location = device))

        print("ENC shape", AE.encoded_shape)
        model = CaloDiffu(AE.encoded_shape,energies.shape[1],batch_size, config=dataset_config).to(device=device)

        #encode data to latent space
        data = AE.encoder_model.predict(data, batch_size = 256)



    else:
        print("Model %s not supported!" % flags.model)
        exit(1)


    os.system('cp CaloDiffu.py {}'.format(checkpoint_folder)) # bkp of model def
    os.system('cp models.py {}'.format(checkpoint_folder)) # bkp of model def
    os.system('cp {} {}'.format(flags.config,checkpoint_folder)) # bkp of config file

    early_stopper = EarlyStopper(patience = dataset_config['EARLYSTOP'], min_delta = 1e-3)
    

    criterion = nn.MSELoss().to(device = device)

    optimizer = optim.Adam(model.parameters(), lr = float(dataset_config["LR"]))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer = optimizer, factor = 0.1, patience = 10, verbose = True) 
    step = 0
    training_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)

    #training loop
    for epoch in range(num_epochs):
        print("Beginning epoch %i" % epoch, flush=True)
        train_loss = 0

        model.train()
        for i, (E,data) in tqdm(enumerate(loader_train, 0), unit="batch", total=len(loader_train)):
            model.zero_grad()
            optimizer.zero_grad()

            data = data.to(device = device)
            E = E.to(device = device)
            t = torch.randint(0, model.nsteps, (data.size()[0],), device=device).long()
            noise = torch.randn_like(data)

            batch_loss = model.compute_loss(data, E, t, noise)
            batch_loss.backward()

            optimizer.step()
            train_loss+=batch_loss.item()

            del data, E, t, noise, batch_loss

        train_loss = train_loss/len(loader_train)
        training_losses[epoch] = train_loss
        print("loss: "+ str(train_loss))

        val_loss = 0
        model.eval()
        for i, (vE, vdata) in tqdm(enumerate(loader_val, 0), unit="batch", total=len(loader_val)):
            vdata = vdata.to(device=device)
            vE = vE.to(device = device)

            t = torch.randint(0, model.nsteps, (vdata.size()[0],), device=device).long()
            noise = torch.randn_like(vdata)
            batch_loss = model.compute_loss(vdata, vE, t, noise)

            val_loss+=batch_loss.item()
            del vdata,vE, t, noise, batch_loss

        val_loss = val_loss/len(loader_val)
        scheduler.step(torch.tensor([val_loss]))
        val_losses[epoch] = val_loss
        print("val_loss: "+ str(val_loss), flush = True)
        if(early_stopper.early_stop(val_loss)):
            print("Early stopping!")
            break

        # save the model
        model.eval()
        torch.save(model.state_dict(), os.path.join(checkpoint_folder, 'checkpoint.pth'))
        with open(checkpoint_folder + "/training_losses.txt","w") as tfileout:
            tfileout.write("\n".join("{}".format(tl) for tl in training_losses)+"\n")
        with open(checkpoint_folder + "/validation_losses.txt","w") as vfileout:
            vfileout.write("\n".join("{}".format(vl) for vl in val_losses)+"\n")


    print("Saving to %s" % checkpoint_folder, flush=True)
    torch.save(model.state_dict(), os.path.join(checkpoint_folder, 'final.pth'))

    with open(checkpoint_folder + "/training_losses.txt","w") as tfileout:
        tfileout.write("\n".join("{}".format(tl) for tl in training_losses)+"\n")
    with open(checkpoint_folder + "/validation_losses.txt","w") as vfileout:
        vfileout.write("\n".join("{}".format(vl) for vl in val_losses)+"\n")

