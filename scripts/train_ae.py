import numpy as np
import os
import argparse
import h5py as h5
from utils import *
import torch.optim as optim
import torch.utils.data as torchdata
from CaloAE import *

if __name__ == '__main__':
    if(torch.cuda.is_available()): device = torch.device('cuda')
    else: device = torch.device('cpu')
        
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--data_folder', default='/wclustre/cms_mlsim/denoise/CaloChallenge/', help='Folder containing data and MC files')
    parser.add_argument('--model', default='AE', help='AE')
    parser.add_argument('--config', default='configs/config_dataset2_ae.json', help='Config file with training parameters')
    parser.add_argument('--nevts', type=float,default=-1, help='Number of events to load')
    parser.add_argument('--frac', type=float,default=0.8, help='Fraction of total events used for training')
    parser.add_argument('--workers', type=int, default=1, help='number of workers for data loaders')
    parser.add_argument('--load', action='store_true', default=False,help='Load pretrained weights to continue the training')
    flags = parser.parse_args()

    dataset_config = utils.LoadJson(flags.config)
    batch_size = dataset_config['BATCH']
    num_epochs = dataset_config['MAXEPOCH']
    early_stop = dataset_config['EARLYSTOP']


    data = []
    energies = []
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
        

    data = np.reshape(data,dataset_config['SHAPE_PAD'])
    data_size = data.shape[0]
    train_data, val_data = utils.split_data_np(data,flags.frac)

    train_data_tensor = torch.from_numpy(train_data)
    #train_dataset = torchdata.TensorDataset(train_data_tensor)
    loader_train = torchdata.DataLoader(train_data_tensor, batch_size = batch_size, shuffle = True, num_workers = flags.workers)


    val_data_tensor = torch.from_numpy(val_data)
    #val_dataset = torchdata.TensorDataset(val_data_tensor)
    loader_val = torchdata.DataLoader(val_data_tensor, batch_size = batch_size, num_workers = flags.workers)


    del data,train_data_tensor, val_data_tensor
    

    checkpoint_folder = '../models/{}_{}/'.format(dataset_config['CHECKPOINT_NAME'],flags.model)
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    early_stopper = EarlyStopper(patience = dataset_config['EARLYSTOP'], min_delta = 1e-3)

    os.system('cp CaloAE.py {}'.format(checkpoint_folder)) # bkp of model def
    os.system('cp {} {}'.format(flags.config,checkpoint_folder)) # bkp of config file
    
    model = CaloAE(dataset_config['SHAPE_PAD'][1:], batch_size, config=dataset_config).to(device=device)
    if(flags.load and os.path.exists(checkpoint_folder + "checkpoint.pth")): 
        load_path = checkpoint_folder + "checkpoint.pth"
        print("Loading saved weights from %s" % load_path)
        model.load_state_dict(torch.load(load_path, map_location = device))

    criterion = nn.MSELoss().to(device = device)

    optimizer = optim.Adam(model.parameters(), lr = float(dataset_config["LR"]))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer = optimizer, factor = 0.1, patience = 10, verbose = True) 
    step = 0
    training_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)

    
    #training loop
    for epoch in range(num_epochs):
        print("Beginning epoch %i" % epoch)
        train_loss = 0

        model.train()
        for i, data in tqdm(enumerate(loader_train, 0), unit="batch", total=len(loader_train)):
            model.zero_grad()
            optimizer.zero_grad()
            data = data.to(device = device)
            output = model(data).to(device=device)
            batch_loss = criterion(output, data).to(device=device)
            batch_loss.backward()
            optimizer.step()
            train_loss+=batch_loss.item()
            del data
            del output
            del batch_loss
        train_loss = train_loss/len(loader_train)
        training_losses[epoch] = train_loss
        print("loss: "+ str(train_loss))

        val_loss = 0
        model.eval()
        for i, vdata in tqdm(enumerate(loader_val, 0), unit="batch", total=len(loader_val)):
            vdata = vdata.to(device=device)
            output = model(vdata).to(device=device)
            output_loss = criterion(vdata, output).to(device=device)
            val_loss+=output_loss.item()
            del vdata
            del output
            del output_loss
        val_loss = val_loss/len(loader_val)
        scheduler.step(torch.tensor([val_loss]))
        val_losses[epoch] = val_loss
        print("val_loss: "+ str(val_loss))
        if(early_stopper.early_stop(val_loss)):
            print("Early stopping!")
            break

        # save the model
        model.eval()
        torch.save(model.state_dict(), os.path.join(checkpoint_folder, 'checkpoint.pth'))


    print("Saving to %s" % checkpoint_folder)
    torch.save(model.state_dict(), os.path.join(checkpoint_folder, 'final.pth'))

    with open(checkpoint_folder + "/training_losses.txt","w") as tfileout:
        tfileout.write("\n".join("{}".format(tl) for tl in training_losses)+"\n")
    with open(checkpoint_folder + "/validation_losses.txt","w") as vfileout:
        vfileout.write("\n".join("{}".format(vl) for vl in val_losses)+"\n")
