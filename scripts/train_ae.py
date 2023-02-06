import sys
sys.path.append(".local/lib/python{}.{}/site-packages".format(sys.version_info.major,sys.version_info.minor))
import numpy as np
import os, shutil
from magiconfig import ArgumentParser, MagiConfigOptions, ArgumentDefaultsRawHelpFormatter
import argparse
import h5py as h5
from utils import *
import torch.optim as optim
import torch.utils.data as torchdata
from CaloAE import *

if __name__ == '__main__':
    transform_choices = ['none','log']
    activation_choices = {'relu' : nn.ReLU, 'silu': nn.SiLU, 'swish': nn.SiLU}
    # todo: choices for showermap

    if(torch.cuda.is_available()): device = torch.device('cuda')
    else: device = torch.device('cpu')

    parser = ArgumentParser(config_options=MagiConfigOptions(), formatter_class=ArgumentDefaultsRawHelpFormatter)
    parser.add_argument('--name', type=str, required=True, help='name for checkpoint')
    parser.add_argument('--data-folder', default='/wclustre/cms_mlsim/denoise/CaloChallenge/', help='Folder containing data and MC files')
    parser.add_argument('--files', type=str, default=['dataset_2_1.hdf5','dataset_2_2.hdf5'], nargs='+', help='data filenames')
    parser.add_argument('--nevts', type=float,default=-1, help='Number of events to load')
    parser.add_argument('--frac', type=float,default=0.8, help='Fraction of total events used for training')
    parser.add_argument('--workers', type=int, default=1, help='number of workers for data loaders')
    parser.add_argument('--load', action='store_true', default=False,help='Load pretrained weights to continue the training')
    parser.add_argument('--batch', type=int, default=128, help='batch size')
    parser.add_argument('--maxepoch', type=int, default=2000, help='max number of epochs to train')
    parser.add_argument('--earlystop', type=int, default=25, help='early stopper patience value')
    parser.add_argument('--shape-pad', type=int, default=[-1,1,45,16,9], nargs='+', help='shape with padding')
    parser.add_argument('--emin', type=float, default=1., help='minimum energy in dataset')
    parser.add_argument('--emax', type=float, default=1000., help='maximum energy in dataset')
    parser.add_argument('--maxdep', type=float, default=2., help='maximum energy deposited')
    parser.add_argument('--transform', type=str, default='log', choices=transform_choices, help='transform for energy values')
    parser.add_argument('--showermap', type=str, default='logit-norm', help='transform for showers')
    parser.add_argument('--lr', type=float, default=4e-4, help='learning rate')
    parser.add_argument('--act', type=str, default='swish', choices=sorted(list(activation_choices.keys())), help='activation function')
    parser.add_argument('--stride', type=int, default=[3,2,2], nargs=3, help='stride dimension')
    parser.add_argument('--kernel', type=int, default=[3,3,3], nargs=3, help='kernel dimensions')
    parser.add_argument('--layer-size', type=int, default=[32,64,64,32], nargs='+', help='layer sizes')
    parser.add_argument('--dim-red', type=int, default=[0,2,0,2], nargs='+', help='dimensional reduction')
    args = parser.parse_args()

    args.act = activation_choices[args.act]

    batch_size = args.batch
    num_epochs = args.maxepoch

    data = []
    energies = []
    for i, dataset in enumerate(args.files):
        data_,e_ = utils.DataLoader(
            os.path.join(args.data_folder,dataset),
            args.shape_pad,
            emax = args.emax, emin = args.emin,
            nevts = args.nevts,
            max_deposit = args.maxdep, #noise can generate more deposited energy than generated
            logE = args.transform=="log",
            showerMap = args.showermap,
            nholdout = 5000 if (i == len(args.files) -1 ) else 0,
        )

        if(i ==0):
            data = data_
            energies = e_
        else:
            data = np.concatenate((data, data_))
            energies = np.concatenate((energies, e_))

    data = np.reshape(data,args.shape_pad)
    data_size = data.shape[0]
    train_data, val_data = utils.split_data_np(data,args.frac)

    train_data_tensor = torch.from_numpy(train_data)
    #train_dataset = torchdata.TensorDataset(train_data_tensor)
    loader_train = torchdata.DataLoader(train_data_tensor, batch_size = batch_size, shuffle = True, num_workers = args.workers)


    val_data_tensor = torch.from_numpy(val_data)
    #val_dataset = torchdata.TensorDataset(val_data_tensor)
    loader_val = torchdata.DataLoader(val_data_tensor, batch_size = batch_size, num_workers = args.workers)


    del data,train_data_tensor, val_data_tensor

    checkpoint_folder = '../models/{}/'.format(args.name)
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    early_stopper = EarlyStopper(patience = args.earlystop, min_delta = 1e-3)

    shutil.copy("CaloAE.py", checkpoint_folder+"CaloAE.py") # bkp of model def
    parser.write_config(args, checkpoint_folder+"config.py") # bkp of config file

    model = CaloAE(args.shape_pad[1:], batch_size, config=args).to(device=device)
    load_path = checkpoint_folder + "checkpoint.pth"
    if(args.load and os.path.exists(load_path)):
        print("Loading saved weights from %s" % load_path)
        model.load_state_dict(torch.load(load_path, map_location = device))

    criterion = nn.MSELoss().to(device = device)

    optimizer = optim.Adam(model.parameters(), lr = args.lr)
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

        with open(checkpoint_folder + "/training_losses_checkpoint.txt","w") as tfileout:
            tfileout.write("\n".join("{}".format(tl) for tl in training_losses)+"\n")
        with open(checkpoint_folder + "/validation_losses_checkpoint.txt","w") as vfileout:
            vfileout.write("\n".join("{}".format(vl) for vl in val_losses)+"\n")


    print("Saving to %s" % checkpoint_folder)
    torch.save(model.state_dict(), os.path.join(checkpoint_folder, 'final.pth'))

    with open(checkpoint_folder + "/training_losses.txt","w") as tfileout:
        tfileout.write("\n".join("{}".format(tl) for tl in training_losses)+"\n")
    with open(checkpoint_folder + "/validation_losses.txt","w") as vfileout:
        vfileout.write("\n".join("{}".format(vl) for vl in val_losses)+"\n")
