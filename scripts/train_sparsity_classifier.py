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
from SparsityClassifier import SparsityClassifier
from models import *


if __name__ == '__main__':
    print("TRAIN DIFFU")

    if(torch.cuda.is_available()): device = torch.device('cuda')
    else: device = torch.device('cpu')
        
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_folder', default='../data/', help='Folder containing data and MC files')
    parser.add_argument('-g', '--generated', default='', help='File containing generated showers to correct')
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

    train_files = []
    val_files = []

    if(len(flags.generated) <1): 
        print("Need to supply generated showers to correct!")
        exit(1)


    fname_gen = flags.generated + ".npz"
    if(not os.path.exists(fname_gen) or flags.reclean):
        showers_gen,E_gen,layers_gen = DataLoader(
            flags.generated,
            dataset_config['SHAPE_PAD'],
            emax = dataset_config['EMAX'],emin = dataset_config['EMIN'],
            hgcal = hgcal,
            nevts = -1,
            max_deposit=dataset_config['MAXDEP'], #noise can generate more deposited energy than generated
            logE=dataset_config['logE'],
            showerMap = dataset_config['SHOWERMAP'],
            max_cells = max_cells,

            nholdout =  0,
            dataset_num  = dataset_num,
            orig_shape = orig_shape,
        )
        nevts = showers_gen.shape[0]

        layers = np.reshape(layers_gen, (layers_gen.shape[0], -1))
        if(orig_shape): showers_gen = np.reshape(showers_gen, dataset_config['SHAPE_ORIG'])
        else : showers_gen = np.reshape(showers_gen, dataset_config['SHAPE_PAD'])

        np.savez_compressed(fname_gen, E=E_gen, layers = layers_gen, showers=showers_gen )

    else:
        f = np.load(fname_gen)
        nevts = f['E'].shape[0]



    for i, dataset in enumerate(dataset_config['FILES'] + dataset_config['VAL_FILES']):
        tag = ".n%i.npz" % nevts
        path_clean = os.path.join(flags.data_folder,dataset + tag)

        if(not os.path.exists(path_clean) or flags.reclean):
        #if(True):
            showers,E,layers = DataLoader(
                os.path.join(flags.data_folder,dataset),
                dataset_config['SHAPE_PAD'],
                emax = dataset_config['EMAX'],emin = dataset_config['EMIN'],
                hgcal = hgcal,
                nevts = nevts,
                max_deposit=dataset_config['MAXDEP'], #noise can generate more deposited energy than generated
                logE=dataset_config['logE'],
                showerMap = dataset_config['SHOWERMAP'],
                max_cells = max_cells,

                nholdout = nholdout if (i == len(dataset_config['FILES']) -1 ) else 0,
                dataset_num  = dataset_num,
                orig_shape = orig_shape,
            )

            layers = np.reshape(layers, (layers.shape[0], -1))
            if(orig_shape): showers = np.reshape(showers, dataset_config['SHAPE_ORIG'])
            else : showers = np.reshape(showers, dataset_config['SHAPE_PAD'])

            print("Pre-processed shower mean %.2f std dev %.2f" % (np.mean(showers), np.std(showers)))

            np.savez_compressed(path_clean, E=E, layers=layers, showers=showers, )

        train_files.append(path_clean)
        if(nevts > 0): break

        del E, layers, showers


    print(train_files)
    dataset_train = Dataset(train_files)
    dataset_train_gen = Dataset([fname_gen])

    loader_train  = torchdata.DataLoader(dataset_train, batch_size = batch_size, num_workers=flags.num_workers, pin_memory = True)
    loader_train_gen  = torchdata.DataLoader(dataset_train_gen, batch_size = batch_size, num_workers=flags.num_workers, pin_memory = True)
        
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
        NN_embed.init()


    dshape = dataset_config['SHAPE_PAD']

    checkpoint_folder = '../models/{}_{}/'.format(dataset_config['CHECKPOINT_NAME'],flags.model)
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    checkpoint = dict()
    checkpoint_path = os.path.join(checkpoint_folder, "checkpoint_spars_class.pth")
    if(flags.load and os.path.exists(checkpoint_path)): 
        print("Loading training checkpoint from %s" % checkpoint_path, flush = True)
        checkpoint = torch.load(checkpoint_path, map_location = device)
        print(checkpoint.keys())

    else:
        if(NN_embed is not None): NN_embed.init()


    shape = dataset_config['SHAPE_FINAL'][1:] 
    model = SparsityClassifier(shape, config=dataset_config, NN_embed = NN_embed ).to(device = device)


    #sometimes save only weights, sometimes save other info
    if('model_state_dict' in checkpoint.keys()): model.load_state_dict(checkpoint['model_state_dict'], strict = False)
    elif(len(checkpoint.keys()) > 1): model.load_state_dict(checkpoint, strict = False)


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

    #Freeze validation noise levels so stable performance
    #val_rnd = torch.randn( (len(loader_val),batch_size,), device=device)


    #training loop
    for epoch in range(start_epoch, num_epochs):
        print("Beginning epoch %i" % epoch, flush=True)
        dataloader_geant = iter(loader_train)

        train_loss = 0

        model.train()
        for i, (E,layers,data_gen) in tqdm(enumerate(loader_train_gen, 0), unit="batch", total=len(loader_train_gen)):

            try:
                (_,_,data_geant) = next(dataloader_geant)
            except StopIteration:
                print("Mismatched dataset sizes?")
                dataloader_iterator = iter(loader_train)
                (_,_, data_geant) = next(dataloader_geant)


            model.zero_grad()
            optimizer.zero_grad()


            data_geant = data_geant.to(device = device)
            data_gen = data_gen.to(device = device)
            E = E.to(device = device)
            layers = layers.to(device = device)


            batch_loss = model.compute_loss(data_gen, data_geant, E, layers = layers)
            batch_loss.backward()

            optimizer.step()
            train_loss+=batch_loss.item()

            del E, layers, data_gen, data_geant


        train_loss = train_loss / len(loader_train_gen)
        training_losses[epoch] = train_loss
        print("loss: "+ str(train_loss))

        # save the model
        model.eval()
        print("SAVING")
        
        #save full training state so can be resumed
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
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
    torch.save(model.state_dict(), os.path.join(checkpoint_folder, 'final_spars_class.pth'))

    with open(checkpoint_folder + "/training_losses.txt","w") as tfileout:
        tfileout.write("\n".join("{}".format(tl) for tl in training_losses)+"\n")
    with open(checkpoint_folder + "/validation_losses.txt","w") as vfileout:
        vfileout.write("\n".join("{}".format(vl) for vl in val_losses)+"\n")

