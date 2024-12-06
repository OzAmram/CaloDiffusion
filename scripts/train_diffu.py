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


if __name__ == '__main__':
    print("TRAIN DIFFU")

    if(torch.cuda.is_available()): device = torch.device('cuda')
    else: device = torch.device('cpu')
        
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_folder', default='../data/', help='Folder containing data and MC files')
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
    pre_embed = ('pre-embed' in shower_embed)
    print('pre_embed', pre_embed)
    layer_norm = 'layer' in dataset_config['SHOWERMAP']
    max_cells = dataset_config.get('MAX_CELLS', None)
    shower_scale = dataset_config.get('SHOWERSCALE', 200.)

    train_files = []
    val_files = []


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
        NN_embed.init(norm = pre_embed, dataset_num = dataset_num)

    files = dataset_config['FILES']
    val_file_list = dataset_config.get('VAL_FILES', [])

    for i, dataset in enumerate(files + val_file_list):
        tag = ".npz"
        if(flags.nevts > 0): tag = ".n%i.npz" % flags.nevts
        path_clean = os.path.join(flags.data_folder,dataset + tag)

        if(not os.path.exists(path_clean) or flags.reclean):
            showers,E,layers = DataLoader(
                os.path.join(flags.data_folder,dataset),
                dataset_config['SHAPE_PAD'],
                emax = dataset_config['EMAX'],emin = dataset_config['EMIN'],
                hgcal = hgcal,
                nevts = flags.nevts,
                max_deposit=dataset_config['MAXDEP'], #noise can generate more deposited energy than generated
                logE=dataset_config['logE'],
                showerMap = dataset_config['SHOWERMAP'],
                shower_scale = shower_scale,
                max_cells = max_cells,

                nholdout = nholdout if (i == len(dataset_config['FILES']) -1 ) else 0,
                dataset_num  = dataset_num,
                orig_shape = orig_shape,
                embed = pre_embed,
                NN_embed = NN_embed,
            )

            layers = np.reshape(layers, (layers.shape[0], -1))


            if(orig_shape): showers = np.reshape(showers, dataset_config['SHAPE_ORIG'])
            else : showers = np.reshape(showers, dataset_config['SHAPE_PAD'])

            print("Pre-processed shower mean %.2f std dev %.2f" % (np.mean(showers), np.std(showers)))

            np.savez_compressed(path_clean, E=E, layers=layers, showers=showers, )
            del E, layers, showers

        if dataset in dataset_config['FILES']: train_files.append(path_clean)
        else: val_files.append(path_clean) 


    avg_showers = std_showers = E_bins = None
    if(cold_diffu):
        f_avg_shower = h5.File(dataset_config["AVG_SHOWER_LOC"])
        #Already pre-processed
        avg_showers = torch.from_numpy(f_avg_shower["avg_showers"][()].astype(np.float32)).to(device = device)
        std_showers = torch.from_numpy(f_avg_shower["std_showers"][()].astype(np.float32)).to(device = device)
        E_bins = torch.from_numpy(f_avg_shower["E_bins"][()].astype(np.float32)).to(device = device)

    dshape = dataset_config['SHAPE_PAD']

    dataset_train = Dataset(train_files)
    loader_train  = torchdata.DataLoader(dataset_train, batch_size = batch_size, num_workers=flags.num_workers, pin_memory = True)

    loader_val = None
    if(len(val_files) > 0):
        dataset_val = Dataset(val_files)
        loader_val  = torchdata.DataLoader(dataset_val, batch_size = batch_size, num_workers=flags.num_workers, pin_memory = True)

    checkpoint_folder = '../models/{}_{}/'.format(dataset_config['CHECKPOINT_NAME'],flags.model)
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    checkpoint = dict()
    checkpoint_path = os.path.join(checkpoint_folder, "checkpoint.pth")
    if(flags.load and os.path.exists(checkpoint_path)): 
        print("Loading training checkpoint from %s" % checkpoint_path, flush = True)
        checkpoint = torch.load(checkpoint_path, map_location = device)
        print(checkpoint.keys())

    else:
        if(NN_embed is not None): NN_embed.init()


    if(flags.model == "Diffu"):
        shape = dataset_config['SHAPE_FINAL'][1:] 
        model = CaloDiffu(shape, config=dataset_config, training_obj = training_obj, NN_embed = NN_embed, nsteps = dataset_config['NSTEPS'],
                cold_diffu = cold_diffu, avg_showers = avg_showers, std_showers = std_showers, E_bins = E_bins ).to(device = device)


        #sometimes save only weights, sometimes save other info
        if('model_state_dict' in checkpoint.keys()): model.load_state_dict(checkpoint['model_state_dict'], strict = False)
        elif(len(checkpoint.keys()) > 1): model.load_state_dict(checkpoint, strict = False)

    else:
        print("Model %s not supported!" % flags.model)
        exit(1)


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
    if(loader_val is not None):
        val_rnd = torch.randn( (len(loader_val),batch_size,), device=device)

    #training loop
    for epoch in range(start_epoch, num_epochs):
        print("Beginning epoch %i" % epoch, flush=True)
        train_loss = 0

        model.train()
        for i, (E,layers,data) in tqdm(enumerate(loader_train, 0), unit="batch", total=len(loader_train)):
            model.zero_grad()
            optimizer.zero_grad()


            data = data.to(device = device)
            E = E.to(device = device)
            layers = layers.to(device = device)

            noise = torch.randn_like(data)

            if(cold_diffu): #cold diffusion interpolates from avg showers instead of pure noise
                noise = model.gen_cold_image(E, cold_noise_scale, noise)
                

            batch_loss = model.compute_loss(data, E, noise = noise, layers = layers,  loss_type = loss_type )
            batch_loss.backward()

            optimizer.step()
            train_loss+=batch_loss.item()

            del data, E, layers, noise, batch_loss


        train_loss = train_loss/len(loader_train)
        training_losses[epoch] = train_loss
        print("loss: "+ str(train_loss))

        val_loss = 0
        if(loader_val is not None):
            model.eval()
            for i, (vE, vlayers, vdata) in tqdm(enumerate(loader_val, 0), unit="batch", total=len(loader_val)):

                #skip last batch to avoid shape issues
                if(vE.shape[0] != batch_size): continue

                vdata = vdata.to(device=device)
                vE = vE.to(device = device)
                vlayers = vlayers.to(device = device)

                rnd_normal = val_rnd[i].to(device = device)

                noise = torch.randn_like(vdata)
                if(cold_diffu): noise = model.gen_cold_image(vE, cold_noise_scale, noise)

                batch_loss = model.compute_loss(vdata, vE, noise = noise, layers = vlayers, rnd_normal = rnd_normal, loss_type = loss_type  )

                val_loss+=batch_loss.item()
                del vdata,vE, vlayers, noise, batch_loss

            val_loss = val_loss/len(loader_val)
            #scheduler.step(torch.tensor([val_loss]))
            val_losses[epoch] = val_loss
            print("val_loss: "+ str(val_loss), flush = True)

            scheduler.step(torch.tensor([train_loss]))

            if(val_loss < min_validation_loss):
                torch.save(model.state_dict(), os.path.join(checkpoint_folder, 'best_val.pth'))
                min_validation_loss = val_loss

            #if(early_stopper.early_stop(val_loss)):
                #print("Early stopping!")
                #break

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
    torch.save(model.state_dict(), os.path.join(checkpoint_folder, 'final.pth'))

    with open(checkpoint_folder + "/training_losses.txt","w") as tfileout:
        tfileout.write("\n".join("{}".format(tl) for tl in training_losses)+"\n")
    with open(checkpoint_folder + "/validation_losses.txt","w") as vfileout:
        vfileout.write("\n".join("{}".format(vl) for vl in val_losses)+"\n")

