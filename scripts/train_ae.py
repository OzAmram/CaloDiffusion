import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping, ModelCheckpoint
#import horovod.tensorflow.keras as hvd
import argparse
import h5py as h5
import utils
from CaloAE import CaloAE

if __name__ == '__main__':
    #hvd.init()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    #if gpus:
        #tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


        
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_folder', default='/pscratch/sd/v/vmikuni/FCC', help='Folder containing data and MC files')
    parser.add_argument('--model', default='AE', help='AE')
    parser.add_argument('--config', default='config_dataset2.json', help='Config file with training parameters')
    parser.add_argument('--nevts', type=float,default=-1, help='Number of events to load')
    parser.add_argument('--frac', type=float,default=0.8, help='Fraction of total events used for training')
    parser.add_argument('--load', action='store_true', default=False,help='Load pretrained weights to continue the training')
    flags = parser.parse_args()

    dataset_config = utils.LoadJson(flags.config)
    data = []
    energies = []
    for dataset in dataset_config['FILES']:
        data_,e_ = utils.DataLoader(
            os.path.join(flags.data_folder,dataset),
            dataset_config['SHAPE_PAD'],flags.nevts,
            emax = dataset_config['EMAX'],emin = dataset_config['EMIN'],
            max_deposit=dataset_config['MAXDEP'], #noise can generate more deposited energy than generated
            logE=dataset_config['logE'],
            norm_data=dataset_config['NORMED'],
            showerMap = dataset_config['SHOWERMAP'],
        )
        
        data.append(data_)
        energies.append(e_)
        

    data = np.reshape(data,dataset_config['SHAPE_PAD'])
    data_size = data.shape[0]
    dataset = tf.data.Dataset.from_tensor_slices((data, data))
    train_data, test_data = utils.split_data(dataset,data_size,flags.frac)
    del dataset, data
    
    BATCH_SIZE = dataset_config['BATCH']
    LR = float(dataset_config['LR'])
    NUM_EPOCHS = dataset_config['MAXEPOCH']
    EARLY_STOP = dataset_config['EARLYSTOP']


    checkpoint_folder = '../models/{}_{}/'.format(dataset_config['CHECKPOINT_NAME'],flags.model)
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)
    
    callbacks = [
        #hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        #hvd.callbacks.MetricAverageCallback(),            
        #ReduceLROnPlateau(patience=100, factor=0.5,
        #                  min_lr=1e-8,verbose=hvd.rank()==0),
        EarlyStopping(patience=EARLY_STOP,restore_best_weights=True),
        ModelCheckpoint(filepath = checkpoint_folder + "checkpoint", save_weights_only = True, save_best_only = True, mode = "min", monitor = "val_loss", save_format = "tf"),
        # hvd.callbacks.LearningRateWarmupCallback(
        #     initial_lr=LR*hvd.size(), warmup_epochs=5,
        #     verbose=hvd.rank()==0)
    ]

    model = CaloAE(dataset_config['SHAPE_PAD'][1:], BATCH_SIZE, config=dataset_config).model
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    if flags.load:
        checkpoint_folder = '../checkpoints_{}_{}'.format(dataset_config['CHECKPOINT_NAME'],flags.model)
        model.load_weights('{}/{}'.format(checkpoint_folder,'checkpoint')).expect_partial()


    
    history = model.fit(
        train_data.batch(BATCH_SIZE),
        epochs=NUM_EPOCHS,
        steps_per_epoch=int(data_size*flags.frac/BATCH_SIZE),
        validation_data=test_data.batch(BATCH_SIZE),
        validation_steps=int(data_size*(1-flags.frac)/BATCH_SIZE),
        verbose=1,
        callbacks=callbacks
    )


    print("Saving to %s" % checkpoint_folder)
    os.system('cp CaloAE.py {}'.format(checkpoint_folder)) # bkp of model def
    os.system('cp {} {}'.format(flags.config,checkpoint_folder)) # bkp of config file
    model.save_weights('{}/{}'.format(checkpoint_folder,'final'),save_format='tf')
