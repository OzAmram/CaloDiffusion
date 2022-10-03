import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping
import horovod.tensorflow.keras as hvd
import argparse
import h5py as h5
import utils
from CaloScore import CaloScore


if __name__ == '__main__':
    hvd.init()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


        
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_folder', default='/pscratch/sd/v/vmikuni/FCC', help='Folder containing data and MC files')
    parser.add_argument('--model', default='VPSDE', help='Diffusion model to train. Options are: VPSDE, VESDE and subVPSDE')
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
            norm_data=dataset_config['NORMED']
        )
        
        data.append(data_)
        energies.append(e_)
        

    data = np.reshape(data,dataset_config['SHAPE_PAD'])
    energies = np.reshape(energies,(-1,1))    
    # print(np.max(data),np.min(data),np.max(energies),np.min(energies))
    # input()
    data_size = data.shape[0]
    tf_data = tf.data.Dataset.from_tensor_slices(data)
    tf_energies = tf.data.Dataset.from_tensor_slices(energies)
    dataset =tf.data.Dataset.zip((tf_data, tf_energies))    
    train_data, test_data = utils.split_data(dataset,data_size,flags.frac)
    del dataset, data, tf_data,tf_energies
    
    BATCH_SIZE = dataset_config['BATCH']
    LR = float(dataset_config['LR'])
    NUM_EPOCHS = dataset_config['MAXEPOCH']
    EARLY_STOP = dataset_config['EARLYSTOP']
    
    callbacks = [
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        hvd.callbacks.MetricAverageCallback(),            
        ReduceLROnPlateau(patience=100, factor=0.5,
                          min_lr=1e-8,verbose=hvd.rank()==0),
        EarlyStopping(patience=EARLY_STOP,restore_best_weights=True),
        # hvd.callbacks.LearningRateWarmupCallback(
        #     initial_lr=LR*hvd.size(), warmup_epochs=5,
        #     verbose=hvd.rank()==0)
    ]

    model = CaloScore(dataset_config['SHAPE_PAD'][1:],energies.shape[1],BATCH_SIZE,sde_type=flags.model,config=dataset_config)
    if flags.load:
        checkpoint_folder = '../checkpoints_{}_{}'.format(dataset_config['CHECKPOINT_NAME'],flags.model)
        model.load_weights('{}/{}'.format(checkpoint_folder,'checkpoint')).expect_partial()

    opt = keras.optimizers.Adam(learning_rate=LR)
    opt = hvd.DistributedOptimizer(
        opt, average_aggregated_gradients=True)

    model.compile(optimizer=opt,experimental_run_tf_function=False)

    
    history = model.fit(
        train_data.batch(BATCH_SIZE),
        epochs=NUM_EPOCHS,
        steps_per_epoch=int(data_size*flags.frac/BATCH_SIZE),
        validation_data=test_data.batch(BATCH_SIZE),
        validation_steps=int(data_size*(1-flags.frac)/BATCH_SIZE),
        verbose=1 if hvd.rank()==0 else 0,
        callbacks=callbacks
    )


    if hvd.rank()==0:
        checkpoint_folder = '../checkpoints_{}_{}'.format(dataset_config['CHECKPOINT_NAME'],flags.model)
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)
        os.system('cp CaloScore.py {}'.format(checkpoint_folder)) # bkp of model def
        os.system('cp {} {}'.format(flags.config,checkpoint_folder)) # bkp of config file
        model.save_weights('{}/{}'.format(checkpoint_folder,'checkpoint'),save_format='tf')
