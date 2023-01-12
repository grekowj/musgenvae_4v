'''Created on 10 gru 2021

@author: jacek
'''
import time
from datetime import timedelta
import numpy as np
import os
from numpy import load
import tensorflow as tf
from tensorflow import keras
from config import MODEL_CONFIG, DATA_CONFIG, TRAIN_CONFIG, EXP_CONFIG
from utils.metrics import Metrics
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model, to_categorical
# import keras_tuner as kt
from sklearn.model_selection import train_test_split

from model.models import VAE, CVAE

import random
tf.random.set_seed(33)
os.environ['PYTHONHASHSEED'] = str(33)
np.random.seed(33)
random.seed(33)


def jg_show_loss(loss, reconstruction_loss, kl_loss, name): 
    epoki = range(len(loss))

    plt.figure()

    plt.plot(epoki, loss, 'ro', label='total_loss')
    plt.plot(epoki, reconstruction_loss, 'g', label='reconstruction_loss')
    plt.plot(epoki, kl_loss, 'b', label='kl_loss')

    plt.title('Strata trenowania')
    plt.legend()

    epoch = len(loss)
    filename1 = '%s_Loss_%04d.png' % (name, epoch)
    path = os.path.join(MODEL_CONFIG['log_dir'], filename1)
    
    plt.savefig(path)

    # plt.show()


def load_data():
    """Load and return the training data."""
    print('[*] Loading data...')    
    dict_data = np.load(DATA_CONFIG['training_data'])
    
    # dict_data = load('../training_data/my_piano_rep_358.npz')
    train_X = dict_data['arr_0']
    print("train_X.shape", train_X.shape)
    train_X = np.reshape(train_X, (-1, MODEL_CONFIG['num_timestep'], MODEL_CONFIG['num_pitch'], 1))
    print("train_X.shape", train_X.shape)
    print('Training set size:', len(train_X))
    return train_X


def load_data_train_test():
    """Ĺ�adowanie danych z podziaĹ‚em train / test"""
    print('[*] Loading data...')    
    dict_data = np.load(DATA_CONFIG['training_data'])
    
    # dict_data = load('../training_data/my_piano_rep_358.npz')
    train_X = dict_data['arr_0']
    print("train_X.shape", train_X.shape)
    train_X = np.reshape(train_X, (-1, MODEL_CONFIG['num_timestep'], MODEL_CONFIG['num_pitch'], 1))
    
    train_X, test_X = train_test_split(train_X, test_size=0.10, random_state=42)
#     X_train, X_test, Y_train, Y_test  = train_test_split(my_tensor_onehot, my_tensor_y, 
#                                                      test_size=0.20, random_state=42, 
#                                                      stratify=my_tensor_y,
# #                                                      shuffle= False                                                     
#                                                     )
    print("train_X.shape", train_X.shape)
    print("test_X.shape", test_X.shape)
    
    return train_X, test_X


def load_data_emo():
    """Ładowanie danych z podziaĹ‚em oznaczonych emo"""
    print('[*] Loading data...')    
    dict_data = np.load(DATA_CONFIG['training_data'])    
    train_X = dict_data['arr_0']
    train_y = dict_data['arr_1']
    
    print("train_X.shape", train_X.shape)
    train_X = np.reshape(train_X, (-1, MODEL_CONFIG['num_timestep'], MODEL_CONFIG['num_pitch'], 1))
    
    print("train_X.shape", train_X.shape)
    print("train_y.shape", train_y.shape)
    train_X, train_y = shuffle_data(train_X, train_y)
       
    return train_X, train_y


def shuffle_data(train_X, train_y):
    ''' miesza dane train_X razem z etykietami train_y'''
    from sklearn.utils import shuffle
    print ("train_y: \n", train_y)
    train_X, train_y = shuffle(train_X, train_y, random_state=42)
    print ("train_y: \n", train_y)
    
    unique_elements, counts_elements = np.unique(train_y, return_counts=True)
    print("Frequency of unique values of the said array:")
    print(np.asarray((unique_elements, counts_elements)))

    return train_X, train_y

# funkcja zamienia dane liczbowe na one_hot
def labels_to_categorical(train_y):
    # labels
    # print(train_y)
    # one-hot encoding
    train_y_onehot = to_categorical(train_y) 
   
    print("train_y_onehot.shape: ", train_y_onehot.shape)
    print("train_y_onehot[0]: ", train_y_onehot[0])
    return train_y_onehot


#*********** Gółwna pęla trenowania Start
def main_train():
    ''' Główna pętla trenowania'''
    print('Start')
    print('{:-^80}'.format(' Start '))
    MODEL_CONFIG['run_main_program'] = True
    
    # train_X = load_data()    
    train_X, train_y = load_data_emo()  
    train_y_onehot = labels_to_categorical(train_y)  
    # train_X, test_X = load_data_train_test()
    
    loss_all = []
    reconstruction_loss_all = []
    kl_loss_all = []
    
    # vae = VAE(MODEL_CONFIG)
    vae = CVAE(MODEL_CONFIG)
    opt = keras.optimizers.Adam(1e-4)
    vae.compile(optimizer=opt)
    # vae.compile(optimizer=opt, metrics=['accuracy'])
    
    print('{:=^80}'.format(' Training Start '))
    print('{:=^80}'.format(' Training Start  epochs={}, batch_size={} '.format (TRAIN_CONFIG['num_epoch'],
                                                              MODEL_CONFIG['batch_size'])))
     
    # metrics = Metrics(MODEL_CONFIG)
    
    training_start_time = time.time() 
    # Stworzenie na nowo plików Latent_space.log eval_samples.log 
    
    filepath = vae.nazwa + '_' + 'Latent_space.log'
    filepath = os.path.join(MODEL_CONFIG['log_dir'], filepath)
    log_latent_space = open(filepath, 'w') 
    log_latent_space.close()
    
    filepath = vae.nazwa + '_' + 'eval_samples.log'
    filepath = os.path.join(MODEL_CONFIG['eval_dir'], filepath)
    log_eval_samples = open(filepath, 'w') 
    log_eval_samples.close()
    
    filepath = vae.nazwa + '_' + 'epoch.log'
    log_epoch = open(os.path.join(MODEL_CONFIG['log_dir'], filepath), 'w')
    vae.save_summary()
    vae.save_plot_model()
    
    log_epoch.write('# epoch,  loss, reconstruction_loss, kl_loss \n')
    
    for epoch in range(TRAIN_CONFIG['num_epoch']):
        print('{:-^80}'.format(' Epoch {} Start '.format(epoch + 1)))
        
        # history = vae.fit(train_X, epochs=1, batch_size=MODEL_CONFIG['batch_size']) #VAE      
        # history = vae.fit(x=[train_X[0:50], train_y_onehot[0:50]], epochs=1, batch_size=MODEL_CONFIG['batch_size'])  # CVAE
        history = vae.fit(x=[train_X, train_y_onehot], epochs=1, batch_size=MODEL_CONFIG['batch_size'])  # CVAE
        
        loss_all.append(history.history['loss'][0])
        reconstruction_loss_all.append(history.history['reconstruction_loss'][0])
        kl_loss_all.append(history.history['kl_loss'][0])
        
        log_epoch.write("{:d}, {:.4f}, {:.4f}, {:.4f} \n".format(
                epoch + 1, history.history['loss'][0], history.history['reconstruction_loss'][0], history.history['kl_loss'][0]
                )
        )
        
        # run evaluation
        if TRAIN_CONFIG['evaluate_along_training']:            
            if (epoch + 1) % 10 == 0:  
            # if (epoch + 1) % 2 == 0:    
                vae.run_sampler(epoch + 1)
                vae.save_model(epoch + 1)
                # vae.plot_latent_space(train_X, epoch + 1)  # JG
                vae.run_eval(epoch + 1)
    
    # print epoch info
    training_time = time.time() - training_start_time
    
    log_epoch.close()
    
    
    if  TRAIN_CONFIG['verbose']:
        print('{:-^80}'.format(' Training time '))
        print("time   {:8.2f} s".format(training_time)) 
        print("time formated: ", str(timedelta(seconds=training_time)))
    # vae.fit(train_X, epochs = 1, batch_size = 4)  
    
    jg_show_loss(loss_all, reconstruction_loss_all, kl_loss_all, name=vae.nazwa)
    print('{:-^80}'.format(' End '))
    
#*********** Główna pętla trenowania End

   
def main_2(): 
    train_X = load_data()  
    # Ładowanie modelu 
    vae = CVAE(MODEL_CONFIG)
    
    epoch = 100    
    vae.load_model(epoch)
    # vae.run_sampler(epoch, save_midi=True)  
    emo=0
    vae.run_sampler_emo(epoch, emo, save_midi=True )  # generowanie plików z emo
    # vae.plot_latent_space(train_X, epoch + 1)  # oprawiÄ‡ Ĺ‚adowanie encoder Sampling

    
def main_3():  # test CustomLayer Sampling    
    vae = VAE(MODEL_CONFIG)
    opt = keras.optimizers.Adam(1e-4)
    vae.compile(optimizer=opt)
    
    epoch = 100
    vae.save_model(epoch)
    epoch = 101
    vae.load_model(epoch)
    vae.encoder.summary()


#**************** Testowanie Keras Tuner Start *******************************    
def model_builder(hp):
    vae = VAE(MODEL_CONFIG)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])

    opt = keras.optimizers.Adam(learning_rate=hp_learning_rate)
    vae.compile(optimizer=opt)
    return vae

        
def main_4():  # Testowanie Keras Tuner  
    train_X = load_data() 
    
    tuner = kt.Hyperband(model_builder,
                     objective='loss',
                     max_epochs=10,
                     factor=3,
                     directory='KTuner_my_dir',
                     project_name='VAE_11')

    tuner.search(train_X, epochs=50, batch_size=MODEL_CONFIG['batch_size'])

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"""
The hyperparameter search is complete.  the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")
#**************** Testowanie Keras Tuner End ******************************* 

    
def main_5(): 
    ''' testowanie Ĺ‚adowania danych '''
    # load_data_train_test()
    train_X, train_y = load_data_emo()
    train_y_onehot = labels_to_categorical(train_y)
    print("train_y_onehot.shape: ", train_y_onehot.shape)
    
if __name__ == '__main__':
    
    main_train()
    
    # main_5()
    # main_2()
    # main_3()
    # main_4()
    
