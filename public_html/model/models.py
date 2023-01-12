'''
Created on 10 gru 2021

@author: jacek
'''
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
import muspy
from utils.metrics import Metrics
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.models import load_model

from config import MODEL_CONFIG


# #   sampling layer
@tf.keras.utils.register_keras_serializable()
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""  #

    def __init__(self, **kwargs):
        super(Sampling, self).__init__(**kwargs)
        
    def get_config(self):
        config = super().get_config()
        return config

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# latent_dim = 2
# latent_dim = 32
# INPUT_SHAPE = (train_X.shape[1], train_X.shape[2],  1)


def define_encoder(config): 
    in_shape = (config['num_timestep'], config['num_pitch'], 1)  # JG określanie input shape
    latent_dim = config['latent_dim']
    
    encoder_inputs = keras.Input(shape=in_shape)
    x = layers.Conv2D(128, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = layers.Conv2D(256, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling(name='Sampling_layer')([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    # encoder.summary()
    
    # Vizualizacja struktury
    # from tensorflow.keras.utils import plot_model
    # plot_model(encoder, to_file='VAE-class-encoder.png', show_shapes=True)
    
    return encoder


def define_CVAE_encoder(config): 
    in_shape = (config['num_timestep'], config['num_pitch'], 1)  # JG określanie input shape
    latent_dim = config['latent_dim']
    
    pianoroll_input = keras.Input(shape=in_shape, name="pianoroll")
    
    in_shape_labels = (config['num_labels'],)    
    labels_input = keras.Input(shape=in_shape_labels, name="emotion_labels")  # JG dodanie label
    
    labels = layers.Dense(config['num_timestep'] * config['num_pitch'])(labels_input)
    labels = layers.Reshape(in_shape)(labels)
        
    encoder_inputs = layers.concatenate([pianoroll_input, labels])  # JG połączenie tensorów
    
    x = layers.Conv2D(128, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = layers.Conv2D(256, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling(name='sampling_layer')([z_mean, z_log_var])
    encoder = keras.Model([pianoroll_input, labels_input], [z_mean, z_log_var, z], name="CVAE_encoder")
    # encoder.summary()    
    # Vizualizacja struktury
    # from tensorflow.keras.utils import plot_model
    # plot_model(encoder, to_file='VAE-class-encoder.png', show_shapes=True)    
    return encoder

def define_CVAE_mus1_encoder (config): 
    kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev=0.02) 
       
    in_shape = (config['num_timestep'], config['num_pitch'], 1)  # JG określanie input shape
    latent_dim = config['latent_dim']    
    # encoder_inputs = keras.Input(shape=in_shape)
    pianoroll_input = keras.Input(shape=in_shape, name="pianoroll")
    
    in_shape_labels = (config['num_labels'],)    
    labels_input = keras.Input(shape=in_shape_labels, name="emotion_labels")  # JG dodanie label
    
    labels = layers.Dense(config['num_timestep'] * config['num_pitch'])(labels_input)
    labels = layers.Reshape(in_shape)(labels)
        
    encoder_inputs = layers.concatenate([pianoroll_input, labels])  # JG połączenie tensorów
    
    x = encoder_inputs
    layer1_units = 64
    layer2_units = 128
    # layer1_units = 128
    # layer2_units = 256
    p1 = layers.Conv2D(layer1_units, kernel_size=(1, 12), strides=(1, 12))(x)
    p1 = layers.LeakyReLU()(p1)
    
    p1 = layers.Conv2D(layer2_units, kernel_size=(4, 1), strides=(4, 1))(p1)
    p1 = layers.LeakyReLU()(p1)
    
    p2 = layers.Conv2D(layer1_units, kernel_size=(4, 1), strides=(4, 1))(x)
    p2 = layers.LeakyReLU()(p2)
    
    p2 = layers.Conv2D(layer2_units, kernel_size=(1, 12), strides=(1, 12))(p2)
    p2 = layers.LeakyReLU()(p2)
    
    x = layers.Concatenate(axis=3)([p1, p2]) 
        
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model( [pianoroll_input, labels_input], [z_mean, z_log_var, z], name="encoder_2")
    
    # encoder = keras.Model(encoder_inputs, x, name="encoder")
    # encoder.summary() 
    # Vizualizacja struktury
    # from tensorflow.keras.utils import plot_model
    # plot_model(encoder, to_file='CVAE-mus1-encoder.png', show_shapes=True)   
          
    return encoder


def define_CVAE_decoder(config): 
    
    in_shape_labels = (config['num_labels'],)    
    labels_input = keras.Input(shape=in_shape_labels, name="emotion_labels")  # JG dodanie label
    
    latent_dim = config['latent_dim']
    latent_inputs = keras.Input(shape=(latent_dim,), name="z_sampling")
    
    x = layers.concatenate([latent_inputs, labels_input])
    
    x = layers.Dense(16 * 15 * 256, activation="relu")(x)
    x = layers.Reshape((16, 15, 256))(x)
    x = layers.Conv2DTranspose(256, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")(x)
    # x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    # x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
    decoder = keras.Model([latent_inputs, labels_input], decoder_outputs, name="CVAE_decoder")
    # decoder.summary()

    # Vizualizacja struktury
    # from tensorflow.keras.utils import plot_model
    # plot_model(decoder, to_file='VAE-class-decoder.png', show_shapes=True)
    
    return decoder

def define_CVAE_mus1_decoder(config): 
    
    in_shape_labels = (config['num_labels'],)    
    labels_input = keras.Input(shape=in_shape_labels, name="emotion_labels")  # JG dodanie label
   
    
    latent_dim = config['latent_dim']
    latent_inputs = keras.Input(shape=(latent_dim,), name="latent_space" )
    
    x = layers.concatenate([latent_inputs, labels_input])
    # x = latent_inputs
    
    x = layers.Dense(16 * 5 * 128)(x)
    # x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Reshape((16, 5, 128))(x)
    layer1_units = 64
    layer2_units = 128
    # layer1_units = 128
    # layer2_units = 256
    
    p1 = layers.Conv2DTranspose(layer2_units, kernel_size=(1, 12), strides=(1, 12))(x)
    # p1 = layers.BatchNormalization()(p1)  # JG Opcja on/off
    p1 = layers.ReLU()(p1)
    p1 = layers.Conv2DTranspose(layer1_units, kernel_size=(4, 1), strides=(4, 1))(p1)
    # p1 = layers.BatchNormalization()(p1) # JG Opcja on/off
    p1 = layers.ReLU()(p1)
    
    p2 = layers.Conv2DTranspose(layer2_units, kernel_size=(4, 1), strides=(4, 1))(x)
    # p2 = layers.BatchNormalization()(p2) # JG Opcja on/off
    p2 = layers.ReLU()(p2)
    p2 = layers.Conv2DTranspose(layer1_units, kernel_size=(1, 12), strides=(1, 12))(p2)
    # p2 = layers.BatchNormalization()(p2) # JG Opcja on/off
    p2 = layers.ReLU()(p2)
    
    x = layers.Concatenate(axis=3)([p1, p2])
    
    decoder_outputs = layers.Conv2DTranspose(1, kernel_size=(1, 1), strides=(1, 1), activation="sigmoid", padding="same")(x)
       
    # decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder_2")
    decoder = keras.Model([latent_inputs, labels_input], decoder_outputs, name="decoder_2")
    # decoder.summary() 
    # # Vizualizacja struktury
    # from tensorflow.keras.utils import plot_model
    # plot_model(decoder, to_file='CVAE-mus1-decoder.png', show_shapes=True)  
       
    return decoder


def define_encoder_2 (config): 
    kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev=0.02) 
       
    in_shape = (config['num_timestep'], config['num_pitch'], 1)  # JG określanie input shape
    latent_dim = config['latent_dim']
    
    encoder_inputs = keras.Input(shape=in_shape)
    x = encoder_inputs
    layer1_units = 32
    layer2_units = 64
    p1 = layers.Conv2D(layer1_units, kernel_size=(1, 12), strides=(1, 12))(x)
    p1 = layers.LeakyReLU()(p1)
    
    p1 = layers.Conv2D(layer2_units, kernel_size=(4, 1), strides=(4, 1))(p1)
    p1 = layers.LeakyReLU()(p1)
    
    p2 = layers.Conv2D(layer1_units, kernel_size=(4, 1), strides=(4, 1))(x)
    p2 = layers.LeakyReLU()(p2)
    
    p2 = layers.Conv2D(layer2_units, kernel_size=(1, 12), strides=(1, 12))(p2)
    p2 = layers.LeakyReLU()(p2)
    
    x = layers.Concatenate(axis=3)([p1, p2]) 
        
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder_2")
    # encoder = keras.Model(encoder_inputs, x, name="encoder")
    # encoder.summary()       
    return encoder


def define_decoder_2(config): 
    latent_dim = config['latent_dim']
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = latent_inputs
    
    x = layers.Dense(16 * 5 * 128)(x)
    # x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Reshape((16, 5, 128))(x)
    layer1_units = 32
    layer2_units = 64
    
    p1 = layers.Conv2DTranspose(layer2_units, kernel_size=(1, 12), strides=(1, 12))(x)
    # p1 = layers.BatchNormalization()(p1)
    p1 = layers.ReLU()(p1)
    p1 = layers.Conv2DTranspose(layer1_units, kernel_size=(4, 1), strides=(4, 1))(p1)
    # p1 = layers.BatchNormalization()(p1)
    p1 = layers.ReLU()(p1)
    
    p2 = layers.Conv2DTranspose(layer2_units, kernel_size=(4, 1), strides=(4, 1))(x)
    # p2 = layers.BatchNormalization()(p2)
    p2 = layers.ReLU()(p2)
    p2 = layers.Conv2DTranspose(layer1_units, kernel_size=(1, 12), strides=(1, 12))(p2)
    # p2 = layers.BatchNormalization()(p2)
    p2 = layers.ReLU()(p2)
    
    x = layers.Concatenate(axis=3)([p1, p2])
    
    decoder_outputs = layers.Conv2DTranspose(1, kernel_size=(1, 1), strides=(1, 1), activation="sigmoid", padding="same")(x)
       
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder_2")
    # decoder.summary()    
    return decoder


def define_encoder_3 (config): 
    kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev=0.02) 
       
    in_shape = (config['num_timestep'], config['num_pitch'], 1)  # JG określanie input shape
    latent_dim = config['latent_dim']
    
    encoder_inputs = keras.Input(shape=in_shape)
    x = encoder_inputs
    layer1_units = 64
    layer2_units = 128
    p1 = layers.Conv2D(layer1_units, kernel_size=(1, 12), strides=(1, 12))(x)
    p1 = layers.LeakyReLU()(p1)
    
    p1 = layers.Conv2D(layer2_units, kernel_size=(4, 1), strides=(4, 1))(p1)
    p1 = layers.LeakyReLU()(p1)
    
    p2 = layers.Conv2D(layer1_units, kernel_size=(4, 1), strides=(4, 1))(x)
    p2 = layers.LeakyReLU()(p2)
    
    p2 = layers.Conv2D(layer2_units, kernel_size=(1, 12), strides=(1, 12))(p2)
    p2 = layers.LeakyReLU()(p2)
    
    x = layers.Concatenate(axis=3)([p1, p2]) 
        
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder_2")
    # encoder = keras.Model(encoder_inputs, x, name="encoder")
    # encoder.summary() 
          
    return encoder


def define_decoder_3(config): 
    latent_dim = config['latent_dim']
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = latent_inputs
    
    x = layers.Dense(16 * 5 * 128)(x)
    # x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Reshape((16, 5, 128))(x)
    layer1_units = 64
    layer2_units = 128
    
    p1 = layers.Conv2DTranspose(layer2_units, kernel_size=(1, 12), strides=(1, 12))(x)
    # p1 = layers.BatchNormalization()(p1)  # JG Opcja on/off
    p1 = layers.ReLU()(p1)
    p1 = layers.Conv2DTranspose(layer1_units, kernel_size=(4, 1), strides=(4, 1))(p1)
    # p1 = layers.BatchNormalization()(p1) # JG Opcja on/off
    p1 = layers.ReLU()(p1)
    
    p2 = layers.Conv2DTranspose(layer2_units, kernel_size=(4, 1), strides=(4, 1))(x)
    # p2 = layers.BatchNormalization()(p2) # JG Opcja on/off
    p2 = layers.ReLU()(p2)
    p2 = layers.Conv2DTranspose(layer1_units, kernel_size=(1, 12), strides=(1, 12))(p2)
    # p2 = layers.BatchNormalization()(p2) # JG Opcja on/off
    p2 = layers.ReLU()(p2)
    
    x = layers.Concatenate(axis=3)([p1, p2])
    
    decoder_outputs = layers.Conv2DTranspose(1, kernel_size=(1, 1), strides=(1, 1), activation="sigmoid", padding="same")(x)
       
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder_2")
    # decoder.summary()    
    return decoder


def define_decoder(config): 
    latent_dim = config['latent_dim']
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(16 * 15 * 256, activation="relu")(latent_inputs)
    x = layers.Reshape((16, 15, 256))(x)
    x = layers.Conv2DTranspose(256, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")(x)
    # x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    # x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    # decoder.summary()

    # Vizualizacja struktury
    # from tensorflow.keras.utils import plot_model
    # plot_model(decoder, to_file='VAE-class-decoder.png', show_shapes=True)
    
    return decoder

# d = define_decoder()


#*****************************************************************************
# # Standard  VAE
# ## jako Model z zmodyfikowaną treningowym krokiem
#*****************************************************************************
Beta_KL = 1  # współczynnik do regulacji wpływu KL_loss


# z_mean,z_log_var dla sigma-VAE
def gaussian_nll(mu, log_sigma, x):
    return 0.5 * ((x - mu) / tf.exp(log_sigma)) ** 2 + log_sigma + 0.5 * np.log(2 * np.pi)


class VAE(keras.Model):

    def __init__(self, config, nazwa='VAE', **kwargs):
    # def __init__(self, config,  name ='VAE'):
        super(VAE, self).__init__(**kwargs)
        # super(VAE, self).__init__()
        self.config = config
        self.nazwa = nazwa
        # self.encoder = define_encoder(self.config)
        # self.decoder = define_decoder(self.config)
        # self.encoder = define_encoder_2(self.config)
        # self.decoder = define_decoder_2(self.config)
        self.encoder = define_encoder_3(self.config)
        self.decoder = define_decoder_3(self.config)
        
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")  
        self.samples_pianoroll = None  # JG wygenerowane przykłady w formie pianoroll
        self.samples_metrics = None
        self.saver = None
        
        self.samples_metrics = Metrics(config, self.nazwa)
        
        # if config['verbose_print_summary'] :
        #     self.encoder.summary()
        #     self.decoder.summary()
            
    # def build(self):
    #     """Build the model."""    
        
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
        
    def jg_plot_rep(self, piano_representations, epoch, name):
        fig = plt.figure(figsize=(30, 20))
        
        num_samples = piano_representations.shape[0]
        axs = fig.subplots(1, num_samples) 

        for i in range(num_samples): 
            axs[i].axis('off')       
            new_my_pianoroll_rep1 = np.rot90 (piano_representations[i])
            axs[i].imshow(new_my_pianoroll_rep1, cmap='gray')
            axs[i].set_title(str(i))
    
    #     epoch = epochs
        # filename1 = 'Samples_%s_images_%04d_%s.png' % (num_samples, epoch + 1, name)
        filename1 = self.nazwa + '_' + 'Samples_%s_images_%04d_%s.png' % (num_samples, epoch, name)
        
        # result_dir = self.config['result_dir']
        # if self.config['run_main_program']: 
        #     result_dir = result_dir[3:]
            
        path = os.path.join(self.config['sample_dir'] , filename1)
        print (path)
        plt.savefig(path)
        print('%s was saved to %s' % (filename1, path))     
        # plt.show() 
          
        # generuje nowe sample       
    def jg_run_sampler(self, epoch): 
        z_sample = tf.random.normal(shape=(self.config['num_sample'], self.config['latent_dim']))
        print("z_sample.shape: ", z_sample.shape) 
        
        num_sample = self.config['num_sample']
        if self.nazwa == "CVAE":
            cond_num = np.zeros( (num_sample, 1) ) 
            print("cond_num.shape: ", cond_num.shape)
            for i in range(num_sample):
                ind = (i % 4)
                print(ind)
                cond_num[i] = ind
            condition_emo = to_categorical(cond_num, 4)
            x_decoded = self.decoder.predict([z_sample, condition_emo])
        
        else:
            x_decoded = self.decoder.predict(z_sample)
                
        
        x_decoded_round = x_decoded > 0.5
        print("x_decoded.shape: ", x_decoded.shape) 
        #     jg_plot_rep(x_decoded, config, name = 'vae') 
        self.jg_plot_rep(x_decoded_round, epoch, name='vae_rounded') 
        return x_decoded_round
    
     # generuje nowe sample  z emo     
    def jg_run_sampler_emo(self, epoch, emo): 
        num_sample = 1
        z_sample = tf.random.normal(shape=(num_sample, self.config['latent_dim']))
        print("z_sample.shape: ", z_sample.shape)       
       
        
        cond_num = np.zeros( (num_sample, 1) ) 
        print("cond_num.shape: ", cond_num.shape)
        cond_num[0] = emo
            
        condition_emo = to_categorical(cond_num, 4)
        x_decoded = self.decoder.predict([z_sample, condition_emo])      
                
        
        x_decoded_round = x_decoded > 0.5
        print("x_decoded.shape: ", x_decoded.shape) 
        
        # self.jg_plot_rep(x_decoded_round, epoch, name='vae_rounded') 
        return x_decoded_round
        
    def jg_back_to_pianoroll(self, samples): 
        print("samples: ", samples.shape) 
    
        samples_piano_representations_arr = np.reshape (samples, (samples.shape[0], samples.shape[1], samples.shape[2]))
        samples_p_rep = samples_piano_representations_arr
        samples_pianoroll_rep = np.pad(samples_p_rep, ((0, 0), (0, 0), (25, 43)) , 'constant', constant_values=(0, 0))
        print("samples_pianoroll_rep.shape: ", samples_pianoroll_rep.shape)
        return samples_pianoroll_rep
    
    # zapisuje saplme do MIDI
    def jg_save_midi_samples(self, samples_pianoroll, epoch): 
        for i in range(samples_pianoroll.shape[0]):
            my_music = muspy.from_pianoroll_representation (samples_pianoroll[i], resolution=4, encode_velocity=False)
            
            # print(my_music)
            # result_dir = self.config['result_dir']  # korekcja path
            # if self.config['run_main_program']: 
            #     result_dir = result_dir[3:]                
                
            # filename1 = 'MIDI_epoch_%04d_%s.mid' % (epoch, i)
            filename1 = self.nazwa + '_' + 'MIDI_epoch_%04d_%s.mid' % (epoch, i)
            path = os.path.join(self.config['sample_dir'], filename1)
            muspy.write_midi(path , my_music) 
            print('MIDI saved to %s' % path)
              
    #***********************************************
    # run_sampler
    # generuje nowe sample zapisuje w postacji PNG i MIDI
    #***********************************************   
    def run_sampler(self, epoch, save_midi=True):
        """Save the samples."""
        print('{:-^80}'.format(' Run sampler and save the samples '))    

        # epoch = 10 
        # result_dir = './../result_dir/'
        samples = self.jg_run_sampler(epoch)
        
        self.samples_pianoroll = self.jg_back_to_pianoroll(samples) 
        if (save_midi):
            self.jg_save_midi_samples(self.samples_pianoroll, epoch)   
        # return self.samples_pianoroll 
    
    def run_sampler_emo(self, epoch, emo, save_midi=True):
        """Save the samples."""
        print('{:-^80}'.format(' Run sampler and save the samples '))    

        # epoch = 10 
        # result_dir = './../result_dir/'
        samples = self.jg_run_sampler_emo(epoch, emo) 
        self.samples_pianoroll = self.jg_back_to_pianoroll(samples)  
        self.jg_save_midi_samples(self.samples_pianoroll, epoch)   
    
    def run_eval(self, epoch):
        """Run evaluation."""
        print('{:-^80}'.format(' Run evaluation '))
        result_dir = self.config['eval_dir']
        title_score_mat = 'score_matrices.npy'
        title_score_mat = "epoch_%04d_" % (epoch) + title_score_mat  # JG
        self.samples_metrics.eval(self.samples_pianoroll, epoch, True,
                                  mat_path=os.path.join(result_dir, title_score_mat),
                                  fig_dir=result_dir)
        pass
    
    def save_plot_model(self):
        # filename1 = 'VAE_decoder.png'
        filename1 = self.nazwa + '_decoder.png'
        path = os.path.join(self.config['log_dir'], filename1)    
        plot_model(self.decoder, to_file=path, show_shapes=True)
        print('%s saved to %s' % (filename1, path))
        
        # filename1 = 'VAE_encoder.png'
        filename1 = self.nazwa + '_encoder.png'
        path = os.path.join(self.config['log_dir'], filename1) 
        plot_model(self.encoder, to_file=path, show_shapes=True)
        print('%s saved to %s' % (filename1, path))
        
    def save_model(self, epoch):
        from tensorflow import keras
        # encoder.save('VAE-class-encoder.h5')
        # filename1 = "VAE_decoder_epoch_%04d_.h5" % (epoch)
        filename1 = self.nazwa + "_decoder_epoch_%04d_.h5" % (epoch)
        path = os.path.join(self.config['checkpoint_dir'], filename1) 
        self.decoder.save(path)
        print('%s saved to %s' % (filename1, path))   
        
        # filename1 = "VAE_encoder_epoch_%04d_.h5" % (epoch) 
        filename1 = self.nazwa + "_encoder_epoch_%04d_.h5" % (epoch) 
        path = os.path.join(self.config['checkpoint_dir'], filename1) 
        self.encoder.save(path) 
        print('%s saved to %s' % (filename1, path))
        
    def load_model(self, epoch): 

        # filename1 = "VAE_decoder_epoch_%04d_.h5" % (epoch)
        filename1 = self.nazwa + "_decoder_epoch_%04d_.h5" % (epoch)
        path = os.path.join(self.config['checkpoint_dir'], filename1) 
        self.decoder = load_model(path)
        print('%s was loaded from %s' % (filename1, path))
        
        # filename1 = "VAE_encoder_epoch_%04d_.h5" % (epoch)
        filename1 = self.nazwa + "_encoder_epoch_%04d_.h5" % (epoch)
        path = os.path.join(self.config['checkpoint_dir'], filename1) 
        self.encoder = load_model(path)
        # self.encoder = load_model(path, custom_objects={'Sampling': Sampling}) , 
        print('%s was loaded from %s' % (filename1, path))
     
    def save_summary(self):
        """Zapisuje model summary to file"""
        filename = self.nazwa + '_' + 'model_summary.txt'
        filepath = os.path.join(self.config['log_dir'], filename)
        with open(filepath, 'w') as f:
            # f.write(self.get_summary())  
            self.encoder.summary(print_fn=lambda x: f.write(x + '\n'))
            self.decoder.summary(print_fn=lambda x: f.write(x + '\n')) 
            
    def plot_latent_space(self, data, epochs, labels=None, name='model_name'):
        ''' zapisuje latent space         '''
        # display a 2D plot of the digit classes in the latent space
        z_mean, _, _ = self.encoder.predict(data)
        plt.figure(figsize=(12, 10))
        if labels == None:
            plt.scatter(z_mean[:, 0], z_mean[:, 1],)
        else:
            plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
            
        plt.colorbar()
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
    #     plt.show()
        
        # save plot to file
        #******************************
        epoch = epochs
        
            # filename1 = 'VAE_latent_space_%04d.png' % (epoch)
        filename1 = self.nazwa + '_latent_space_%04d.png' % (epoch)
        path = os.path.join(self.config['log_dir'], filename1) 
        plt.savefig(path)
        print('%s was saved to %s' % (filename1, path))
        # plt.show()
        
         #******************************
        filename = self.nazwa + '_' + 'Latent_space.log'
        filepath = os.path.join(self.config['log_dir'], filename)
        with open(filepath, 'a') as f:
            f.write ("********Parametry Latent space********************* \n") 
            f.write ("Epoch {:d} \n".format(epochs)) 
            f.write ("*************************************************** \n") 
        #     Wydrukowanie z_mean() np.std(a)
            print ("**********************")
            print ("Parametry Latent space")
            print ("**********************")
            print ("z_mean.shape", z_mean.shape) 
              
            for j in range(0, z_mean.shape[1]):
                a = z_mean[:, j]
        #         print (a.shape)
                print (j, "mean: ", np.mean(a), "std: ", np.std(a))                
                f.write ("{:d},  mean: {:.4f},  std: {:.4f} \n".format(j, np.mean(a), np.std(a)))        

    def train_step(self, data):
        """Główna pętla trenowania"""
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            
            if (self.config['vae_type'] == 'sigma-vae'):
                #**************** Sigma-VAE loss modification **********************

                log_sigma = tf.math.log(tf.sqrt(tf.reduce_mean((data - reconstruction) ** 2, [0, 1, 2, 3], keepdims=True)))
                reconstruction_loss = tf.reduce_sum(gaussian_nll(reconstruction, log_sigma, data))  # 1.JG Sigma-VAE
    #              rec_loss = tf.reduce_sum(gaussian_nll(img_rec, log_sigma, img))  

                kl_loss = -tf.reduce_sum(0.5 * (1 + z_log_var - z_mean ** 2 - tf.exp(z_log_var)))  # JG Sigma-VAE
    #              kld_loss = -tf.reduce_sum(0.5 * (1 + z_log_sigma_sq - z_mu ** 2 - tf.exp(z_log_sigma_sq)))
            else: 
            
                # JG VAE loss orginal**************************************
                reconstruction_loss = tf.reduce_mean(
                    tf.reduce_sum(
    #                     tf.keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2))
                        # JG tsprawdzić jak dział mse()
                        # tf.losses.mean_squared_error(data, reconstruction) , axis=(1, 2) ) 
                        tf.losses.mean_squared_error(data, reconstruction))  # JG lepiej bez axis
                )
                kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))  # JG tak było VAE orginal
                kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))  # JG tak było VAE orginal
            
# ********************************************************************           
            total_loss = reconstruction_loss + Beta_KL * kl_loss   
# ********************************************************************
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        x = self.decoder(z)
        return x
#************************** Standard  VAE END ***************************************************    


class CVAE(VAE):

    def __init__(self, config, nazwa='CVAE', **kwargs):
        super(CVAE, self).__init__(config, nazwa, **kwargs)
        
        self.config = config
        # self.name = name

        # self.encoder = define_encoder(self.config)
        # self.decoder = define_decoder(self.config)
        
        self.encoder = define_CVAE_encoder(self.config)   # base model
        self.decoder = define_CVAE_decoder(self.config)
        # self.encoder = define_CVAE_mus1_encoder(self.config) # mus1 model
        # self.decoder = define_CVAE_mus1_decoder(self.config)
        
     
    def train_step(self, data):
        """Główna pętla trenowania"""
        with tf.GradientTape() as tape:
            print("Główna pętla trenowania: train_step")
            pianoroll = data[0][0]
            labels = data[0][1]
            z_mean, z_log_var, z = self.encoder(data)
            
            reconstruction = self.decoder([z, labels])           
             
            
            # JG VAE loss orginal**************************************
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
#                     tf.keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2))
                    # JG tsprawdzić jak dział mse()
                    # tf.losses.mean_squared_error(data, reconstruction) , axis=(1, 2) ) 
                    tf.losses.mean_squared_error(pianoroll, reconstruction))  # JG lepiej bez axis
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))  # JG tak było VAE orginal
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))  # JG tak było VAE orginal
            
# ********************************************************************           
            total_loss = reconstruction_loss + Beta_KL * kl_loss   
# ********************************************************************
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }    
        
    def call(self, inputs):
        # print("*** UWAGA **** call --> inputs.shape: ", inputs.shape)
        y_pianoroll = inputs[0]
        y_labels = inputs[1]
        
        z_mean, z_log_var, z = self.encoder(inputs)
        x = self.decoder([z, inputs[1]])
        return x    


if __name__ == '__main__':
    # enc = define_encoder(MODEL_CONFIG)
    #
    # enc_2 = define_encoder_2(MODEL_CONFIG)
    #
    #
    # dec_2 =define_decoder_2(MODEL_CONFIG)
    
    # define_CVAE_mus1_encoder(MODEL_CONFIG)
    define_CVAE_mus1_decoder(MODEL_CONFIG)
     
    # vae = CVAE(MODEL_CONFIG)    
    # # vae = VAE(MODEL_CONFIG)
    # opt = keras.optimizers.Adam(1e-4)
    # vae.compile(optimizer=opt)
    # vae.save_plot_model()
    
   
