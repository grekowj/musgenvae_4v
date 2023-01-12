'''
Created on 10 gru 2021

@author: jacek
'''
import os
import shutil
import distutils.dir_util
import importlib
import numpy as np
import tensorflow as tf

SETUP = {
    'exp_name': None,
    # nazwa eksperymentu
    
    # 'training_data': './training_data/my_piano_rep_358.npz',   # data bez labeling
    'training_data': './training_data/my_tensor_338_v2.npz',  # emo-data
    # Plik z danymi treningowymi
    
    'prefix': 'jacekbachdata',
    # 'jacekbachdata' "jacekbachdata_80_20"  "jacekbachdata_oldData"  'jacekbachdata_oldData_GenBN'
    # Prefix for the experiment name. Useful when training with different
    # training data to avoid replacing the previous experiment outputs.
    
    'training_phase': 'train',
    # {'train', 'pretrain'}
    
    'preset_enc': 'proposed_3',
    # {'proposed', 'proposed_2',  'proposed_small', None}
    # Use a preset network architecture for the generator or set to None and
    # setup `MODEL_CONFIG['net_g']` to define the network architecture.

    'preset_dec': 'proposed_3',
    # {'proposed',  'proposed_2','proposed_small', 'ablated', 'baseline', None}
    # Use a preset network architecture for the discriminator or set to None
    # and setup `MODEL_CONFIG['net_d']` to define the network architecture.
    }

DATA_CONFIG = {
    'training_data': None,
    }
DATA_CONFIG['training_data'] = SETUP['training_data']

TRAIN_CONFIG = {
    'num_epoch': 100,  # 20, 100, 300
    'evaluate_along_training': True,
    'sample_along_training': True,
    
    'verbose': True,
    
    }
#===============================================================================
#========================== Experiment Configuration ===========================
#===============================================================================
EXP_CONFIG = {
    'exp_name': None,
    
    }
EXP_CONFIG['exp_name'] = '_'.join(
                (SETUP['prefix'], SETUP['training_phase'], 'enc',
                 SETUP['preset_enc'], 'dec', SETUP['preset_dec'],
                 )
                )

#===============================================================================
#============================= Model Configuration =============================
#===============================================================================
MODEL_CONFIG = {
    'num_timestep': 64,
    'num_pitch': 60,
    
    'num_labels': 4,
    
    'vae_type': 'vae',  # 'sigma-vae'
    
    'batch_size': 4,  # 4, 8, 16
    'latent_dim': 32,
    
    # Tracks
    # 'track_names': (
    #     'Drums', 'Piano', 'Guitar', 'Bass', 'Ensemble', 'Reed', 'Synth Lead',
    #     'Synth Pad'
    # ),
    'track_names': (
        'All',
    ),
    # Samples  # number of genarating samples
    'num_sample': 10,
    
     # Metrics
    'metric_map': np.array([
        # indices of tracks for the metrics to compute
        [True] * 8,  # empty bar rate
        [True] * 8,  # number of pitch used
        
    ], dtype=bool),
    
    'scale_mask_Cmajor': list(map(bool, [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])),
    'scale_mask_Cminor': list(map(bool, [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1])),
    
    'result_dir': './../result_dir/',
    'run_main_program': False,
    
    # Directories
    'checkpoint_dir': None,
    'sample_dir': None,
    'eval_dir': None,
    'log_dir': None,
    'src_dir': None,
    
    'verbose_print_summary': False,
    }

# Set default directories
if MODEL_CONFIG['checkpoint_dir'] is None:
    MODEL_CONFIG['checkpoint_dir'] = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 'exp',
        EXP_CONFIG['exp_name'], 'checkpoints'
    )
if MODEL_CONFIG['sample_dir'] is None:
    MODEL_CONFIG['sample_dir'] = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 'exp',
        EXP_CONFIG['exp_name'], 'samples'
    )
if MODEL_CONFIG['eval_dir'] is None:
    MODEL_CONFIG['eval_dir'] = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 'exp',
        EXP_CONFIG['exp_name'], 'eval'
    )
if MODEL_CONFIG['log_dir'] is None:
    MODEL_CONFIG['log_dir'] = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 'exp',
        EXP_CONFIG['exp_name'], 'logs'
    )
if MODEL_CONFIG['src_dir'] is None:
    MODEL_CONFIG['src_dir'] = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 'exp',
        EXP_CONFIG['exp_name'], 'src'
    )

#===============================================================================
#=================== Make directories & Backup source code =====================
#===============================================================================
# Make sure directories exist
for path in (MODEL_CONFIG['checkpoint_dir'], MODEL_CONFIG['sample_dir'],
             MODEL_CONFIG['eval_dir'], MODEL_CONFIG['log_dir'],
             MODEL_CONFIG['src_dir']):
    # print(path)
    if not os.path.exists(path):
         os.makedirs(path)
         
# Backup source code
for path in os.listdir(os.path.dirname(os.path.realpath(__file__))):
    if os.path.isfile(path):
        # print(os.path.basename(path) )
        if path.endswith('.py'):
            shutil.copyfile(
                os.path.basename(path),
                os.path.join(MODEL_CONFIG['src_dir'], os.path.basename(path))
            )  
            
distutils.dir_util.copy_tree(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model'),
    os.path.join(MODEL_CONFIG['src_dir'], 'model')
)                   
