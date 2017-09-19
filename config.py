# coding: utf-8

import numpy as np
import os 
import os.path as op
import pickle

from rootpy.io import root_open
from root_numpy import root2array

# Training parameters
BATCH_SIZE = 64
TRAINING_RATIO = 5  
GRADIENT_PENALTY_WEIGHT = 10  
n_cells = 89
nb_epochs = 1001
SIZE_Z = 100


# X_true contains real energies (in MeV), with energy cuts
# X contains normalized energies --> we'll feed critic with events taken from this array
# E_tot contains total energy per event (condition). A logarithm and a StandardScaler have been applied
# That's why we use the exponential and scaler in the training to switc back to a value in MeV

data_path = '/media/pklein/33fd2f13-8113-44ae-9dab-6d98aa410224/pklein/Documents/CERN/DATA/condWGAN_0'
X_true = np.load(op.join(data_path, 'X_clean.npy'))
E_tot = np.load(op.join(data_path, 'E_tot.npy'))
X_unclean = np.load('/media/pklein/33fd2f13-8113-44ae-9dab-6d98aa410224/pklein/Documents/CERN/Np_Arrays/All/X_true_all.npy')
X = np.load(op.join(data_path, 'X_NORMALIZE.npy'))

with open(op.join(data_path, 'scaler_E_tot.pickle'), 'rb') as handle:
    scaler = pickle.load(handle)

models_path = '/media/pklein/33fd2f13-8113-44ae-9dab-6d98aa410224/pklein/Documents/CERN/GRID_scripts/Amir_Grid/Layers'

f = root_open("/media/pklein/33fd2f13-8113-44ae-9dab-6d98aa410224/pklein/Documents/CERN/Dataset/outputTree_10files_Cluster_tru.root")
path = '/media/pklein/33fd2f13-8113-44ae-9dab-6d98aa410224/pklein/Documents/CERN/Plots/'
t = f.tau

a = root2array("/media/pklein/33fd2f13-8113-44ae-9dab-6d98aa410224/pklein/Documents/CERN/Dataset/cluster_ok.root", "tau", ["off_cells_e",
                                                                              "off_cells_samp",
                                                                              "off_ncells",
                                                                              "off_cells_eta",
                                                                              "off_cells_phi",
                                                                              "off_pt", ])
indices = np.load(op.join(data_path, 'indices.npy'))
a = np.delete(a, indices, axis=0)








    