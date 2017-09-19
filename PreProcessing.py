# coding: utf-8

#Preprocessing steps to have relevant inputs to feed the WGAN

import numpy as np
import os.path as op
from sklearn.preprocessing import StandardScaler
import pickle

###################################  LOAD DATA  ####################################
data_path = '/home/pklein/Documents/CERN/scripts/Models'
X_ = np.load(op.join(data_path, 'X.npy')) #load energy cells deposit in the cluster
print "Dataset containing {} events\n".format(X_.shape[0])
print "Energy located in {} cells for each event".format(X_.shape[1])


############################  REDUCE DATASET (<300GeV)  ############################
E_tot_ = np.zeros(X_.shape[0])
for i in range(X_.shape[0]):
    E_tot_[i] = X_[i].sum()
    
indices = np.where(E_tot_>300000)
X_ = np.delete(X_, indices, axis=0)
E_tot = np.delete(E_tot_, indices)


#############################  REMOVE ELECTRONIC NOISE  ############################
#The threshold in MeV depends on the layer of the calorimeter:
# - 200 MeV for Middle
# - 60 MeV for Strip
# - 120 MeV for Pre-Sampler
# - 200 MeV for Back

for i in range(X_.shape[0]):
    for j in range(89):
        if (j<21) & (X_[i][j] < 200):
            X_[i][j] = 0
        elif (21<=j<69) & (X_[i][j] < 60):
            X_[i][j] = 0
        elif (69<=j<75) & (X_[i][j] < 120):
            X_[i][j] = 0
        elif (75<=j<89) & (X_[i][j] < 200):
            X_[i][j] = 0

np.save('X_true.npy', X_) # --> energy cells we'll use as input of the Critic


########################  NORMALIZE ENERGIES for the training  #######################
X = np.divide(X_, X_.sum(axis=1)[:, None])
np.save('X_NORMALIZE.npy', X)


##########################  Preprocessing for total energy  ##########################
E_tot_ = np.zeros(X_.shape[0])
for i in range(X_.shape[0]):
    E_tot_[i] = X_[i].sum()

E_tot = np.log(E_tot_) #apply a logarithm to have a lower span of values
scaler = StandardScaler()
E_tot = scaler.fit_transform(E_tot.reshape(-1,1)) # total energy of the event well use as input of both Generator and Critic

np.save('E_tot.npy', E_tot)
with open('scaler_E_tot.pickle', 'wb') as handle:
    scaler = pickle.dump(scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)

