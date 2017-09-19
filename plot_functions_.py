# coding: utf-8

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from cycler import cycler
import os
import os.path as op
from functools import partial

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import (SIZE_Z,
                    X_true,
                    E_tot,
                    data_path,
                    scaler,
                    a,
                    X_unclean)


plt.rc('axes', prop_cycle=(cycler('color', ['b', 'k', 'm', 'c', 'g', 'y'])))

X_t = X_true

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def energy_per_event(X_t=X_t, **kwargs):
    energy_tot = []
    '''Compute the total energy of each event, by summing all cells energy deposit in the cluster. '''
    E_tot_t = np.array([X_i.sum() for X_i in X_t])
    energy_tot.append(E_tot_t)
    for key, value in kwargs.items():
        energy_tot.append(np.array([X_i.sum() for X_i in value]))
    return energy_tot


def energy_per_layer(X_t=X_t, **kwargs):
    '''Compute the total energy per layer of the calorimeter for each event.
    Returns two arrays of shape (nb_events, 4).
    Each rows defines an event, each column a layer of the calorimeter:
    col 0 for the Pre-Sampler, 1 for the Strip, 2 for the Middle and 3 for the Back. '''
    
    energy_layer = []
    E_layer_t = np.zeros((X_t.shape[0], 4))
    for i in range(E_layer_t.shape[0]):
        E_layer_t[i][0] = X_t[i][69:75].sum() #Pre_sampler
        E_layer_t[i][1] = X_t[i][21:69].sum() #Strip
        E_layer_t[i][2] = X_t[i][:21].sum() #Middle
        E_layer_t[i][3] = X_t[i][75:89].sum() #Back
    energy_layer.append(E_layer_t)

    for k in sorted(kwargs):
        E_layer_g = np.zeros((X_t.shape[0], 4))
        for i in range(E_layer_g.shape[0]):
            E_layer_g[i][0] = kwargs[k][i][69:75].sum() #Pre_sampler
            E_layer_g[i][1] = kwargs[k][i][21:69].sum() #Strip
            E_layer_g[i][2] = kwargs[k][i][:21].sum() #Middle
            E_layer_g[i][3] = kwargs[k][i][75:89].sum() #Back
        energy_layer.append(E_layer_g)

    return energy_layer


def plot_energy_per_layer(E_layer_t=None, epoch=None, directory=None, **kwargs):

    fig, ax = plt.subplots(2, 2, figsize=(10,10))
    
    ax[0, 0].hist(E_layer_t[:,0], bins=50, range=(0,20000), color='r', normed=1, histtype='step', label="Monte-Carlo")
    ax[0, 0].set_title('Total Energy per event in SAMPLER', fontsize=16)
    ax[0, 0].set_xlabel('Energy (in MeV)', fontsize=14)
    
    ax[0, 1].hist(E_layer_t[:,1], bins=50, range=(0,100000), color='r', normed=1, histtype='step', label="Monte-Carlo")
    ax[0, 1].set_title('Total Energy per event in STRIP', fontsize=16)
    ax[0, 1].set_xlabel('Energy (in MeV)', fontsize=14)
    
    ax[1, 0].hist(E_layer_t[:,2], bins=50, range=(0,300000), color='r', normed=1, histtype='step', label="Monte-Carlo")
    ax[1, 0].set_title('Total Energy per event in MIDDLE', fontsize=16)
    ax[1, 0].set_xlabel('Energy (in MeV)', fontsize=14)
    
    ax[1, 1].hist(E_layer_t[:,3], bins=50, range=(0,3000), color='r', normed=1, histtype='step', label="Monte-Carlo")
    ax[1, 1].set_title('Total Energy per event in BACK', fontsize=16)
    ax[1,1].set_xlabel('Energy (in MeV)', fontsize=14)
    
    
    for k in sorted(kwargs):
        
        ax[0, 0].hist(kwargs[k][:,0], bins=50, range=(0,20000), normed=1, histtype='step', label=k, linestyle=':')
        ax[0, 1].hist(kwargs[k][:,1], bins=50, range=(0,100000), normed=1, histtype='step', label=k, linestyle=':')  
        ax[1, 0].hist(kwargs[k][:,2], bins=50, range=(0,300000), normed=1, histtype='step', label=k, linestyle=':')   
        ax[1, 1].hist(kwargs[k][:,3], bins=50, range=(0,3000), normed=1, histtype='step', label=k, linestyle=':')
        
         
    ax[0, 0].legend()
    ax[0, 1].legend()
    ax[1, 0].legend()
    ax[1, 1].legend()
    ax[0, 0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax[0, 1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax[1, 0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax[1, 1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    plt.savefig(op.join(directory, 'Layer_energies_{}.eps'.format(epoch)), dpi=600)
    plt.savefig(op.join(directory, 'Layer_energies_{}.pdf'.format(epoch)), dpi=600)
    plt.close()


def energy_per_layer_normed(E_layer_t=None, E_tot_t=None, **kwargs):
    '''Compute the percent in energy for each layer of the calorimeter.
    Returns two arrays of shape (nb_events, 4).
    For kwargs, feed with tuples (E_layer, E_tot)'''
    energy_layer_normed = []

    E_layer_norm_t = np.zeros((X_t.shape[0], 4))
    for i in range(X_t.shape[0]):
        for j in range(4):
            E_layer_norm_t[i][j] = E_layer_t[i][j]/E_tot_t[i]
    energy_layer_normed.append(E_layer_norm_t)

    for k in sorted(kwargs):
        
        E_layer_norm_g = np.zeros((X_t.shape[0], 4))
        for i in range(X_t.shape[0]):
            for j in range(4):
                E_layer_norm_g[i][j] = kwargs[k][0][i][j]/kwargs[k][1][i]
        energy_layer_normed.append(E_layer_norm_g)

    return energy_layer_normed


def plot_energy_per_layer_normed(E_layer_norm_t=None, epoch=None, directory=None, **kwargs):

    fig, ax = plt.subplots(2, 2, figsize=(10,10))
    ax[0, 0].hist(E_layer_norm_t[:,0], bins=100, range=(0,1), color='r', normed=1, histtype='step', label="Monte-Carlo")
    ax[0, 0].set_title('Proportion of Energy in PRE-SAMPLER', fontsize=16)
    ax[0, 0].set_xlabel('Percentage (%)', fontsize=14)

    ax[0, 1].hist(E_layer_norm_t[:,1], bins=100, range=(0,1), color='r', normed=1,histtype='step', label="Monte-Carlo")
    ax[0, 1].set_title('Proportion of  Energy in STRIP', fontsize=16)
    ax[0, 1].set_xlabel('Percentage (%)', fontsize=14)

    ax[1, 0].hist(E_layer_norm_t[:,2], bins=100, range=(0,1), color='r', normed=1,histtype='step', label="Monte-Carlo")
    ax[1, 0].set_title('Proportion of Energy in MIDDLE', fontsize=16)
    ax[1, 0].set_xlabel('Percentage (%)', fontsize=14)

    ax[1, 1].hist(E_layer_norm_t[:,3], bins=100, range=(0,1), color='r', normed=1,histtype='step', label="Monte-Carlo")
    ax[1, 1].set_title('Proportion of Energy in BACK', fontsize=16)
    ax[1, 1].set_xlabel('Percentage (%)', fontsize=14)

    for k in sorted(kwargs):

        ax[0, 0].hist(kwargs[k][:,0], bins=100, range=(0,1), normed=1,histtype='step', label=k, linestyle=':')
        ax[0, 1].hist(kwargs[k][:,1], bins=100, range=(0,1), normed=1,histtype='step', label=k, linestyle=':')
        ax[1, 0].hist(kwargs[k][:,2], bins=100, range=(0,1), normed=1,histtype='step', label=k, linestyle=':')
        ax[1, 1].hist(kwargs[k][:,3], bins=100, range=(0,1), normed=1,histtype='step', label=k, linestyle=':')

    ax[1, 0].legend(loc='upper left')
    ax[1, 1].legend(loc='upper center')
    ax[0, 1].legend(loc='upper right')
    ax[0, 0].legend(loc='upper center')

    plt.savefig(op.join(directory, 'Normed_Layer_Energies_{}.eps'.format(epoch)), dpi=600)
    plt.savefig(op.join(directory, 'Normed_Layer_Energies_{}.pdf'.format(epoch)), dpi=600)
    plt.close()


def energy_centre_middle(X_t=X_t, E_layer_t=None, **kwargs):
    ''' Compute E(3*3)/E(3*7), taking cluster at the center of the 3*7 rectangle cells in Middle'''
    centre_middle = []
    E_middle_t = E_layer_t[:,2]
    E_middle_centre_t = np.zeros((X_t.shape[0]))


    for i in range(X_t.shape[0]):
        E_middle_centre_t[i] = X_t[i][:21][::-1].reshape(3,7).transpose()[2:5].sum()/E_middle_t[i]
    centre_middle.append(E_middle_centre_t)
        

    for k in sorted(kwargs):
        E_middle_g = kwargs[k][1][:,2]
        E_middle_centre_g = np.zeros((X_t.shape[0]))
        for j in range(X_t.shape[0]):
            E_middle_centre_g[j] = kwargs[k][0][j][:21][::-1].reshape(3,7).transpose()[2:5].sum()/E_middle_g[j]
        centre_middle.append(E_middle_centre_g)

    return centre_middle


def plot_centre_middle(E_middle_centre_t=None, directory=None, **kwargs):
    plt.hist(E_middle_centre_t, histtype='step', color='r', label='Monte-Carlo', bins=30, normed=1, range=(0.7,1))

    for k in sorted(kwargs):
        plt.hist(kwargs[k], histtype='step', label=k, bins=30, normed=1, range=(0.7,1), linestyle=':')

    plt.legend(loc='upper center')
    plt.title('Percent of energy in center of Middle layer', fontsize=16) 
    plt.xlabel('Percentage (%)', fontsize=14)
    plt.savefig(op.join(directory, 'Centre_Middle.eps'), dpi=600)
    plt.savefig(op.join(directory, 'Centre_Middle.pdf'), dpi=600)
    plt.close()



def avg_eta(X_t=X_t, E_layer_t=None, **kwargs):
    avg_eta = []
    avg_eta_t = np.zeros((X_t.shape[0], 4))
    for i in range(X_t.shape[0]):
        avg_eta_t[i][0] = np.dot((a[i][3][69:75]-a[i][3][10]), X_t[i][69:75])/E_layer_t[i][0] #pre-sampler
        avg_eta_t[i][1] = np.dot((a[i][3][21:69]-a[i][3][10]), X_t[i][21:69])/E_layer_t[i][1] #strip
        avg_eta_t[i][2] = np.dot((a[i][3][:21]-a[i][3][10]), X_t[i][:21])/E_layer_t[i][2] #middle
        avg_eta_t[i][3] = np.dot((a[i][3][75:89]-a[i][3][10]), X_t[i][75:89])/E_layer_t[i][3] #back
    avg_eta.append(avg_eta_t)

    for k in sorted(kwargs):
        avg_eta_g = np.zeros((X_t.shape[0], 4))
        for i in range(X_t.shape[0]):
            avg_eta_g[i][0] = np.dot((a[i][3][69:75]-a[i][3][10]), kwargs[k][0][i][69:75])/kwargs[k][1][i][0] #pre-sampler
            avg_eta_g[i][1] = np.dot((a[i][3][21:69]-a[i][3][10]), kwargs[k][0][i][21:69])/kwargs[k][1][i][1] #strip
            avg_eta_g[i][2] = np.dot((a[i][3][:21]-a[i][3][10]), kwargs[k][0][i][:21])/kwargs[k][1][i][2] #middle
            avg_eta_g[i][3] = np.dot((a[i][3][75:89]-a[i][3][10]), kwargs[k][0][i][75:89])/kwargs[k][1][i][3] #back
        avg_eta.append(avg_eta_g)

    return avg_eta


def avg_phi(X_t=X_t, E_layer_t=None, **kwargs):
    avg_phi=[]
    avg_phi_t = np.zeros((X_t.shape[0], 4))
    for i in range(X_t.shape[0]):
        avg_phi_t[i][0] = np.dot((a[i][4][69:75]-a[i][4][10]), X_t[i][69:75])/E_layer_t[i][0] #pre-sampler
        avg_phi_t[i][1] = np.dot((a[i][4][21:69]-a[i][4][10]), X_t[i][21:69])/E_layer_t[i][1] #strip
        avg_phi_t[i][2] = np.dot((a[i][4][:21]-a[i][4][10]), X_t[i][:21])/E_layer_t[i][2] #middle
        avg_phi_t[i][3] = np.dot((a[i][4][75:89]-a[i][4][10]), X_t[i][75:89])/E_layer_t[i][3] #back
    avg_phi.append(avg_phi_t)

    for k in sorted(kwargs):
        avg_phi_g = np.zeros((X_t.shape[0], 4))
        for i in range(X_t.shape[0]):
            avg_phi_g[i][0] = np.dot((a[i][4][69:75]-a[i][4][10]), kwargs[k][0][i][69:75])/kwargs[k][1][i][0] #pre-sampler
            avg_phi_g[i][1] = np.dot((a[i][4][21:69]-a[i][4][10]), kwargs[k][0][i][21:69])/kwargs[k][1][i][1] #strip
            avg_phi_g[i][2] = np.dot((a[i][4][:21]-a[i][4][10]), kwargs[k][0][i][:21])/kwargs[k][1][i][2] #middle
            avg_phi_g[i][3] = np.dot((a[i][4][75:89]-a[i][4][10]), kwargs[k][0][i][75:89])/kwargs[k][1][i][3] #back
        avg_phi.append(avg_phi_g)

    return avg_phi


def plot_avg_eta(avg_eta_t=None, directory=None, **kwargs):
    fig, ax = plt.subplots(2, 2, figsize=(10,10))
    ax[0, 0].hist(avg_eta_t[:,0][~np.isnan(avg_eta_t[:,0])], bins=30, color='r', normed=1, histtype='step', label="Monte-Carlo", range=(-0.05, 0.05))
    ax[0, 0].set_title('Averaged Eta in PRE-SAMPLER', fontsize=16)
    ax[0, 0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    ax[0, 1].hist(avg_eta_t[:,1][~np.isnan(avg_eta_t[:,1])], bins=30, color='r', normed=1,histtype='step', label="Monte-Carlo", range=(-0.02, 0.02)) 
    ax[0, 1].set_title('Averaged Eta in STRIP', fontsize=16)    
    ax[0, 1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    ax[1, 0].hist(avg_eta_t[:,2][~np.isnan(avg_eta_t[:,2])], bins=30, color='r',  normed=1,histtype='step', label="Monte-Carlo", range=(-0.02, 0.02))   
    ax[1, 0].set_title('Averaged ETA in MIDDLE', fontsize=16)   
    ax[1, 0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    ax[1, 1].hist(avg_eta_t[:,3][~np.isnan(avg_eta_t[:,3])], bins=30, color='r',  normed=1, histtype='step', label="Monte-Carlo", range=(-0.05, 0.05))    
    ax[1, 1].set_title('Averaged ETA in BACK', fontsize=16)    
    ax[1, 1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    for k in sorted(kwargs):

        ax[0, 0].hist(kwargs[k][:,0], bins=30, normed=1,histtype='step', label=k, linestyle=':', range=(-0.05, 0.05))
        ax[0, 1].hist(kwargs[k][:,1], bins=30, normed=1,histtype='step', label=k, linestyle=':', range=(-0.02, 0.02))
        ax[1, 0].hist(kwargs[k][:,2], bins=30, normed=1,histtype='step', label=k, linestyle=':', range=(-0.02, 0.02))
        ax[1, 1].hist(kwargs[k][:,3], bins=30, normed=1, histtype='step', label=k, linestyle=':', range=(-0.05, 0.05))

    ax[0, 0].legend(prop={'size':6})
    ax[0, 1].legend(prop={'size':6})
    ax[1, 0].legend(prop={'size':6})
    ax[1, 1].legend(prop={'size':6})

    plt.savefig(op.join(directory, 'Averaged_Eta_layer.eps'), dpi=600)
    plt.savefig(op.join(directory, 'Averaged_Eta_layer.pdf'), dpi=600)
    plt.close()


def plot_avg_phi(avg_phi_t=None, directory=None, **kwargs):
    fig, ax = plt.subplots(2, 2, figsize=(10,10))
    ax[0, 0].hist(avg_phi_t[:,0][~np.isnan(avg_phi_t[:,0])], bins=30, color='r',range=(-0.25,0.25),normed=1,histtype='step', label="Monte-Carlo")   
    ax[0, 0].set_title('Averaged Phi in PRE-SAMPLER', fontsize=16)
    ax[0, 0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    ax[0, 1].hist(avg_phi_t[:,1][~np.isnan(avg_phi_t[:,1])], bins=30, color='r',normed=1, range=(-0.2,0.2),histtype='step', label="Monte-Carlo")
    ax[0, 1].set_title('Averaged Phi in STRIP', fontsize=16)
    ax[0, 1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    ax[1, 0].hist(avg_phi_t[:,2][~np.isnan(avg_phi_t[:,2])], bins=30, color='r',normed=1,histtype='step', range=(-0.05,0.05), label="Monte-Carlo")
    ax[1, 0].set_title('Averaged Phi in MIDDLE', fontsize=16)
    ax[1, 0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    ax[1, 1].hist(avg_phi_t[:,3][~np.isnan(avg_phi_t[:,3])], bins=30, color='r',normed=1,histtype='step', range=(-0.05,0.05), label="Monte-Carlo")
    ax[1, 1].set_title('Averaged Phi in BACK', fontsize=16)
    ax[1, 1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    for k in sorted(kwargs):

        ax[0, 0].hist(kwargs[k][:,0], bins=30, range=(-0.25,0.25),normed=1,histtype='step', label=k, linestyle=':')
        ax[0, 1].hist(kwargs[k][:,1], bins=30,normed=1,range=(-0.2,0.2),histtype='step', label=k, linestyle=':')
        ax[1, 0].hist(kwargs[k][:,2], bins=30, normed=1,histtype='step', range=(-0.05,0.05), label=k, linestyle=':')
        ax[1, 1].hist(kwargs[k][:,3], bins=30,normed=1,histtype='step',  range=(-0.05,0.05),label=k, linestyle=':')

    ax[0, 0].legend()
    ax[0, 1].legend()
    ax[1, 0].legend()
    ax[1, 1].legend()

    plt.savefig(op.join(directory, 'Averaged phi per layer.eps'), dpi=600)
    plt.savefig(op.join(directory, 'Averaged phi per layer.pdf'), dpi=600)
    plt.close()




def rms_eta(X_t=X_t, E_layer_t=None, **kwargs):
    '''Compute the Shower Width in Eta for each layer of the calorimeter'''
    rms_eta = []
    eta = [a[i][3][:89] for i in range(X_t.shape[0])]
    eta_square = [e*e for e in eta]

    rms_eta_middle_t = [(np.dot(X_t[i][:21], eta_square[i][:21])*E_layer_t[i][2] - np.dot(X_t[i][:21], eta[i][:21])**2)/(E_layer_t[i][2])**2 for i in range(X_t.shape[0])]
    rms_eta_sampler_t = [(np.dot(X_t[i][69:75], eta_square[i][69:75])*E_layer_t[i][0] - np.dot(X_t[i][69:75], eta[i][69:75])**2)/(E_layer_t[i][0])**2 for i in range(X_t.shape[0])]
    rms_eta_strip_t = [(np.dot(X_t[i][21:69], eta_square[i][21:69])*E_layer_t[i][1] - np.dot(X_t[i][21:69],eta[i][21:69])**2)/(E_layer_t[i][1])**2 for i in range(X_t.shape[0])]
    rms_eta_back_t = [(np.dot(X_t[i][75:89], eta_square[i][75:89])*E_layer_t[i][3] - np.dot(X_t[i][75:89], eta[i][75:89])**2)/(E_layer_t[i][2])**2 for i in range(X_t.shape[0])]
    rms_eta.append((rms_eta_sampler_t, rms_eta_strip_t, rms_eta_middle_t, rms_eta_back_t))

    for k in sorted(kwargs):
        rms_eta_middle_g = [(np.dot(kwargs[k][0][i][:21], eta_square[i][:21])*kwargs[k][1][i][2] - np.dot(kwargs[k][0][i][:21], eta[i][:21])**2)/(kwargs[k][1][i][2])**2 for i in range(X_t.shape[0])]
        rms_eta_sampler_g = [(np.dot(kwargs[k][0][i][69:75], eta_square[i][69:75])*kwargs[k][1][i][0] - np.dot(kwargs[k][0][i][69:75], eta[i][69:75])**2)/(kwargs[k][1][i][0])**2 for i in range(X_t.shape[0])]
        rms_eta_strip_g = [(np.dot(kwargs[k][0][i][21:69], eta_square[i][21:69])*kwargs[k][1][i][1] - np.dot(kwargs[k][0][i][21:69],eta[i][21:69])**2)/(kwargs[k][1][i][1])**2 for i in range(X_t.shape[0])]
        rms_eta_back_g = [(np.dot(kwargs[k][0][i][75:89], eta_square[i][75:89])*kwargs[k][1][i][3] - np.dot(kwargs[k][0][i][75:89], eta[i][75:89])**2)/(kwargs[k][1][i][2])**2 for i in range(X_t.shape[0])]
        rms_eta.append((rms_eta_sampler_g, rms_eta_strip_g, rms_eta_middle_g, rms_eta_back_g))

    return rms_eta


def rms_phi(X_t=X_t, E_layer_t=None, **kwargs):
    rms_phi = []
    phi = [a[i][4][:89] for i in range(X_t.shape[0])]
    phi_square = [p*p for p in phi]

    rms_phi_middle_t = [(np.dot(X_t[i][:21], phi_square[i][:21])*E_layer_t[i][2] - np.dot(X_t[i][:21], phi[i][:21])**2)/(E_layer_t[i][2])**2 for i in range(X_t.shape[0])]
    rms_phi_sampler_t = [(np.dot(X_t[i][69:75], phi_square[i][69:75])*E_layer_t[i][0] - np.dot(X_t[i][69:75], phi[i][69:75])**2)/(E_layer_t[i][0])**2 for i in range(X_t.shape[0])]
    rms_phi_strip_t = [(np.dot(X_t[i][21:69], phi_square[i][21:69])*E_layer_t[i][1] - np.dot(X_t[i][21:69],phi[i][21:69])**2)/(E_layer_t[i][1])**2 for i in range(X_t.shape[0])]
    rms_phi_back_t = [(np.dot(X_t[i][75:89], phi_square[i][75:89])*E_layer_t[i][3] - np.dot(X_t[i][75:89], phi[i][75:89])**2)/(E_layer_t[i][2])**2 for i in range(X_t.shape[0])]
    rms_phi.append((rms_phi_sampler_t, rms_phi_strip_t, rms_phi_middle_t, rms_phi_back_t))

    for k in sorted(kwargs):
        rms_phi_middle_g = [(np.dot(kwargs[k][0][i][:21], phi_square[i][:21])*kwargs[k][1][i][2] - np.dot(kwargs[k][0][i][:21], phi[i][:21])**2)/(kwargs[k][1][i][2])**2 for i in range(kwargs[k][0].shape[0])]
        rms_phi_sampler_g = [(np.dot(kwargs[k][0][i][69:75], phi_square[i][69:75])*kwargs[k][1][i][0] - np.dot(kwargs[k][0][i][69:75], phi[i][69:75])**2)/(kwargs[k][1][i][0])**2 for i in range(kwargs[k][0].shape[0])]
        rms_phi_strip_g = [(np.dot(kwargs[k][0][i][21:69], phi_square[i][21:69])*kwargs[k][1][i][1] - np.dot(kwargs[k][0][i][21:69], phi[i][21:69])**2)/(kwargs[k][1][i][1])**2 for i in range(kwargs[k][0].shape[0])]
        rms_phi_back_g = [(np.dot(kwargs[k][0][i][75:89], phi_square[i][75:89])*kwargs[k][1][i][3] - np.dot(kwargs[k][0][i][75:89], phi[i][75:89])**2)/(kwargs[k][1][i][2])**2 for i in range(kwargs[k][0].shape[0])]
        rms_phi.append((rms_phi_sampler_g, rms_phi_strip_g, rms_phi_middle_g, rms_phi_back_g))
    return rms_phi


def plot_rms_eta(rms_eta_t=None, directory=None, **kwargs):
    fig, ax = plt.subplots(2, 2, figsize=(10,10))
    ax[0, 0].hist(rms_eta_t[0], histtype='step', range=(0,0.0001), bins=50, normed=1, color='r', label='Monte-Carlo')
    ax[0, 0].set_title('Shower Width in Eta - SAMPLER', fontsize=16)
    ax[0, 0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    ax[0, 1].hist(rms_eta_t[1], bins=50, color='r',range=(0,0.0002), normed=1,histtype='step', label="Monte-Carlo")
    ax[0, 1].set_title('Shower Width in Eta - STRIP', fontsize=16)
    ax[0, 1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    ax[1, 0].hist(rms_eta_t[2], bins=50, color='r', normed=1,range=(0,0.0002), histtype='step', label="Monte-Carlo")
    ax[1, 0].set_title('Shower Width in Eta - MIDDLE', fontsize=16)
    ax[1, 0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    ax[1, 1].hist(rms_eta_t[3], bins=50, color='r', normed=1,histtype='step',range=(0,0.0000001), label="Monte-Carlo")
    ax[1, 1].set_title('Shower Width in Eta - BACK', fontsize=16)
    ax[1, 1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    for k in sorted(kwargs):

        ax[0, 0].hist(kwargs[k][0], histtype='step', range=(0,0.0001),bins=30, normed=1, label=k, linestyle=':')
        ax[0, 1].hist(kwargs[k][1], bins=30, range=(0,0.0002),normed=1,histtype='step', label=k, linestyle=':')
        ax[1, 0].hist(kwargs[k][2], bins=30, normed=1,range=(0,0.0002), histtype='step', label=k, linestyle=':')
        ax[1, 1].hist(kwargs[k][3], bins=30, normed=1,histtype='step',range=(0,0.0000001), label=k, linestyle=':')

    ax[0, 0].legend()
    ax[0, 1].legend()
    ax[1, 0].legend()
    ax[1, 1].legend()


    plt.savefig(op.join(directory, 'RMS_Eta.eps'), dpi=600)
    plt.savefig(op.join(directory, 'RMS_Eta.pdf'), dpi=600)
    plt.close()


def plot_rms_phi(rms_phi_t=None, directory=None, **kwargs):

    fig, ax = plt.subplots(2, 2, figsize=(15,15))
    ax[0, 0].hist(rms_phi_t[0], histtype='step', bins=50, normed=1, range=(0,0.001), color='r', label='Monte-Carlo')
    ax[0, 0].set_title('Shower Width in Phi - SAMPLER', fontsize=16)
    
    ax[0, 1].hist(rms_phi_t[1], bins=50, color='r', normed=1,histtype='step', range=(0,0.0005), label="Monte-Carlo")
    ax[0, 1].set_title('Shower Width in Phi - STRIP', fontsize=16)
    
    ax[1, 0].hist(rms_phi_t[2], bins=50, color='r', normed=1,histtype='step',range=(0,0.0005), label="Monte-Carlo")
    ax[1, 0].set_title('Shower Width in Phi - MIDDLE', fontsize=16)
    
    ax[1, 1].hist(rms_phi_t[3], bins=50, color='r', normed=1,histtype='step', range=(0,0.00000002),label="Monte-Carlo")
    ax[1, 1].set_title('Shower Width in Phi - BACK', fontsize=16)
    
    for k in sorted(kwargs):
        ax[0, 0].hist(kwargs[k][0], histtype='step', bins=50, normed=1, range=(0,0.001), label=k, linestyle=':')
        ax[0, 1].hist(kwargs[k][1], bins=50, normed=1,histtype='step', range=(0,0.0005),label=k, linestyle=':')
        ax[1, 0].hist(kwargs[k][2], bins=50, normed=1,histtype='step',range=(0,0.0005), label=k, linestyle=':')
        ax[1, 1].hist(kwargs[k][3], bins=50, normed=1,histtype='step', range=(0,0.00000002),label=k, linestyle=':')

    ax[0, 0].legend()
    ax[0, 1].legend()
    ax[1, 0].legend()
    ax[1, 1].legend()

    plt.savefig(op.join(directory, 'RMS_Phi.eps'), dpi=600)
    plt.savefig(op.join(directory, 'RMS_Phi.pdf'), dpi=600)
    plt.close()




def generate_pixels(generators=None, labels=None, nb_events=2000, epoch=None, directory=None):
    '''Generate an average image per layer (mean across the pixels).
       By default, we generate 2000 events.
       Give a dictionnary of generator (key is label, and value is the generator.'''
    X_t, E = unison_shuffled_copies(X_true, E_tot)
    E_random = E[:2000]
    X_t = X_t[:2000]
    X_g = []

    for generator in generators:
        X_gen_norm = generator.predict([np.random.normal(0,1,(nb_events, SIZE_Z)).astype(np.float32), E_random])
        E_random_ = scaler.inverse_transform(E_random) #Apply inverse StandardScaler
        X_g_ = X_gen_norm * np.exp(E_random_.reshape(E_random_.shape[0])[:,np.newaxis]) #Apply the inverse tranform of log
        X_g.append(X_g_)

# compute averaged pixels deposit
    X_t_mean = X_t.mean(axis=0)
    X_g_mean = [X.mean(axis=0) for X in X_g]

# MIDDLE pixels
    X_t_mean_ = X_t_mean[:21][::-1].reshape(3,7).transpose()
    X_g_mean_ = [X[:21][::-1].reshape(3,7).transpose() for X in X_g_mean]
    fig, ax = plt.subplots(1, len(X_g_mean_)+1)


    vmin = min((np.asarray(X_g_mean_).min(), X_t_mean_.min()))
    vmax = max((np.asarray(X_g_mean_).max(), X_t_mean_.max()))
    norm = colors.SymLogNorm(linthresh=0.03, linscale=0.03, vmin=vmin, vmax=vmax)
    ax[0].imshow(X_t_mean_, norm=norm, origin='lower', cmap='viridis')
    ax[0].set_title('Monte-Carlo')
    ax[0].set_yticks([])
    ax[0].set_xticks([])

    for i in range(len(X_g_mean_)):
        im = ax[i+1].imshow(X_g_mean_[i], norm=norm, origin='lower', cmap='viridis')
        ax[i+1].set_title(labels[i])
        ax[i+1].set_yticks([])
        ax[i+1].set_xticks([])

    plt.tight_layout()
    plt.savefig(op.join(directory, 'Middle_EPOCH{}.eps'.format(epoch)), dpi=600)
    plt.savefig(op.join(directory, 'Middle_EPOCH{}.pdf'.format(epoch)), dpi=600)
    plt.close()
        
    # fig.subplots_adjust(right=0.8)
    # cax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(im, cax=cax)
    fig,ax = plt.subplots(figsize=(2,3))
    plt.colorbar(im)
    ax.remove()
    plt.savefig(op.join(directory,'color_MIDDLE.eps'), bbox_inches='tight')
    plt.savefig(op.join(directory,'color_MIDDLE.pdf'), bbox_inches='tight')
    plt.close()

# STRIP pixels
    X_t_mean_ = X_t_mean[21:69][::-1].reshape(24,2).transpose()
    X_g_mean_ = [X[21:69][::-1].reshape(24,2).transpose() for X in X_g_mean]
    fig, ax = plt.subplots(1, len(X_g_mean_)+1)
    vmin = min((np.asarray(X_g_mean_).min(), X_t_mean_.min()))
    vmax = max((np.asarray(X_g_mean_).max(), X_t_mean_.max()))
    norm = colors.SymLogNorm(linthresh=0.03, linscale=0.03, vmin=vmin, vmax=vmax)
    ax[0].imshow(X_t_mean_, norm=norm, origin='lower', cmap='viridis', extent=[0, 24,0, 2], aspect=24)
    ax[0].set_title('Monte-Carlo')
    ax[0].set_yticks([])
    ax[0].set_xticks([])

    for i in range(len(X_g_mean_)):
        im = ax[i+1].imshow(X_g_mean_[i], norm=norm, origin='lower', cmap='viridis', extent=[0, 24,0, 2], aspect=24)
        ax[i+1].set_title(labels[i])
        ax[i+1].set_yticks([])
        ax[i+1].set_xticks([])
    plt.tight_layout()
    plt.savefig(op.join(directory, 'Strip_EPOCH{}.eps'.format(epoch)), dpi=600)
    plt.savefig(op.join(directory, 'Strip_EPOCH{}.pdf'.format(epoch)), dpi=600)
    plt.close()

    fig,ax = plt.subplots(figsize=(2,3))
    plt.colorbar(im)
    ax.remove()
    plt.savefig(op.join(directory,'color_STRIP.eps'), bbox_inches='tight')
    plt.savefig(op.join(directory,'color_STRIP.pdf'), bbox_inches='tight')
    plt.close()

# BACK pixels
    X_t_mean_ = X_t_mean[75:89][::-1].reshape(2,7).transpose()
    X_g_mean_ = [X[75:89][::-1].reshape(2,7).transpose() for X in X_g_mean]
    fig, ax = plt.subplots(1, len(X_g_mean_)+1)
    vmin = min((np.asarray(X_g_mean_).min(), X_t_mean_.min()))
    vmax = max((np.asarray(X_g_mean_).max(), X_t_mean_.max()))
    norm = colors.SymLogNorm(linthresh=0.03, linscale=0.03, vmin=vmin, vmax=vmax)
    ax[0].imshow(X_t_mean_, norm=norm, origin='lower', cmap='viridis', extent=[0, 7,0, 2], aspect=7)
    ax[0].set_title('Monte-Carlo')
    ax[0].set_yticks([])
    ax[0].set_xticks([])

    for i in range(len(X_g_mean_)):
        im = ax[i+1].imshow(X_g_mean_[i], norm=norm, origin='lower', cmap='viridis', extent=[0, 7,0, 2], aspect=7)
        ax[i+1].set_title(labels[i])

        ax[i+1].set_yticks([])
        ax[i+1].set_xticks([])
    plt.tight_layout()
    plt.savefig(op.join(directory, 'Back_EPOCH{}.eps'.format(epoch)), dpi=600)
    plt.savefig(op.join(directory, 'Back_EPOCH{}.pdf'.format(epoch)), dpi=600)
    plt.close()

    fig,ax = plt.subplots(figsize=(2,3))
    plt.colorbar(im)
    ax.remove()
    plt.savefig(op.join(directory,'color_BACK.eps'), bbox_inches='tight')
    plt.savefig(op.join(directory,'color_BACK.pdf'), bbox_inches='tight')
    plt.close()

# PRE-SAMPLER pixels
    X_t_mean_ = X_t_mean[69:75][::-1].reshape(3,2).transpose()
    X_g_mean_ = [X[69:75][::-1].reshape(3,2).transpose() for X in X_g_mean]
    fig, ax = plt.subplots(1, len(X_g_mean_)+1)
    vmin = min((np.asarray(X_g_mean_).min(), X_t_mean_.min()))
    vmax = max((np.asarray(X_g_mean_).max(), X_t_mean_.max()))
    norm = colors.SymLogNorm(linthresh=0.03, linscale=0.03, vmin=vmin, vmax=vmax)
    ax[0].imshow(X_t_mean_, norm=norm, origin='lower', cmap='viridis')
    ax[0].set_title('Monte-Carlo')
    ax[0].set_yticks([])
    ax[0].set_xticks([])

    for i in range(len(X_g_mean_)):
        im = ax[i+1].imshow(X_g_mean_[i], norm=norm, origin='lower', cmap='viridis')
        ax[i+1].set_title(labels[i])
        ax[i+1].set_yticks([])
        ax[i+1].set_xticks([])
    plt.tight_layout()
    plt.savefig(op.join(directory, 'Sampler_EPOCH{}.eps'.format(epoch)), dpi=600)
    plt.savefig(op.join(directory, 'Sampler_EPOCH{}.pdf'.format(epoch)), dpi=600)
    plt.close()

    fig,ax = plt.subplots(figsize=(2,3))
    plt.colorbar(im)
    ax.remove()
    plt.savefig(op.join(directory,'color_SAMPLER.eps'), bbox_inches='tight')
    plt.savefig(op.join(directory,'color_SAMPLER.pdf'), bbox_inches='tight')
    plt.close()



def plot_PCA(directory=None, **kwargs):

    cov = []
    fig, ax = plt.subplots(1, len(kwargs)+1, figsize=(10,10))

    scaler = StandardScaler()
    X_t_ = scaler.fit_transform(X_t)
    pca = PCA()
    pca.fit(X_t_)
    cov.append(pca.get_covariance())
    
    for k in sorted(kwargs):
        
        scaler = StandardScaler()
        value_ = scaler.fit_transform(kwargs[k])
        pca = PCA()
        pca.fit(value_)
        cov.append(pca.get_covariance())

    #Plots
    vmin = -1
    vmax = 1


    ax[0].imshow(cov[0], vmin=vmin, vmax=vmax)
    ax[0].set_title('Monte-Carlo', fontsize=15)

    j=1
    for k in sorted(kwargs):
        ax[j].imshow(cov[j], vmin=vmin, vmax=vmax)
        ax[j].set_title(k, fontsize=15)
        mse = np.mean((cov[0] - cov[j])**2)
        ax[j].set_xlabel('MSE ={} '.format(mse))
        j+=1

    print('WARNING: to compute PCA, I applied noise cuts to generated samples !!!')
    plt.savefig(op.join(directory,'Covariance.eps'), bbox_inches='tight')
    plt.savefig(op.join(directory,'Covariance.pdf'), bbox_inches='tight')
    plt.close()




