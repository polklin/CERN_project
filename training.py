# coding: utf-8

import sys
import pickle
import os
import os.path as op
import numpy as np
import time 
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Flatten, concatenate
from keras.layers.merge import _Merge
from keras.layers.convolutional import Convolution2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.regularizers import l1
from keras.datasets import mnist
from keras import backend as K
from functools import partial
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors

from IPython import embed

from plot_functions_ import (generate_pixels,
                            unison_shuffled_copies,
                            energy_per_layer,
                            energy_per_event,
                            energy_per_layer_normed,
                            plot_energy_per_layer,
                            plot_energy_per_layer_normed,
                            avg_eta, 
                            plot_avg_eta,
                            avg_phi,
                            plot_avg_phi,
                            energy_centre_middle,
                            plot_centre_middle,
                            rms_eta,
                            plot_rms_eta,
                            rms_phi,
                            plot_rms_phi,
                            plot_PCA,
                            )

from config import (BATCH_SIZE,
                    TRAINING_RATIO,
                    GRADIENT_PENALTY_WEIGHT,
                    n_cells,
                    SIZE_Z,
                    nb_epochs,
                    data_path,
                    X,
                    E_tot,
                    scaler,
                    )

'''An implementation of the improved WGAN described in https://arxiv.org/abs/1704.00028

The improved WGAN has a term in the loss function which penalizes the network if its gradient
norm moves away from 1. This is included because the Earth Mover (EM) distance used in WGANs is only easy
to calculate for 1-Lipschitz functions (i.e. functions where the gradient norm has a constant upper bound of 1).

The original WGAN paper enforced this by clipping weights to very small values [-0.01, 0.01]. However, this
drastically reduced network capacity. Penalizing the gradient norm is more natural, but this requires
second-order gradients.'''


time_start = time.clock()
seed = sys.argv[1] 



def wasserstein_loss(y_true, y_pred):
    '''The Wasserstein loss function is very simple to calculate. In a standard GAN, the discriminator
    has a sigmoid output, representing the probability that samples are real or generated. In Wasserstein
    GANs, however, the output is linear with no activation function! Instead of being constrained to [0, 1],
    the discriminator wants to make the distance between its output for real and generated samples as large as possible.

    The most natural way to achieve this is to label generated samples -1 and real samples 1, instead of the
    0 and 1 used in normal GANs, so that multiplying the outputs by the labels will give you the loss immediately.

    Note that the nature of this loss means that it can be (and frequently will be) less than 0.'''
    return K.mean(y_true * y_pred) 



def gradient_penalty_loss(y_true, y_pred, averaged_samples):
    """Calculates the gradient penalty loss for a batch of "averaged" samples.

    In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term to the loss function
    that penalizes the network if the gradient norm moves away from 1. However, it is impossible to evaluate
    this function at all points in the input space. The compromise used in the paper is to choose random points
    on the lines between real and generated samples, and check the gradients at these points. Note that it is the
    gradient w.r.t. the input averaged samples, not the weights of the discriminator, that we're penalizing!

    In order to evaluate the gradients, we must first run samples through the generator and evaluate the loss.
    Then we get the gradients of the discriminator w.r.t. the input averaged samples.

    Note that this loss function requires the original averaged samples as input, but Keras only supports passing
    y_true and y_pred to loss functions. To get around this, we make a partial() of the function with the
    averaged_samples argument, and use that for model training."""
    if K.backend()=='theano':
        J, _ = theano.scan(lambda i, y, x: T.square(T.grad(y[i].sum(), x[i])).sum() ** 0.5,
                           sequences=T.arange(y_pred.shape[0]),
                           non_sequences=[y_pred, averaged_samples])
        return GRADIENT_PENALTY_WEIGHT * K.square(1 - J)
    
    else:
        gradients = K.gradients(y_pred, averaged_samples)
        gradients = K.concatenate([K.flatten(tensor) for tensor in gradients])
        gradient_l2_norm = K.sqrt(K.sum(K.square(gradients)))
        gradient_penalty = GRADIENT_PENALTY_WEIGHT * K.square(1 - gradient_l2_norm)
        return gradient_penalty



def make_generator():

    g_input_noise = Input(shape=(SIZE_Z,), name='Input_noise')
    g_input_energy = Input(shape=(1,), name='Input_Energy')
    g_input = concatenate([g_input_noise, g_input_energy], axis=-1)
    x = Dense(128)(g_input)
    x = LeakyReLU()(x)
    x = Dense(128)(x)
    x = LeakyReLU()(x)
    x = Dense(128)(x)
    x = LeakyReLU()(x)
    outputs = Dense(n_cells, activation='softmax', activity_regularizer=l1(1e-5))(x) #L1 regularization term has been added
    G = Model(inputs=[g_input_noise, g_input_energy], outputs=outputs)

    return G



def make_discriminator():
    
    d_input_cells = Input(shape=(89,), name='Input_cells')
    d_input_energy = Input(shape=(1,), name='Input_energy')
    d_input = concatenate([d_input_cells, d_input_energy], axis=-1)
    x = Dense(128)(d_input)
    x = LeakyReLU()(x)
    x = Dense(128)(x)
    x = LeakyReLU()(x)
    x = Dense(128)(x)
    x = LeakyReLU()(x)
    predictions = Dense(1, activation='linear')(x)   
    D =  Model(inputs=[d_input_cells, d_input_energy], outputs=predictions)
    return D


class RandomWeightedAverage(_Merge):
    '''Takes a randomly-weighted average of two tensors. In geometric terms, this outputs a random point on the line
    between each pair of input points.

    Inheriting from _Merge is a little messy but it was the quickest solution I could think of.'''

    def _merge_function(self, inputs):
        weights = K.random_uniform((BATCH_SIZE, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])


print("Loading data...")


X_train = X[:35000]
X_test = X[35000:]

print "- Train data =", X_train.shape
print "- Test data =", X_test.shape

E_train = E_tot[:35000]
E_test = E_tot[35000:]



generator = make_generator()
discriminator = make_discriminator()

for layer in discriminator.layers:
    layer.trainable = False
discriminator.trainable = False
generator_input = Input(shape=(100,))
generator_energy = Input(shape=(1,))
generator_layers = generator([generator_input, generator_energy])
discriminator_layers_for_generator = discriminator([generator_layers, generator_energy]) 

generator_model = Model(inputs=[generator_input, generator_energy], outputs=discriminator_layers_for_generator)
generator_model.compile(optimizer=Adam(0.00005, beta_1=0.5, beta_2=0.5), loss=wasserstein_loss)


for layer in discriminator.layers:
    layer.trainable = True
for layer in generator.layers:
    layer.trainable = False
discriminator.trainable = True
generator.trainable = False


real_samples = Input(shape=(n_cells,))
energy_real = Input(shape=(1,)) 
generator_input_for_discriminator = Input(shape=(100,))
generated_samples_for_discriminator = generator([generator_input_for_discriminator, energy_real])
discriminator_output_from_generator = discriminator([generated_samples_for_discriminator, energy_real])
discriminator_output_from_real_samples = discriminator([real_samples, energy_real])
averaged_samples = RandomWeightedAverage()([real_samples, generated_samples_for_discriminator])

averaged_samples_out = discriminator([averaged_samples, energy_real])


partial_gp_loss = partial(gradient_penalty_loss, averaged_samples=averaged_samples)
partial_gp_loss.__name__ = 'gradient_penalty' 

discriminator_model = Model(inputs=[real_samples, generator_input_for_discriminator, energy_real],
                            outputs=[discriminator_output_from_real_samples,
                                     discriminator_output_from_generator,
                                     averaged_samples_out])


discriminator_model.compile(optimizer=Adam(0.00005, beta_1=0.5, beta_2=0.5),
                            loss=[wasserstein_loss,
                                  wasserstein_loss,
                                  partial_gp_loss])


positive_y = np.ones((BATCH_SIZE, 1), dtype=np.float32)
negative_y = -positive_y
dummy_y = np.zeros((BATCH_SIZE, 1), dtype=np.float32)


positive_y_test = np.ones((1024, 1), dtype=np.float32)
negative_y_test = -positive_y_test
dummy_y_test = np.zeros((1024, 1), dtype=np.float32)

loss_tot = []
loss_real = []
loss_gen = []
loss_penalty = []




for epoch in range(nb_epochs):
    X_train, E_train = unison_shuffled_copies(X_train, E_train)
    discriminator_loss = []
    generator_loss = []
    minibatches_size = BATCH_SIZE * TRAINING_RATIO

    for i in range(int(X_train.shape[0] // (BATCH_SIZE * TRAINING_RATIO))):
        discriminator_minibatches = X_train[i * minibatches_size:(i + 1) * minibatches_size]
        E_minibatches = E_train[i * minibatches_size:(i + 1) * minibatches_size]

        for j in range(TRAINING_RATIO):
            image_batch = discriminator_minibatches[j * BATCH_SIZE:(j + 1) * BATCH_SIZE]
            energy_batch = E_minibatches[j * BATCH_SIZE:(j + 1) * BATCH_SIZE]
            noise = np.random.normal(0,1,(BATCH_SIZE, SIZE_Z)).astype(np.float32)
            discriminator_loss.append(discriminator_model.train_on_batch([image_batch, noise, energy_batch],
                                                                         [positive_y, negative_y, dummy_y]))

        energy_generator= np.random.choice(E_train.reshape(len(E_train)), BATCH_SIZE, replace=False).reshape(BATCH_SIZE,1)
        generator_loss.append(generator_model.train_on_batch([np.random.normal(0,1,(BATCH_SIZE, SIZE_Z)).astype(np.float32), 
                                                              energy_generator],
                                                              positive_y))

    if epoch!=0 and epoch%250==0:
        X_test, E_test = unison_shuffled_copies(X_test, E_test)
        X_test = X_test[:1024]
        E_test = E_test[:1024]
        noise_test = np.random.normal(0,1,(1024, SIZE_Z)).astype(np.float32)
        (tot, real, gen, penalty) = discriminator_model.evaluate([X_test, noise_test, E_test],
                                    [positive_y_test, negative_y_test, dummy_y_test],
                                     batch_size=BATCH_SIZE)
        
        loss_tot.append(tot)
        loss_real.append(real)
        loss_gen.append(gen)
        loss_penalty.append(penalty)

        #### SAVINGS ####
        directory = '{}/Epoch_{}'.format(seed,epoch)
        if not op.exists(directory):
            os.makedirs(directory)

        # save weights
        discriminator_model.save_weights(op.join(directory, 'discriminator_model_weights_EPOCH{}.h5'.format(epoch)))
        generator_model.save_weights(op.join(directory, 'generator_model_weights_EPOCH{}.h5'.format(epoch)))
        generator.save_weights(op.join(directory, 'generator_weights_EPOCH{}.h5'.format(epoch)))
        discriminator.save_weights(op.join(directory, 'discriminator_weights_EPOCH{}.h5'.format(epoch)))
        print "\nles poids entraînés ont été enregistrés avec succès !"

        # save distributions
        X_gen_norm = generator.predict([np.random.normal(0,1,(X.shape[0], SIZE_Z)).astype(np.float32), E_tot])
        E_ = scaler.inverse_transform(E_tot) #Apply inverse StandardScaler
        X_g = X_gen_norm * np.exp(E_.reshape(E_.shape[0])[:,np.newaxis]) 

        # Electronic threshold for generated data
        for i in range(X_g[0].shape[0]):
          for j in range(89):
              if (j<21) & (X_g[i][j] < 200):
                  X_g[i][j] = 0
              elif (21<=j<69) & (X_g[i][j] < 60):
                  X_g[i][j] = 0
              elif (69<=j<75) & (X_g[i][j] < 120):
                  X_g[i][j] = 0
              elif (75<=j<89) & (X_g[i][j] < 200):
                  X_g[i][j] = 0

        energy_tot = energy_per_event(WGAN=X_g)
        energy_layer = energy_per_layer(WGAN=X_g)
        energy_layer_normed = energy_per_layer_normed(E_layer_t=energy_layer[0], E_tot_t=energy_tot[0],\
                                                      WGAN=(energy_layer[1], energy_tot[1]))
        avg_eta_ = avg_eta(E_layer_t=energy_layer[0], WGAN=(X_g, energy_layer[1]))
        avg_phi_ = avg_phi(E_layer_t=energy_layer[0], WGAN=(X_g, energy_layer[1]))
        centre_middle = energy_centre_middle(E_layer_t=energy_layer[0], WGAN=(X_g, energy_layer[1]))
        rms_eta_ = rms_eta(E_layer_t=energy_layer[0], WGAN=(X_g, energy_layer[1]))
        rms_phi_ = rms_phi(E_layer_t=energy_layer[0], WGAN=(X_g, energy_layer[1]))


        generate_pixels([generator], labels=['WGAN'], epoch=epoch, directory=directory)
        plot_energy_per_layer(E_layer_t=energy_layer[0], epoch=epoch, directory=directory, WGAN=energy_layer[1])
        plot_energy_per_layer_normed(E_layer_norm_t=energy_layer_normed[0], epoch=epoch, directory=directory, WGAN=energy_layer_normed[1])
        plot_centre_middle(E_middle_centre_t=centre_middle[0], directory=directory, WGAN=centre_middle[1])
        plot_avg_eta(avg_eta_t=avg_eta_[0], directory=directory, WGAN=avg_eta_[1])
        plot_avg_phi(avg_phi_t=avg_phi_[0], directory=directory, WGAN=avg_phi_[1])
        plot_rms_eta(rms_eta_t=rms_eta_[0], directory=directory, WGAN=rms_eta_[1])
        plot_rms_phi(rms_phi_t=rms_phi_[0], directory=directory, WGAN=rms_phi_[1])
        plot_PCA(directory, WGAN=X_g)

        print "All the plots have been generated !"


        if len(loss_tot) != 0:
            print "\nLoss_tot after epoch {}:....... ".format(epoch), loss_tot[-1]
            print "Loss_real after epoch {}:...... ".format(epoch), loss_real[-1]
            print "Loss_gen after epoch {}:....... ".format(epoch), loss_gen[-1]
            print "Loss_penalty after epoch {}:..... ".format(epoch), loss_penalty[-1]
            time_elapsed = (time.clock() - time_start)
            print "Time elapsed since beginning:{} sec".format(time_elapsed)


