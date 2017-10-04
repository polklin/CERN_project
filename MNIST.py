# coding: utf-8

import os
import os.path as op
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers.merge import _Merge
from keras.layers.convolutional import Convolution2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.datasets import mnist
from keras import backend as K
from functools import partial
import matplotlib.pyplot as plt
from IPython import embed


BATCH_SIZE = 64
TRAINING_RATIO = 5  
GRADIENT_PENALTY_WEIGHT = 10  


def wasserstein_loss(y_true, y_pred): 
    return K.mean(y_true * y_pred) 



def gradient_penalty_loss(y_true, y_pred, averaged_samples): 
    gradients = K.gradients(y_pred, averaged_samples)
    gradients = K.concatenate([K.flatten(tensor) for tensor in gradients])
    gradient_l2_norm = K.sqrt(K.sum(K.square(gradients)))
    gradient_penalty = GRADIENT_PENALTY_WEIGHT * K.square(1 - gradient_l2_norm)
    return gradient_penalty

def make_generator():
    model = Sequential()
    model.add(Dense(1024, input_dim=100))
    model.add(LeakyReLU())
    model.add(Dense(128 * 7 * 7))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    if K.image_data_format() == 'channels_first':
        model.add(Reshape((128, 7, 7), input_shape=(128 * 7 * 7,)))
    else:
        model.add(Reshape((7, 7, 128), input_shape=(128 * 7 * 7,)))
    model.add(Conv2DTranspose(128, (5, 5), strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Convolution2D(64, (5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Conv2DTranspose(64, (5, 5), strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Convolution2D(1, (5, 5), padding='same', activation='tanh'))
    return model


def make_discriminator():
    model = Sequential()
    if K.image_data_format() == 'channels_first':
        model.add(Convolution2D(64, (5, 5), padding='same', input_shape=(1, 28, 28)))
    else:
        model.add(Convolution2D(64, (5, 5), padding='same', input_shape=(28, 28, 1)))
    model.add(LeakyReLU())
    model.add(Convolution2D(128, (5, 5), kernel_initializer='he_normal', strides=[2, 2]))
    model.add(LeakyReLU())
    model.add(Convolution2D(128, (5, 5), kernel_initializer='he_normal', padding='same', strides=[2, 2]))
    model.add(LeakyReLU())
    model.add(Flatten())
    model.add(Dense(1024, kernel_initializer='he_normal'))
    model.add(LeakyReLU())
    model.add(Dense(1, kernel_initializer='he_normal'))
    return model

    
class RandomWeightedAverage(_Merge):
    """Takes a randomly-weighted average of two tensors. In geometric terms, this outputs a random point on the line
    between each pair of input points.

    Inheriting from _Merge is a little messy but it was the quickest solution I could think of.
    Improvements appreciated."""

    def _merge_function(self, inputs):
        weights = K.random_uniform((BATCH_SIZE, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])
    
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_train = X_train[:10000]
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))
X_test = (X_test.astype(np.float32) - 127.5) / 127.5

generator = make_generator()
discriminator = make_discriminator()

for layer in discriminator.layers:
    layer.trainable = False
discriminator.trainable = False
generator_input = Input(shape=(100,))
generator_layers = generator(generator_input)
discriminator_layers_for_generator = discriminator(generator_layers)
generator_model = Model(inputs=[generator_input], outputs=[discriminator_layers_for_generator])
generator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9), loss=wasserstein_loss)


for layer in discriminator.layers:
    layer.trainable = True
for layer in generator.layers:
    layer.trainable = False
discriminator.trainable = True
generator.trainable = False  

# The discriminator_model is more complex. It takes both real image samples and random noise seeds as input.
# The noise seed is run through the generator model to get generated images. Both real and generated images
# are then run through the discriminator. Although we could concatenate the real and generated images into a
# single tensor, we don't (see model compilation for why).
real_samples = Input(shape=X_train.shape[1:])
generator_input_for_discriminator = Input(shape=(100,))
generated_samples_for_discriminator = generator(generator_input_for_discriminator)
discriminator_output_from_generator = discriminator(generated_samples_for_discriminator)
discriminator_output_from_real_samples = discriminator(real_samples)

# We also need to generate weighted-averages of real and generated samples, to use for the gradient norm penalty.
averaged_samples = RandomWeightedAverage()([real_samples, generated_samples_for_discriminator])
# We then run these samples through the discriminator as well. Note that we never really use the discriminator
# output for these samples - we're only running them to get the gradient norm for the gradient penalty loss.
averaged_samples_out = discriminator(averaged_samples)

# The gradient penalty loss function requires a list of trainable weights to get gradients on. However,
# Keras loss functions can only have two arguments, y_true and y_pred. We get around this by making a partial()
# of the function with the discriminator's trainable weights here.
partial_gp_loss = partial(gradient_penalty_loss, averaged_samples=averaged_samples)
partial_gp_loss.__name__ = 'gradient_penalty'  # Functions need names or Keras will throw an error

# Keras requires that inputs and outputs have the same number of samples. This is why we didn't concatenate the
# real samples and generated samples before passing them to the discriminator: If we had, it would create an
# output with 2 * BATCH_SIZE samples, while the output of the "averaged" samples for gradient penalty
# would have only BATCH_SIZE samples.

# If we don't concatenate the real and generated samples, however, we get three outputs: One of the generated
# samples, one of the real samples, and one of the averaged samples, all of size BATCH_SIZE. This works neatly!
discriminator_model = Model(inputs=[real_samples, generator_input_for_discriminator],
                            outputs=[discriminator_output_from_real_samples,
                                     discriminator_output_from_generator,
                                     averaged_samples_out])

# We use the Adam paramaters from Gulrajani et al. We use the Wasserstein loss for both the real and generated
# samples, and the gradient penalty loss for the averaged samples.
discriminator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
                            loss=[wasserstein_loss,
                                  wasserstein_loss,
                                  partial_gp_loss])

positive_y = np.ones((BATCH_SIZE, 1), dtype=np.float32)
negative_y = -positive_y
dummy_y = np.zeros((BATCH_SIZE, 1), dtype=np.float32)

positive_y_test = np.ones((1024, 1), dtype=np.float32)
negative_y_test = -positive_y_test
dummy_y_test = np.zeros((1024, 1), dtype=np.float32)

# loss_tot = []
# loss_real = []
# loss_gen = []
# loss_penalty = []

nb_epochs = 5

for epoch in range(nb_epochs):
    np.random.shuffle(X_train)
    print("Epoch: ", epoch)
    print("Number of batches: ", int(X_train.shape[0] // BATCH_SIZE))
    discriminator_loss = []
    generator_loss = []
    minibatches_size = BATCH_SIZE * TRAINING_RATIO
    for i in range(int(X_train.shape[0] // (BATCH_SIZE * TRAINING_RATIO))):
        discriminator_minibatches = X_train[i * minibatches_size:(i + 1) * minibatches_size]
        for j in range(TRAINING_RATIO):
            image_batch = discriminator_minibatches[j * BATCH_SIZE:(j + 1) * BATCH_SIZE]
            noise = np.random.rand(BATCH_SIZE, 100).astype(np.float32)
            discriminator_loss.append(discriminator_model.train_on_batch([image_batch, noise],
                                                                          [positive_y, negative_y, dummy_y]))
            
            
        generator_loss.append(generator_model.train_on_batch(np.random.rand(BATCH_SIZE, 100), positive_y))

    directory = 'Epoch_{}'.format(epoch+1)
    if not op.exists(directory):
        os.makedirs(directory)

    # generates 10 random images
    generated_images = generator.predict(np.random.rand(10, 100))

    fig, axs = plt.subplots(2,5, figsize=(15, 6), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .5, wspace=.001)
    axs = axs.ravel()
    for i in range(10):
        axs[i].imshow(generated_images[i].reshape(28,28),cmap='gray')
        axs[i].set_title('Image '+str(i))
        axs[i].set_xticks([])
        axs[i].set_yticks([])

    plt.savefig(op.join(directory, 'Images_epoch{}.eps'.format(epoch+1)), dpi=600)
    plt.savefig(op.join(directory, 'Images_epoch{}.pdf'.format(epoch+1)), dpi=600)
    plt.close()



#    print("\nLoss_tot after epoch {}:....... ".format(epoch), loss_tot[-1])
#    print("Loss_real after epoch {}:...... ".format(epoch), loss_real[-1])
#    print("Loss_gen after epoch {}:....... ".format(epoch), loss_gen[-1])
#    print("Loss_penalty after epoch {}:..... ".format(epoch), loss_penalty[-1])
    discriminator.save_weights(op.join(directory, 'discriminator_weights.h5'))
    generator.save_weights(op.join(directory, 'generator_weights.h5'))
    discriminator_model.save_weights(op.join(directory, 'discriminator_model_weights.h5'))
    generator_model.save_weights(op.join(directory, 'generator_model_weights.h5'))