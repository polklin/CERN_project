# CERN_project
GANs for simulation of electromagnetic showers in the ATLAS calorimeter 

1) MNIST.py
First implementation of Improved Wasserstein GAN (WGAN-gp), on the MNIST dataset.
See the article : https://arxiv.org/abs/1704.00028

2) WGAN for electromagnetic showers in the ATLAS calorimeter
3 Python files work together :
- config.py (contains training parameters, path to files, ...)
- plot_functions.py (where all plot functions are defined)
- training.py (main file to train the WGAN)

To launch a training, you should write in the terminal : python training.py Name (Name = name of your folder in which all plots and weights will be saved)

Plots are generated automatically each 250 epochs, a folder is created each time.
