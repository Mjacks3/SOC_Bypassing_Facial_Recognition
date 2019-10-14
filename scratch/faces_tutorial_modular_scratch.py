#**Original Author**: `Nathan Inkawhich <https://github.com/inkawhich>`__
#**Additions by Malik , Nahum, Anupama


#Though this service was designed as a TaaS (Tool as a Service), we shall develop a few built in services to utilize with this.
#As soon as we think of some...


from __future__ import print_function
#%matplotlib inline
import argparse
import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

manualSeed = random.randint(1, 10000) 

print("Random Seed: ", manualSeed)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Generator Code

class Generator(nn.Module):

    def __init__(self, ngpu, num_channels=3, feat_map_size=64,lat_vect_size=100):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( lat_vect_size, feat_map_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feat_map_size * 8),
            nn.ReLU(True),
            # state size. (feat_map_size*8) x 4 x 4
            nn.ConvTranspose2d(feat_map_size * 8, feat_map_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map_size * 4),
            nn.ReLU(True),
            # state size. (feat_map_size*4) x 8 x 8
            nn.ConvTranspose2d( feat_map_size * 4, feat_map_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map_size * 2),
            nn.ReLU(True),
            # state size. (feat_map_size*2) x 16 x 16
            nn.ConvTranspose2d( feat_map_size * 2, feat_map_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map_size),
            nn.ReLU(True),
            # state size. (feat_map_size) x 32 x 32
            nn.ConvTranspose2d( feat_map_size, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (num_channels) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, ngpu, num_channels=3, feat_map_size=64,lat_vect_size=100):
        
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (num_channels) x 64 x 64
            nn.Conv2d(num_channels, feat_map_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (feat_map_size) x 32 x 32
            nn.Conv2d(feat_map_size, feat_map_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (feat_map_size*2) x 16 x 16
            nn.Conv2d(feat_map_size * 2, feat_map_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (feat_map_size*4) x 8 x 8
            nn.Conv2d(feat_map_size * 4, feat_map_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (feat_map_size*8) x 4 x 4
            nn.Conv2d(feat_map_size * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

def instantiate(configurations):

    #Defaults
    latent_vector_size = 100
    num_channels = 3
    feat_map_size_D = 64
    feat_map_size_G = 64
    
    for config in configurations:
        if "latent_vector_size" in config:
            try:
                latent_vector_size = int( config.split(":") [-1])
            except ValueError:
                print("Latent Vector Size: Non Numeric Parameter Entered. Proceeding...")

        elif "num_channels" in config:
            try:
                num_channels = int( config.split(":") [-1])
            except ValueError:
                print("Num Channels: Non Numeric Parameter Entered. Proceeding...")

        elif "feat_map_size_D" in config:
            try:
                feat_map_size_D = int( config.split(":") [-1])
            except ValueError:
                print("Feature Map (Discriminator) Size: Non Numeric Parameter Entered. Proceeding...")

        elif "feat_map_size_G" in config:
            try:
                feat_map_size_G = int( config.split(":") [-1])
            except ValueError:
                print("Feature Map (Generator) Size: Non Numeric Parameter Entered. Proceeding...")

    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # Create the generator
    #num channels, feat vec, la vec
    netG = Generator(ngpu, num_channels, feat_map_size_G, latent_vector_size).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the weights_init funum_channelstion to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netG.apply(weights_init)

    # Create the Discriminator
    netD = Discriminator(ngpu, num_channels, feat_map_size_D, latent_vector_size).to(device)
    return netG, netD

def load_project(name):
    if os.path.exists(name):
        print("Project Bones Found")
        d = torch.load(name+"/"+name+"D")
        g =  torch.load(name+"/"+name+"G")
        return d,g
    else:
        print("No Project Found")
        return 0


def save_project(name, discriminator,generator):
    print("Saving Project as " + name + "...")
    os.makedirs(name)
    torch.save(project[0][0], name+"/"+name+"D")
    torch.save(project[0][1], name+"/"+name+"G")
    return 0 


if __name__ == "__main__":
    print("Start Yeet")

    parser = argparse.ArgumentParser(description='Service Oriented Computing: GANS-Based Facial Recognition Project Interface')
    parser.add_argument('-l', '--load', action='store', help='Load existing GAN. Omit to work on new instanum_channelse') 
    parser.add_argument('-c', '--config', action='store',nargs="*", help='Initialize new GAN and configure HyperParameters. Possible arguments are [~]') #Use nargs = '*' for multiple arguments
    parser.add_argument('-t', '--train', action='store', help='Train Model on provided dataset with possible ')
    parser.add_argument('-r', '--results', action='store_true', help='Display Graphical and Visual Training Results. ')
    parser.add_argument('-v', '--validate', action='store', help='Validate Single Image (Real or Fake)')
    parser.add_argument('-o', '--output_name', action='store', help='Name to save model as. Omit to use existing name or randomized name. ')
    args = parser.parse_args()

    print("SOC Facial Recognition Project")

    project = []
    name = str(manualSeed)

    if args.load: #If no load flag, initialize a new D&G
        print("Loading Data for Project...")
        name = args.load
        project.append(load_project(name))
        if project[0] == 0:
            print("Load Failed. Exiting")
            sys.exit(0)

    else: #args.config: #Users can set specific configurations or leave blank to default all values
        print("Initializing New Project with Given or Default Configuratons")
        configurations = []
        if args.config != None:
            configurations = args.config
        print(args.config)
        project.append(instantiate(configurations))

    print(project)

    print("DEV BREAK")
    sys.exit(0)

    if args.train: #Users can set specific configurations or leave blank to default
        print("Training Project on Given Dataset")
        print(args.train)
        print("Stump Train")

    if args.results: # No arguments needed for flag. On blank, do not call funum_channelstion. 
        print("Display Results on Current Project")
        print(args.results)
        print("Stump Results")

    if args.validate: # One "Service Validate a face to verify if....... (?)  On blank, do not call funum_channelstion. 
        print("Display Results on Current Project")
        print(args.validate)
        print("Stump Validate")




    if args.output_name: # Set output name when saving. On blank, use default name for project saving and loading 
        #TODO: Need to check if has legal characters
        if not os.path.exists(args.output_name):
            name = args.output_name
        else: 
            print("Project Name Already Exists")


    #Always save project at end of run
    save_project(name,project[0][0],project[0][1])

    print("End Yeet")
