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

#TODO: make configureable to instantiate
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64




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
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

def configure_and_instantiate():

    # Set random seed for reproducibility
    #manualSeed = 999
    manualSeed = random.randint(1, 10000) # use if you want new results
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)


    #hyperparamters  + Training ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Root directory for dataset
    dataroot = "data/basic"

    # Number of workers for dataloader
    workers = 2

    # Batch size during training
    batch_size = 128

    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    image_size = 64

    # Number of channels in the training images. For color images this is 3
    nc = 3

    # Size of z latent vector (i.e. size of generator input)
    nz = 100

    # Size of feature maps in generator
    ngf = 64

    # Size of feature maps in discriminator
    ndf = 64

    # Number of training epochs
    num_epochs = 1

    # Learning rate for optimizers
    lr = 0.0002

    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5

    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1


    # We can use an image folder dataset the way we have it setup.
    # Create the dataset
    dataset = dset.ImageFolder(root=dataroot,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # Create the generator
    netG = Generator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netG.apply(weights_init)

    # Print the model
    print(netG)


    # Discriminator Code



    ######################################################################
    # Now, as with the generator, we can create the discriminator, apply the
    # ``weights_init`` function, and print the model’s structure.
    # 
    #hyperparamters  + Training ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Create the Discriminator
    netD = Discriminator(ngpu).to(device)

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


def instantiate():
    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # Create the generator
    netG = Generator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netG.apply(weights_init)

    # Create the Discriminator
    netD = Discriminator(ngpu).to(device)
    return netG, netD

def save_project(name, discriminator,generator):
    if os.path.exists(name):
        print("Saving Project as " + name + "...")
        torch.save(project[0][0], name+"/"+name+"D")
        torch.save(project[0][1], name+"/"+name+"G")
    else:
        print("Something Went Wrong When Saving")
    return 0 




if __name__ == "__main__":
    print("Start Yeet")

    parser = argparse.ArgumentParser(description='Service Oriented Computing: GANS-Based Facial Recognition Project Interface')
    parser.add_argument('-l', '--load', action='store', help='Load existing GAN. Omit to work on new instance') 
    parser.add_argument('-c', '--config', action='store', help='Initialize new GAN and configure HyperParameters. Possible arguments are [~]') #Use nargs = '*' for multiple arguments
    parser.add_argument('-t', '--train', action='store', help='Train Model on provided dataset with possible ')
    parser.add_argument('-r', '--results', action='store_true', help='Display Graphical and Visual Training Results. ')
    parser.add_argument('-v', '--validate', action='store', help='Validate Single Image (Real or Fake)')
    parser.add_argument('-o', '--output_name', action='store', help='Name to save model as. Omit to use existing name or randomized name. ')
    args = parser.parse_args()

    print("SOC Facial Recognition Project")

    project = []
    name =  "" 

    if args.load: #If no load flag, initialize a new D&G
        print("Loading Data for Project: ")
        name = args.load
        project.append(load_project(name))
        if project[0] == 0:
            print("Load Failed. Exiting")
            sys.exit(0)

    else:
        name = str(manualSeed)
        if not os.path.exists(name):
            print("STUMP MAKING NEW PROJECT_DIRECTORY")
            os.makedirs(name)
        project.append(instantiate())


    print(project)



    if args.config: #Users can set specific configurations or leave blank to default
        print("Configuring Project with given Configuratons")
        print(args.config)
        print("Stump Configure")

    if args.train: #Users can set specific configurations or leave blank to default
        print("Training Project on Given Dataset")
        print(args.train)
        print("Stump Train")

    if args.results: # No arguments needed for flag. On blank, do not call function. 
        print("Display Results on Current Project")
        print(args.results)
        print("Stump Results")

    if args.validate: # One "Service Validate a face to verify if....... (?)  On blank, do not call function. 
        print("Display Results on Current Project")
        print(args.validate)
        print("Stump Validate")


        

    if args.output_name: # Set output name when saving. On blank, use default name for project saving and 
        #TODO: Need to check if has legal characters
        if not os.path.exists(args.output_name):
            name = args.output_name
            os.makedirs(name)

        else: 
            print("Project Name Already Exists")



    save_project(name,project[0][0],project[0][1])
    print("End Yeet")


"""
torch.save(netD.state_dict(), "test2")

model = torch.load("test")
model.eval()
"""