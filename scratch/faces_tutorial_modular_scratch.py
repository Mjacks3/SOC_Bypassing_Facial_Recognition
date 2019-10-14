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

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


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
        self.num_channels = num_channels
        self.feat_map_size = feat_map_size
        self.lat_vect_size = lat_vect_size
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
        self.num_channels = num_channels
        self.feat_map_size = feat_map_size
        self.lat_vect_size = lat_vect_size
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
    if not os.path.exists(name):
        os.makedirs(name)

    torch.save(project[0][0], name+"/"+name+"D")
    torch.save(project[0][1], name+"/"+name+"G")
    return 0 

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

    # Create the generator & discriminator
    netG = Generator(ngpu, num_channels=num_channels, feat_map_size=feat_map_size_G, lat_vect_size=latent_vector_size).to(device)
    netD = Discriminator(ngpu, num_channels=num_channels, feat_map_size=feat_map_size_D, lat_vect_size=latent_vector_size).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply the weights_init funum_channelstion to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netG.apply(weights_init)
    netD.apply(weights_init)


    return netG, netD


def train_project(netD, netG, configurations):

    # Root directory for dataset
    dataroot = "data/basic"

    # Number of workers for dataloader
    workers = 2 #should be contoled by cpu resources, not set by user. Lookup to see what the otimal number for this i

    # Batch size during training
    batch_size = 128

    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    image_size = 64

    # Number of training epochs
    num_epochs = 1

    # Learning rate for optimizers
    lr = 0.0002

    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5



    for config in configurations:
        if "batch_size" in config:
            try:
                batch_size = int(config.split(":")[-1])
            except ValueError:
                print("Batch Size: Non Numeric Parameter Entered. Proceeding...")

        elif "image_size" in config:
            try:
                image_size = int(config.split(":")[-1])
            except ValueError:
                print("Image Size: Non Numeric Parameter Entered. Proceeding...")

        elif "num_epochs" in config:
            try:
                num_epochs = int(config.split(":")[-1])
            except ValueError:
                print("Number Epochs Size: Non Numeric Parameter Entered. Proceeding...")

        elif "learning_rate" in config:
            try:
                learning_rate = int(config.split(":")[-1])
            except ValueError:
                print("Learning Rate: Non Numeric Parameter Entered. Proceeding...")
                
        elif "beta" in config:
            try:
                beta1 = int( config.split(":")[-1])
            except ValueError:
                print("Beta: Non Numeric Parameter Entered. Proceeding...")

        elif "source" in config:
            if os.path.exists(config.split(":")[-1]): 
                dataroot = config.split(":")[-1]
            else:
                print("Data Source Not Found.")
        

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator

    fixed_noise = torch.randn(64,netD.lat_vect_size, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))



    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0


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



    print("Starting Training Loop...")
    
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, netG.lat_vect_size, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()
            
            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            
            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            
            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                
            iters += 1



    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.show()
    

    return 0


if __name__ == "__main__":
    print("Start Yeet")

    parser = argparse.ArgumentParser(description='Service Oriented Computing: GANS-Based Facial Recognition Project Interface')
    parser.add_argument('-l', '--load', action='store', help='Load existing GAN. Omit to work on new instance') 
    parser.add_argument('-c', '--config', action='store',nargs="*", help='Initialize new GAN and configure HyperParameters. Possible arguments are [~]') #Use nargs = '*' for multiple arguments
    parser.add_argument('-t', '--train', action='store', nargs="*", help='Train Model on provided dataset and Display Training Results')
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
        project.append(instantiate(configurations))

    if args.train: #Users can set specific configurations or leave blank to default
        configurations = []
        if args.train != None:
            configurations = args.train
        print(args.train)

        train_project(project[0][1],project[0][0], configurations)

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
    save_project(name,project[0][1],project[0][0])

    print("End Yeet")
