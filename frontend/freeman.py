#**Original Author**: `Nathan Inkawhich <https://github.com/inkawhich>`__
#**Adaptation by Malik, Nahum, Anupama


#Though this service was designed as a TaaS (Tool as a Service), we shall  a  built in service to utilize with this.
#As soon as we think of one...


from __future__ import print_function
#%matplotlib inline
import argparse
import os
import shutil
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


device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
device = "cpu"
#device = "cuda"

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


def load_project(dire, name):
        print("Project Bones Found\n")
        d = torch.load(dire+"/"+name+"D")
        g =  torch.load(dire+"/"+name+"G")
        return d,g



def save_project(dire, name, discriminator,generator):
    print("Saving Project as " + name + "... \n")

    torch.save(discriminator, dire+"/"+name+"D")
    torch.save(generator,dire+"/"+ name+"G")
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

def train_project_only_negatives(netD, netG, configurations):
    #Not using the GAN, only positive and negative images on the discriminator
    #Basically Just a Neural Net.
    #Low Priority TODO May not be needed for mid term
    return 0

def train_project_auto(netD, netG, configurations):
    #Train only with GANS. ALA, our default.

    # Root directory for dataset
    dataroot = "data/basic"

    # Number of workers for dataloader
    workers = 4 #should be contoled by cpu resources, not set by user. Lookup to see what the otimal number for this i

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
                lr = float(config.split(":")[-1])
            except ValueError:
                print("Learning Rate: Non Numeric Parameter Entered. Proceeding...")
                
        elif "beta" in config:
            try:
                beta1 = float( config.split(":")[-1])
            except ValueError:
                print("Beta: Non Numeric Parameter Entered. Proceeding...")

        elif "source" in config:
            if os.path.exists("data/"+ config.split(":")[-1]): 
                dataroot = "data/" + config.split(":")[-1]
            else:
                print("Data Source Not Found.")

    print("Conifgs")
    print("learning_rate: " + str(lr))
    print("num_epochs: " + str(num_epochs))
    print("beta: " + str(beta1))
    print("image_size: "+ str(image_size))
    print("batch_size: "+ str(batch_size))



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
            #print (i)
            #print(data)
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
            #print(type(netG))

            

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
            if (iters % 25 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
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


def train_project_plus_negatives(netD, netG, configurations):
    #Train with GANS and user submitted negative class images. Good for specializing on a topic 

    # Root directory for dataset
    dataroot_pos= "data/basic"
    dataroot_neg= "data/negative"

    # Number of workers for dataloader
    workers = 4 #should be contoled by cpu resources, not set by user. Lookup to see what the otimal number for this i

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
                lr = float(config.split(":")[-1])
            except ValueError:
                print("Learning Rate: Non Numeric Parameter Entered. Proceeding...")
                
        elif "beta" in config:
            try:
                beta1 = float( config.split(":")[-1])
            except ValueError:
                print("Beta: Non Numeric Parameter Entered. Proceeding...")

        elif "pos_source" in config:
            if os.path.exists("data/"+ config.split(":")[-1]): 
                dataroot_pos = "data/" + config.split(":")[-1]
            else:
                print("Data Source Not Found. Using Defaults")

        elif "neg_source" in config:
            if os.path.exists("data/"+ config.split(":")[-1]): 
                dataroot_neg = "data/" + config.split(":")[-1]
            else:
                print("Data Source Not Found.Using Defaults")
        

    print("\nConifgs ")
    print("pos_source: "+dataroot_pos)
    print("neg_source: "+ dataroot_neg)
    print("learning_rate: " + str(lr))
    print("num_epochs: " + str(num_epochs))
    print("beta: " + str(beta1))
    print("image_size: "+ str(image_size))
    print("batch_size: "+ str(batch_size)+"\n\n")



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
    dataset = dset.ImageFolder(root=dataroot_pos,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)




    # We can use an image folder dataset the way we have it setup.
    # Create the dataset
    dataset_neg = dset.ImageFolder(root=dataroot_neg,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    # Create the dataloader
    dataloader_neg = torch.utils.data.DataLoader(dataset_neg, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)
    

    print("Starting Training Loop...")
    
    #TODO Change the training so that examples are intmixed more
    # For each epoch
    for epoch in range(num_epochs):

        # For each batch in the user supplied negative 
        for i, data in enumerate(dataloader_neg, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with  batch of user supplied negatives
            netD.zero_grad()
            # Format batch
            usr_fake_cpu = data[0].to(device)    
            b_size = usr_fake_cpu.size(0)
            label = torch.full((b_size,), fake_label, device=device)
            # Forward pass real batch through D
            output = netD(usr_fake_cpu).view(-1)
     
            # Calculate loss on negative batch
            errD_real = criterion(output, label)
            # Calculate gradient for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            errD = errD_real
            optimizerD.step()
        
            # Save Loss for plotting later
            D_losses.append(errD.item())            


        # For each batch in the regular dataloader
        for i, data in enumerate(dataloader, 0):
            #print (i)
            #print(data)
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
            #print(type(netG))

            

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
            if (iters % 25 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
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




def validate(netD, netG, source):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    device = "cpu"
    #device = "cuda"


    # Copy the content of 
    # source to destination 
    shutil.copyfile(source, "data/validate_folder/validate/"  + "validate.pgm") 

    

    dataset = dset.ImageFolder(root="data/validate_folder",
                            transform=transforms.Compose([
                                transforms.Resize(64),
                                transforms.CenterCrop(64),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                            shuffle=False, num_workers=2)
    
    for i, data in enumerate(dataloader, 0):
            real_cpu = data[0].to(device)
            
            output = netD(real_cpu).view(-1)
            
            outcome =  "Authentic"
            action = "Access: Granted"

            if output[0].item() < .5:
                outcome = "Inauthentic"
                action = "Access: Denied"
            
            print("FREE-M.A.N. predicts sample ~"+source+"~ is "+ outcome+ " with "  + str( int(round(abs(output[0].item() -.5)/.50 , 2) *100) ) + "% confidence. " + action )
            print("Raw OUTPUT: "   + str(output[0].item())  +"\n" )

    os.remove("data/validate_folder/validate/validate.pgm")
    
    return 0







if __name__ == "__main__":

    print("\nService Oriented Computing:  Project Free-M.A.N. ")
    print(" ~An Open Source, cloud hosted Tool As A Service (TaaS) for GAN based Machine Learning \n  ")

    print("Program Starting... Yeet \n \n")

    #Program Arguments, Later, we will automate this behind HTML
    #TODO List flag args
    parser = argparse.ArgumentParser(description='Service Oriented Computing:  Project Free-M.A.N. Interface')
    parser.add_argument('-l', '--load', action='store', help='Load existing GAN. Omit to work on new instance') 
    parser.add_argument('-c', '--config', action='store',nargs="*", help='Initialize new GAN and configure HyperParameters.') #Use nargs = '*' for multiple arguments
    
    parser.add_argument('-1', '--non_gan_train', action='store', nargs="*", help='Train Mode lwith provided positive and negative class examples without Generator + Display Loss')
    parser.add_argument('-2', '--train', action='store', nargs="*", help='Train Model on  provided positive examples and Display Training Results')
    parser.add_argument('-3', '--gan_plus_train', action='store', nargs="*", help='Train Mode lwith provided positive and negative class examples without Generator + Display  Results')

    parser.add_argument('-v', '--validate', action='store', help='Validate Single Image (Real or Fake)')
    parser.add_argument('-o', '--output_name', action='store', help='Name to save model as. Omit to use existing name or randomized name. ')
    args = parser.parse_args()

 

    project = [] #Holds  our Discriminator and Generator for later function calls
    name = str(manualSeed) #Will be our default generated name unless we load an existing project name, or an output name is specifed

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load Project vs Configure and Instantiate New Project

    
    if args.load: #If no load flag, initialize a new GAN
        print("Loading Data for Project... \n")
        name = args.load 
        #project.append(load_project(name))
        if project[0] == 0:
            print("\n \nLoad Failed. Exiting")
            sys.exit(0)

    else: #args.config: #Users can set specific configurations or leave blank to default all values
        print("Initializing New Project with User-Provided or Default Configuratons... \n ")
        configurations = []
        if args.config != None:
            configurations = args.config
        project.append(instantiate(configurations))


    #  END Load Project vs Configure and Instantiate New Project
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # TRAIN OPTIONS

    # Base train.User provides positive examples and let the generator create negs.
    #  Users can set specific configurations or leave blank to default
    if args.train: 
        configurations = []
        if args.train != None:
            configurations = args.train
        print("\n raw train configs: ")
        print(args.train )
        print("\n\n ")
        train_project_auto(project[0][1],project[0][0], configurations)

    # "Reductive" train. User provides positive examples and negative examples to train discriminator. No Generator 
    #  Users can set specific configurations or leave blank to default
    elif args.non_gan_train:
        configurations = []
        if args.non_gan_train != None:
            configurations = args.train
        print("raw non_train configs: ")
        print(args.train)
        print("STUMP NON GAN TRAIN (NON GAN TRAIN IS NOT IMPLEMENTED AT THIS TIME)")
        print("\n\n ")
    
    # "Combo" train. User provides positive examples and negative examples to train discriminator alongside Generator generated fakes/negatives
    #  Users can set specific configurations or leave blank to default
    elif args.gan_plus_train:
        configurations = []
        if args.gan_plus_train != None:
            configurations = args.gan_plus_train
        print("\n Pure Gan_plus_train configs: ")
        print(args.gan_plus_train)
        print("\n\n ")
        train_project_plus_negatives(project[0][1],project[0][0], configurations)

    

    # END TRAIN OPTIONS
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    # Main Driver for Authentication Service.
    # User Provides a single image, GAN predicts image as positive class instance or negative class instance
    # Needs argument (picture location) to run
    if args.validate: 
        #check if file exisists
        if os.path.exists(name):
            print("Validating Picture : " + args.validate + "\n" )
            validate(project[0][1],project[0][0], args.validate )

        else:
            print("Validation Picture Not Found")

       
       



    if args.output_name: # Set output name when saving. On blank, use default name for project saving and loading 
        #TODO: Need to check if has legal characters
        if not os.path.exists(args.output_name):
            name = args.output_name
        else: 
            print("Project Name Already Exists")


    #Always save project at end of run
    #save_project(name,project[0][1],project[0][0])

    print(" \nProgam Finished. \nExiting... Yeet")
