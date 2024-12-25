#!/usr/bin/env python
# coding: utf-8

# In[241]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import logging
from imp import reload
from tqdm import tqdm
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn

from PIL import Image
import os
import torch
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from torchvision.utils import make_grid

from torchvision.datasets import ImageFolder


# In[242]:



class CustomImageDataset(Dataset):
    def __init__(self, folder_path, attributes_file, transform=None):
        """
        Args:
            folder_path (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.folder_path = folder_path
        self.transform = transform
        self.image_filenames = [f for f in sorted(os.listdir(folder_path))[:12000] if not f.startswith('.')]
        self.labels = self.load_attributes(attributes_file)
        
    def load_attributes(self, attributes_file):
        with open(attributes_file, 'r') as file:
            lines = file.readlines()[2:]  # Skip header lines
            labels = {line.split()[0]: [int(x) for x in line.split()[1:]] for line in lines}
        return labels


    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.image_filenames[idx])
        try:
            image = Image.open(img_path).crop((0, 15, 178, 193))
            if self.transform:
                image = self.transform(image)
        except UnidentifiedImageError:
            print(f"Skipping file (not an image or corrupted): {img_path}")
            return None
        labels = self.labels[self.image_filenames[idx]]
        labels = torch.tensor(labels).float()
        labels_converted = (labels + 1) / 2
        batch_sample = {'images': image, 'features': labels_converted}
        return batch_sample


# In[243]:


batch_size = 64
path_to_data_root = '/Users/ranykhirbawi/Downloads/img_align_celeba'
attributes_file_path = '/Users/ranykhirbawi/Downloads/celeba_text_files/list_attr_celeba.txt'

your_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) #####
])



# In[244]:


dataset = CustomImageDataset(folder_path=path_to_data_root, attributes_file=attributes_file_path, transform=your_transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# In[245]:


data_iterator = iter(dataloader)
next_batch = next(data_iterator)
next_batch


# In[246]:


# Print the shape of each tensor
print("Shape of 'images':", next_batch['images'].shape)
print("Shape of 'features':", next_batch['features'].shape)


# In[247]:


stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) # mean, std for normalize imagess
def denorm(img_tensors):
    "Denormalize image tensor with specified mean and std"
    return img_tensors * stats[1][0] + stats[0][0]
def show_images(images, nmax=64):
      fig, ax = plt.subplots(figsize=(8,8))
      ax.set_xticks([]); ax.set_yticks([])
      ax.imshow(make_grid(denorm(images.detach()[:nmax]), nrow=8).permute(1, 2, 0))
  
def show_batch(dl, nmax=64):
      for batch in dataloader:
        images=batch['images']
        show_images(images, nmax)
        break
show_batch(dataloader)


# In[248]:


# CHANGE HERE
import torch
import torch.nn as nn

class Generator(nn.Module):

    def __init__(self, latent_dim, num_attributes):
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        self.num_attributes = num_attributes
        
        self.model = nn.Sequential(
        nn.ConvTranspose2d(latent_dim + num_attributes, 512, kernel_size=4, stride=1, padding=0, bias = False), 
        nn.BatchNorm2d(512),
        nn.ReLU(True),
        # out: 512 x 4 x 4

        nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias = False),
        nn.BatchNorm2d(256),
        nn.ReLU(True),
        # out: 256 x 8 x 8

        nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias = False),
        nn.BatchNorm2d(128),
        nn.ReLU(True),
        # out: 128 x 16 x 16

        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias = False),
        nn.BatchNorm2d(64),
        nn.ReLU(True),
        # out: 64 x 32 x 32

        nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias = False),
        nn.Tanh()  # output is between -1 to 1
        )
        

    

    def forward(self, noise, attributes):
        # Concatenate noise and attributes along dimension 1
        gen_input = torch.cat([noise, attributes], dim=1)
        gen_input = gen_input.view(-1, gen_input.size()[-1], 1, 1)
        img = self.model(gen_input)
        return img


# In[249]:




class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
        # in: 3x 64 x 64
        nn.Conv2d(3+num_classes, 64, kernel_size=4, stride=2, padding=1, bias = False),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.2, inplace=True),
        # out: 64 x 32 x 32

        nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias = False),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2, inplace=True),
        # out: 128 x 16 x 16

        nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias = False),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(0.2, inplace=True),
        # out: 256 x 8 x 8

        nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias = False),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(0.2, inplace=True),
        # out: 512 x 4 x 4

        nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias = False),
        # out: 1 x 1 x 1
        nn.Flatten(),
        nn.Sigmoid()
        )

    def forward(self, img, attributes):
        attributes = attributes.view(-1,attributes.size()[-1],1,1)
        attributes = attributes.expand(-1,-1,img.size()[-2], img.size()[-1])
        d_in = torch.cat([img, attributes], dim=1)
        validity = self.model(d_in)
        return validity


# In[255]:


import torch
import torch.nn as nn
import torch.optim as optim

# Initialize the Generator and Discriminator
generator = Generator(latent_dim=128, num_attributes=40)  # Assuming 40 binary attributes
discriminator = Discriminator(num_classes=40)
# Loss function
adversarial_loss = nn.BCELoss()

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=0.0004)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0004)

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator.to(device)
discriminator.to(device)
adversarial_loss.to(device)


# In[256]:


import matplotlib.pyplot as plt
import numpy as np

def show_generated_images(images, num_images=10):
    images = images.to('cpu').detach().numpy() 
    images = images * 0.5 + 0.5  
    plt.figure(figsize=(10, 10)) 

    for i in range(num_images):
        plt.subplot(5, 2, i+1)  
        plt.imshow(np.transpose(images[i], (1, 2, 0)))  
        plt.axis('off')
    
    plt.show()


# In[257]:


epochs = 85  # Number of epochs
latent_dim = 128  # Size of the latent space
batch_size = 64  # Adjust based on your GPU capacity


# In[258]:


from tqdm import tqdm
discriminator_losses = []
generator_losses = []


# Training Loop
for epoch in range(epochs):
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        
        current_batch_size = batch['images'].size(0)

        
        real_imgs = batch['images'].to(device)
        real_labels = batch['features'].to(device)
        real_labels.size()
                
        

        valid =  torch.ones(current_batch_size).to(device)
        fake =  torch.zeros(current_batch_size).to(device)


        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()

        # 1. Train with fake images and flipped labels

        # Generate a batch of fake images
        z = torch.randn(current_batch_size, latent_dim).to(device)
        # gen_labels = torch.randint(0, 2, (current_batch_size, 40)).float().to(device)

        
        
        gen_imgs = generator(z, real_labels)

        # Generator loss on generated images
        d_fake = discriminator(gen_imgs, real_labels)
        # torch.Size([64, 1])
        
        # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))        
        g_loss = adversarial_loss(d_fake.squeeze(), valid)
        generator_losses.append(g_loss.item())

        
        g_loss.backward()
        optimizer_G.step()

        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        
            # Compute BCE_Loss using real images where BCE_Loss(x, y):
        #         - y * log(D(x)) - (1-y) * log(1 - D(x))

        
        # 1. Train with real images
        
        # Compute the discriminator losses on real images
        # Second term of the loss is always zero since real_labels == 1
        # Here we are maximizing D reward by minimizing -ylog(D(x)) = -log(D(x)) in respect to D weights
        d_real = discriminator(real_imgs, real_labels)
        d_real_loss = adversarial_loss(d_real.squeeze(), valid)
        
        # 2. Train with fake images
        
        # Generate fake images
        z = torch.randn(current_batch_size, latent_dim).to(device)
        gen_imgs = generator(z, real_labels)

        # Compute the discriminator losses on fake images
        # First term of the loss is always zero since fake_labels == 0
        # Here we are maximizing D reward by minimizing - (1-y) * log(1 - D(x)) = log(1 - D(x))
        # (becuase these are fake and we want the discriminator to be able to give them low probability)
        d_fake = discriminator(gen_imgs, real_labels)
        d_fake_loss = adversarial_loss(d_fake.squeeze(), fake)

        d_loss = (d_real_loss + d_fake_loss)/2
        discriminator_losses.append(d_loss.item())

        # add up loss and perform backprop
        d_loss.backward()
        optimizer_D.step()

        # Print training status
        if i % 188 == 0:  
            print(f"[Epoch {epoch+1}/{epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")
    


    # Generate some images for visual inspection
    with torch.no_grad():
        test_z = torch.randn(batch_size, latent_dim).to(device)
        test_labels = torch.randint(0, 2, (batch_size, 40)).float().to(device) ####
        generated_images = generator(test_z, test_labels).detach().cpu()
    show_generated_images(generated_images, num_images=10)


# In[259]:


epochs = range(1, len(discriminator_losses) // len(dataloader) + 1)
avg_d_losses = [np.mean(discriminator_losses[i * len(dataloader):(i + 1) * len(dataloader)]) for i in range(len(epochs))]
avg_g_losses = [np.mean(generator_losses[i * len(dataloader):(i + 1) * len(dataloader)]) for i in range(len(epochs))]

plt.figure(figsize=(10, 5))
plt.plot(epochs, avg_d_losses, label='Discriminator Loss')
plt.plot(epochs, avg_g_losses, label='Generator Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Losses')
plt.legend()
plt.show()


# In[260]:


plt.figure(figsize=(10, 5))
plt.plot(discriminator_losses, label='Discriminator Loss', alpha=0.7)
plt.plot(generator_losses, label='Generator Loss', alpha=0.7)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training Losses')
plt.legend()
plt.show()


# In[261]:


torch.save(generator.state_dict(), '/Users/ranykhirbawi/Downloads/generator_weights_final.pkl')
torch.save(discriminator.state_dict(), '/Users/ranykhirbawi/Downloads/discriminator_weights_final.pkl')


# In[262]:


import torch
import pickle

# Load the PyTorch model weights
generator_weights = torch.load('/Users/ranykhirbawi/Downloads/generator_weights_final.pkl')
discriminator_weights = torch.load('/Users/ranykhirbawi/Downloads/discriminator_weights_final.pkl')

# Save the model weights as .pkl files
with open('/Users/ranykhirbawi/Downloads/generator_weights_final.pkl', 'wb') as f:
    pickle.dump(generator_weights, f)

with open('/Users/ranykhirbawi/Downloads/discriminator_weights_final.pkl', 'wb') as f:
    pickle.dump(discriminator_weights, f)


# In[263]:


def load_pickle_as_state_dict(pickle_file_path):
    with open(pickle_file_path, 'rb') as f:
        return pickle.load(f)


# In[264]:


attributes_dict = {'5_o_Clock_Shadow': 0, 'Arched_Eyebrows': 1, 'Attractive': 2, 'Bags_Under_Eyes': 3, 'Bald': 4, 'Bangs': 5, 'Big_Lips': 6, 'Big_Nose': 7, 'Black_Hair': 8, 'Blond_Hair': 9, 'Blurry': 10, 'Brown_Hair': 11, 'Bushy_Eyebrows': 12, 'Chubby': 13, 'Double_Chin': 14, 'Eyeglasses': 15, 'Goatee': 16, 'Gray_Hair': 17, 'Heavy_Makeup': 18, 'High_Cheekbones': 19, 'Male': 20, 'Mouth_Slightly_Open': 21, 'Mustache': 22, 'Narrow_Eyes': 23, 'No_Beard': 24, 'Oval_Face': 25, 'Pale_Skin': 26, 'Pointy_Nose': 27, 'Receding_Hairline': 28, 'Rosy_Cheeks': 29, 'Sideburns': 30, 'Smiling': 31, 'Straight_Hair': 32, 'Wavy_Hair': 33, 'Wearing_Earrings': 34, 'Wearing_Hat': 35, 'Wearing_Lipstick': 36, 'Wearing_Necklace': 37, 'Wearing_Necktie': 38, 'Young': 39}

def generate_attribute_tensor(batch_size, attributes, attribute_dict):
    attribute_tensor = torch.zeros(batch_size, len(attribute_dict))

    for attribute in attributes:
        if attribute in attribute_dict:
            idx = attribute_dict[attribute]
            attribute_tensor[:, idx] = 1

    return attribute_tensor


# In[278]:


from torchvision.utils import save_image

def reproduce_hw3(generator_weights_path, discriminator_weights_path, attributes_to_set, num_images=24, save_path="/Users/ranykhirbawi/Downloads/"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(os.path.abspath(generator_weights_path))


    generator = Generator(latent_dim=128, num_attributes=40).to(device)  
    discriminator = Discriminator(num_classes=40).to(device)


    generator.load_state_dict(load_pickle_as_state_dict(generator_weights_path))
    discriminator.load_state_dict(load_pickle_as_state_dict(discriminator_weights_path))


    generator.eval()
    discriminator.eval()

    # Generate images
    with torch.no_grad():
        z = torch.randn(num_images, latent_dim).to(device)
        attribute_tensor = generate_attribute_tensor(num_images, attributes_to_set, attributes_dict)
        gen_imgs = generator(z, attribute_tensor)

    os.makedirs(save_path, exist_ok=True)

    save_image(gen_imgs, save_path + "generated_images_final.png", nrow=8, normalize=True)

    # Display the generated images
    print("Generated Images:")
    plt.figure(figsize=(18, 18))  
    for j, img in enumerate(gen_imgs):
        img = img.cpu().detach()  
        img = (img + 1) / 2       
        img = img.permute(1, 2, 0)  
        plt.subplot(8, 8, j + 1)
        plt.imshow(img.numpy().squeeze())  
        plt.axis('off')
    plt.show()
attributes_dict = {'5_o_Clock_Shadow': 0, 'Arched_Eyebrows': 1, 'Attractive': 2, 'Bags_Under_Eyes': 3, 'Bald': 4, 'Bangs': 5, 'Big_Lips': 6, 'Big_Nose': 7, 'Black_Hair': 8, 'Blond_Hair': 9, 'Blurry': 10, 'Brown_Hair': 11, 'Bushy_Eyebrows': 12, 'Chubby': 13, 'Double_Chin': 14, 'Eyeglasses': 15, 'Goatee': 16, 'Gray_Hair': 17, 'Heavy_Makeup': 18, 'High_Cheekbones': 19, 'Male': 20, 'Mouth_Slightly_Open': 21, 'Mustache': 22, 'Narrow_Eyes': 23, 'No_Beard': 24, 'Oval_Face': 25, 'Pale_Skin': 26, 'Pointy_Nose': 27, 'Receding_Hairline': 28, 'Rosy_Cheeks': 29, 'Sideburns': 30, 'Smiling': 31, 'Straight_Hair': 32, 'Wavy_Hair': 33, 'Wearing_Earrings': 34, 'Wearing_Hat': 35, 'Wearing_Lipstick': 36, 'Wearing_Necklace': 37, 'Wearing_Necktie': 38, 'Young': 39}
attributes_to_set = ['Attractive', 'Heavy_Makeup', 'High_Cheekbones', 'No_Beard', 'Wavy_Hair', 'Wearing_Lipstick','Young']

reproduce_hw3("/Users/ranykhirbawi/Downloads/generator_weights_final.pkl","/Users/ranykhirbawi/Downloads/discriminator_weights_final.pkl", attributes_to_set)


# In[ ]:




