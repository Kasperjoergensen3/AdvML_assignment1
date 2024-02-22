from pathlib import Path
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch


from AMLsrc.utilities.load_model import load_model, load_flow_model
from AMLsrc.data.dataloader import get_MNIST_dataloader

#set seed
torch.manual_seed(1236)

_, mnist_test_loader = get_MNIST_dataloader(batch_size=32, transform_description='standard')


model = ['',load_model(Path('models/partB_VAE_test_version/1708100371')),
        load_flow_model(Path('models/partB_flow_test_version/1708099010')),
        load_flow_model(Path('models/partB_ddpm_v1/1708259498'))
        ]

titles = ['MNIST', 'VAE', 'Flow', 'DDMP']

fig, ax = plt.subplots(4, 8, figsize=(10, 5))
for i in range(4):
    print(i)
    for j in range(8):
        if i == 0:
            sample = mnist_test_loader.dataset[j][0].view(28, 28)
        else:
            with torch.no_grad():
                if i == 3:
                    sample = model[i].sample((1,784)).view(28, 28)
                else:
                    sample = model[i].sample(1).view(28, 28)
        ax[i, j].imshow(sample, cmap='gray')
        if j == 0:
            ax[i, j].set_ylabel(titles[i], fontsize=15)
        #remove ticks and tick labels
        ax[i, j].set_xticks([])
        ax[i, j].set_yticks([])
        ax[i, j].set_xticklabels([])
        ax[i, j].set_yticklabels([])


plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig('reports/PartB_samples.png', bbox_inches='tight', dpi=300)


# compute FID for each model
# generate two slightly overlapping image intensity distributions
# imgs_dist1 = torch.randint(0, 200, (100, 3, 299, 299), dtype=torch.uint8)
test_data = []
for x, _ in mnist_test_loader:
    test_data.append(x)
test_data = torch.cat(test_data, 0).unsqueeze(1)
print(type(test_data))
print(test_data.shape)


# make samples from models
vae_samples = model[1].sample(10000)
print(type(vae_samples))
print(vae_samples.shape)

# flow_samples = model[2].sample(10000).view(10000, 1, 28, 28)
# print(type(flow_samples))
# print(flow_samples.shape)

# ddpm_samples = model[3].sample(10000)


import torch
from torchmetrics.image.kid import KernelInceptionDistance

#set up for cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_samples = vae_samples
model_samples = model_samples.to(device)
test_data = test_data.to(device)

kid = KernelInceptionDistance(subset_size=50, normalize=True)
# generate two slightly overlapping image intensity distributions
imgs_dist1 = test_data.repeat(1,3,1,1)
imgs_dist2 = model_samples.repeat(1, 3, 1, 1)
kid.update(imgs_dist1, real=True)
kid.update(imgs_dist2, real=False)
mean_kid, std_kid = kid.compute()

print(f"The mean KID {mean_kid} +- {std_kid}")

del kid

# from torchmetrics.image.fid import FrechetInceptionDistance
# fid = FrechetInceptionDistance(feature=64, normalize=True)
# # generate two slightly overlapping image intensity distributions
# fid.update(imgs_dist1, real=True)
# fid.update(imgs_dist2, real=False)
# fid = fid.compute()
# print(f"The mean KID {mean_kid} +- {std_kid} , and FID {fid}")

