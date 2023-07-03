from model import ModelCT, Model_moj
from matplotlib import pyplot as plt
import numpy as np
import torch

model_folder = 'modelCT' # mapa naucenega modela
Model = ModelCT() # model
# Nalozimo model z utezmi
Model.load_state_dict(torch.load("D:/strojno_ucenje/5Covid/trained_models/"+model_folder+"/BEST_model.pth"))

# Nalozimo npr. filtre iz prve konvolucijske plasti backbone.conv1
weights = Model.convolution2d.weight.data.numpy()
print(weights.shape)

# weights.shape = (64,1,7,7) -> 64 filtrov, z 1 kanalom, velikosti 7x7
print()
# Vizualizacija 21. filtra iz prve plasti (backbone.conv1)

"""
for i in range(0,64):
    for j in range(1):
        print(weights[i,j,:,:])
        plt.imshow(weights[i,j,:,:], cmap='gray')
        plt.show()
"""        

