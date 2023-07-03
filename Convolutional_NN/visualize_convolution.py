import pickle
import torch
from model import ModelCT, Model_moj
from datareader import DataReader
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import torch.nn.functional as F
import numpy as np

# Razred za shranjevanje konvolucij
class SaveOutput:
    def __init__(self):
        self.outputs = []
        
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
        
    def clear(self):
        self.outputs = []
if __name__ == '__main__':
    
    # Nalozimo testne podatke
    main_path_to_data = 'D:/strojno_ucenje/5Covid/processed' 
    model_folder = 'model_moj'
    with open (main_path_to_data + "/test_info", 'rb') as fp:
            test_info = pickle.load(fp) 
    
    # Nalozimo model, ki ga damo v eval mode
    Model = Model_moj()
    Model.load_state_dict(torch.load("D:/strojno_ucenje/5Covid/trained_models/"+model_folder+"/BEST_model.pth"))
    Model.eval()
    Model.cpu()

    # Inicializiramo razred v SO in registriramo kavlje v nasem modelu
    SO = SaveOutput()
    for layer in Model.modules():
        if isinstance(layer, torch.nn.modules.conv.Conv2d):
            handle = layer.register_forward_hook(SO)

    # Naredimo test_generator z batch_size=1
    test_datareader = DataReader(main_path_to_data, test_info)
    test_generator = DataLoader(test_datareader, batch_size=1, shuffle=False, pin_memory=True, num_workers=2)

    # Vzamemo prvi testni primer npr.
    item_test = next(iter(test_generator))
    # Propagiramo prvi testni primer skozi mrezo, razred SaveOutput si shrani vse konvolucije
    input = item_test[0].cpu()
    output = Model(input)

    # Izberemo katero konvolucijo bi radi pogledali (color_channel bo vedno 0)
    color_channel = 0 # indeks barvnega kanala (pri nas le 1 kanal) 
    idx_layer =0  # indeks konvolucijske plasti (Conv2d) - (pri nas 21) 
    idx_convolution = 2 # indeks konvolucije na dani plasti (max odvisen od plasti)

    # Vizualiziramo

    for i in range(21):

        x = SO.outputs[i][color_channel][0][:,:].cpu().detach().numpy()
        plt.imshow(np.rot90(x), cmap='gray')
        plt.title(f"plast {i}")
        plt.show()