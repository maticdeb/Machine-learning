import torch.nn.functional as F
import torch
import numpy as np
from captum.attr import Saliency
from datareader import DataReader
from model import ModelCT, Model_moj
from torch.utils.data import DataLoader
import pickle
from matplotlib import pyplot as plt

if __name__ == "__main__":
    # Nalozimo testne podatke
    main_path_to_data = 'D:/strojno_ucenje/5Covid/processed' 
    model_folder = 'model_moj'
    
    with open (main_path_to_data + "/test_info", 'rb') as fp:
            test_info = pickle.load(fp)     

    # Nalozimo model, ga damo v eval mode
    Model = Model_moj()
    Model.load_state_dict(torch.load("trained_models/"+model_folder+"/BEST_model.pth"))
    Model.eval()
    Model.cpu()

    # Naredimo testni generator
    test_datareader = DataReader(main_path_to_data, test_info)
    test_generator = DataLoader(test_datareader, batch_size=1, shuffle=False, pin_memory = True, num_workers=2)

    # Iz knjiznice captum nalozimo Saliency
    saliency = Saliency(Model)
    
    # V testnih podatkih poiscemo primer z dobro klasifikacijo hude okuzbe (y_true==1, y_pred > 0.95)
    for item_test in test_generator:
        
        input = item_test[0].cpu()
        output = Model(input)
        y_pred = F.sigmoid(output).cpu().detach().numpy()[0][0]
        y_true = item_test[1].numpy()[0][0]
        
        if y_true == 0 and y_pred > 0.9:
            attributions = saliency.attribute(input)
            attribution = np.rot90(attributions.detach().cpu().numpy()[0,0,:,:].reshape((512,512)))
            original = np.rot90(input.cpu().numpy()[0,0,:,:].reshape((512,512)))
            break

    # Vizualiziramo saliency map in originalno sliko
    plt.figure()
    plt.imshow(original, cmap='Greys_r')
    plt.imshow(attribution, alpha=0.7, cmap='rainbow')
    plt.axis('off')
    plt.savefig("plots/attribution2.png", dpi=500, bbox_inches='tight')
    
    plt.figure()
    plt.imshow(original, cmap='Greys_r')
    plt.axis('off')
    plt.savefig("plots/original2.png", dpi=500, bbox_inches='tight')