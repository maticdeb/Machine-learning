import numpy as np
import torch
from torch.utils.data import DataLoader
from model import ModelCT, Model_moj, Model_vektor
from datareader import DataReader
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt


class Testing():
    def __init__(self, main_path_to_data):
        self.main_path_to_data = main_path_to_data
        
    def test(self, test_info, path_to_model):
        """Test given model with task, path to the model and model datareder names."""
        
        # 1. Load trained model and set it to eval mode
        Model = ModelCT()
        Model.load_state_dict(torch.load(path_to_model))
        Model.eval()
        Model.cuda()
        
        # 2. Create datalodaer
        test_datareader = DataReader(self.main_path_to_data, test_info)
        test_generator = DataLoader(test_datareader, batch_size=10, shuffle=False, pin_memory = True, num_workers=2)

        # 3. Calculate metrics
        predictions = []
        trues = []
        
        for item_test in test_generator:
            prediction = Model.predict(item_test, is_prob=True)
            #predictions.append(np.mean(prediction.cpu().numpy()))
            predictions.append(np.max(prediction.cpu().numpy()))
            #predictions.append(np.mean(sorted(prediction.cpu().numpy(), reverse=True)[:3]))
            trues.append(item_test[1].numpy()[0])
            
        auc = roc_auc_score(trues, predictions)
        fpr, tpr, thresholds = roc_curve(trues, predictions, pos_label=1)

        print(thresholds)
        a = sorted(zip(predictions,trues))

        b = []
        c = []
        for i in range(len(a)):
            b.append(a[i][0])
            c.append(a[i][1])

        plt.plot(c,"bo", label="true")
        plt.plot(b,"ko", label="prediction")
        plt.ylabel("napoved")
        plt.xlabel("število pacienta")
        plt.legend()
        plt.show()

        return auc, fpr, tpr, thresholds, trues, predictions