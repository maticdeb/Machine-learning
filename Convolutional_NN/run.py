import pickle

from matplotlib.colors import Normalize
from train import Training
from test import Testing
import sklearn.metrics 
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__': 

    main_path_to_data = 'D:/strojno_ucenje/5Covid/processed' 
    unique_name_for_this_run = 'yolo123' # neobvezno  

    # Nalozimo sezname za ucno, validacijsko in testno mnozico
    with open (main_path_to_data + "/train_info", 'rb') as fp:
        train_info = pickle.load(fp)
    with open (main_path_to_data + "/val_info", 'rb') as fp:
        valid_info = pickle.load(fp)
    with open (main_path_to_data + "/test_info", 'rb') as fp:
        test_info = pickle.load(fp) 

    # Nastavimo hiperparametre v slovarju
    hyperparameters = {}
    hyperparameters['learning_rate'] = 0.2e-3 # learning rate
    hyperparameters['weight_decay'] = 0.0001 # weight decay
    hyperparameters['total_epoch'] = 20 # total number of epochs
    hyperparameters['multiplicator'] = 0.95 # each epoch learning rate is decreased on LR*multiplicator

    # Ustvarimo ucni in testni razred
    TrainClass = Training(main_path_to_data, unique_name=unique_name_for_this_run)
    TestClass = Testing(main_path_to_data)

    # Naucimo model za izbrane hiperparametre
    aucs, losses, path_to_model = TrainClass.train(train_info, valid_info, hyperparameters)

    # Najboljsi model glede na validacijsko mnozico (zadnji je /LAST_model.pth)
    best_model = path_to_model + '/BEST_model.pth' 

    # Testiramo nas model na testni mnozici
    auc, fpr, tpr, thresholds, trues, predictions = TestClass.test(test_info, best_model)

    print("Test set AUC result: ", auc)

    plt.plot(fpr,tpr)
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.show()

    kriterij = np.linspace(0,1,101)
    threshold = 0
    najmnajsa_cenilka = 10000
    for k in kriterij:
        napoved_zdravi_ampak_so_bolni = 0
        napoved_bolni_ampak_so_zdravi = 0
        prav_bolni = 0
        prav_zdravi = 0
        cenilka = 0
        for i in range(len(trues)):
            if trues[i] == 1 and predictions[i] < k:
                napoved_zdravi_ampak_so_bolni += 1
                #print(trues[i], predictions[i])
            elif trues[i] == 0 and predictions[i] > k:
                napoved_bolni_ampak_so_zdravi += 1
                #print(trues[i], predictions[i])
            elif trues[i] == 1 and predictions[i] > k:
                prav_bolni +=1
            else:
                prav_zdravi += 1
        cenilka = napoved_bolni_ampak_so_zdravi + napoved_zdravi_ampak_so_bolni
        print(cenilka)
        if cenilka < najmnajsa_cenilka:
                najmnajsa_cenilka = cenilka
                threshold = k
    print("threshold:", threshold)
    for i in range(len(predictions)):
        if predictions[i] > threshold:
            predictions[i] = 1
        else:
            predictions[i] = 0      

    print("f1", sklearn.metrics.f1_score(trues,predictions))
    print("natančnost:" ,sklearn.metrics.accuracy_score(trues, predictions))
    print("matrika:", sklearn.metrics.confusion_matrix(trues, predictions))

    plt.imshow(sklearn.metrics.confusion_matrix(trues,predictions))
    plt.colorbar()
    plt.show()

    plt.imshow(sklearn.metrics.confusion_matrix(trues,predictions, normalize='true'))
    plt.colorbar()
    plt.show()