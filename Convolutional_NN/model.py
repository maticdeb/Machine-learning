import torch
import torchvision.models as tmodels
import torch.nn as nn
import torch.nn.functional as F


def batch_to_cpu(item):
    return (i.cuda() for i in item)

class ModelCT(nn.Module):
    def __init__(self):
        super(ModelCT, self).__init__()
        self.backbone = tmodels.resnet18(pretrained=True)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.convolution2d = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1), bias=True)
        self.fc_maxpool = nn.AdaptiveMaxPool2d((1, 1))

            
    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x) 
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.convolution2d(x)
        x = self.fc_maxpool(x)
        
        x = torch.flatten(x, 1)
        return x
    
    
    def train_update(self, train_tensor_tuple, criterion):
        xb, yb = batch_to_cpu(train_tensor_tuple) 

        outputs = self.forward(xb)
        loss = criterion(outputs, yb) 
        
        return loss
    
    
    def predict(self, item, is_prob = False):
        xb, _ = item 
           
        xb = xb.cuda()        
        with torch.no_grad():
            outputs = self.forward(xb)
            if is_prob:
                outputs = torch.sigmoid(outputs)
        return outputs

def num_flat_features(self, x):
    size = x.size()[1:]
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

class Model_moj(nn.Module):
    def __init__(self):
        super(Model_moj, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
        self.conv2 = nn.Conv2d(4, 4, kernel_size=(2,2), stride=(1,1), padding=(1,1), bias=False)
        self.bn1 = nn.BatchNorm2d(1)
        self.bn2 = nn.BatchNorm2d(4)
        self.fc1 = nn.Linear(65536, 1000)
        self.fc2 = nn.Linear(800, 100)
        self.fc3 = nn.Linear(1000, 1)


    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x,(2,2))
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x,(2,2))

        x = x.view(-1, self.num_flat_features(x))
        
        #print(x.shape)
        x = F.relu(self.fc1(x))
        x = self.fc3(x)  
    
        return x
    
    
    def train_update(self, train_tensor_tuple, criterion):
        xb, yb = batch_to_cpu(train_tensor_tuple) 

        outputs = self.forward(xb)
        loss = criterion(outputs, yb) 
        
        return loss
    
    
    def predict(self, item, is_prob = False):
        xb, _ = item 
           
        xb = xb.cuda()        
        with torch.no_grad():
            outputs = self.forward(xb)
            if is_prob:
                outputs = torch.sigmoid(outputs)
        return outputs        

class Model_vektor(nn.Module):
    def __init__(self):
        super(Model_vektor, self).__init__()

        self.fc1 = nn.Linear(262144, 1000)
        self.fc2 = nn.Linear(10000, 1000)
        self.fc3 = nn.Linear(1000, 1)


    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        x = x.view(-1, self.num_flat_features(x))
        #print(x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        #x = self.fc2(x)
        #x = F.relu(x)
        x = self.fc3(x)

        return x
    
    
    def train_update(self, train_tensor_tuple, criterion):
        xb, yb = batch_to_cpu(train_tensor_tuple) 

        outputs = self.forward(xb)
        loss = criterion(outputs, yb) 
        
        return loss
    
    
    def predict(self, item, is_prob = False):
        xb, _ = item 
           
        xb = xb.cpu()        
        with torch.no_grad():
            outputs = self.forward(xb)
            if is_prob:
                outputs = torch.sigmoid(outputs)
        return outputs  


