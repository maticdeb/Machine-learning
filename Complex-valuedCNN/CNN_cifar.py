import numpy as np
import torch as t
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import time

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
trainset = datasets.CIFAR10('../data', train=True, transform=trans, download=True)
valset = datasets.CIFAR10('../data', train=False, transform=trans, download=True)

trainloader = t.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = t.utils.data.DataLoader(valset, batch_size=64, shuffle=True)
        
dataiter = iter(trainloader)
images, labels = dataiter.next()   
    
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(3,64,3,1)
        self.bn = nn.BatchNorm2d(30)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(64,64,2,1)
        self.fc1 = nn.Linear(3136,128)
        self.fc2 = nn.Linear(128,10)
        self.relu = nn.ReLU()
    def forward(self,x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = t.flatten(x, 1) 
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x 



device = t.device('cuda' if t.cuda.is_available() else 'cpu')
CNN1 = CNN().to(device)
cross_el = nn.CrossEntropyLoss()



pytorch_total_params = sum(p.numel() for p in CNN1.parameters() if p.requires_grad)

print(pytorch_total_params)

optimizer = t.optim.Adam(CNN1.parameters(), lr=0.001) #e-1
najboljsa = 0
epoch = 25
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = cross_el(output, target)
        loss.backward()
        optimizer.step()
    
    print(f'Train Epoch: {epoch} \tLoss: {loss.item()}')

for epoch in range(epoch):
    train(CNN1, device, trainloader, optimizer, epoch)


correct, total = 0,0
with t.no_grad():
    for batch_idx, (data, target) in enumerate(valloader):
        data, target = data.to(device), target.to(device)
        output = CNN1(data)
        for idx, i in enumerate(output):
            if t.argmax(i) == target[idx]:
                correct +=1
            total +=1
            
natancnost = correct/total
print(f'accuracy: {round(natancnost, 3)}')      
if natancnost > najboljsa:
        t.save(CNN1.state_dict(), "D:/seminar/koda/utezi_CNN_cifar")

print("CNN")   
print(natancnost)



