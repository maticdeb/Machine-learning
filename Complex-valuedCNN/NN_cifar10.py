import numpy as np
import torch as t
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import time
import statistics 


trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = datasets.CIFAR10('../data', train=True, transform=trans, download=True)
valset = datasets.CIFAR10('../data', train=False, transform=trans, download=True)

trainloader = t.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = t.utils.data.DataLoader(valset, batch_size=64, shuffle=True)
        

class ANN(nn.Module):
    def __init__(self):
        super(ANN,self).__init__()
        self.linear1 = nn.Linear(3*32*32, 512) 
        self.linear2 = nn.Linear(512, 128) 
        self.linear3 = nn.Linear(128, 32) 
        self.final = nn.Linear(32, 10)
        self.relu = nn.ReLU()

    def forward(self, img): #convert + flatten
        x = img.view(-1, 3*32*32)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.final(x)
        return x
    

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

epoch = 25
najboljsa = 0

dataiter = iter(trainloader)
images, labels = dataiter.next()    
ANN1 = ANN()
ANN1.cuda()
cross_el = nn.CrossEntropyLoss()

optimizer = t.optim.Adam(ANN1.parameters(), lr=0.001) #e-1
def train(ANN1, device, train_loader, optimizer, epoch):
    ANN1.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = ANN1(data.view(-1, 3*32*32))
        loss = cross_el(output, target)
        loss.backward()
        optimizer.step()
    
    print(f'Train Epoch: {epoch} \tLoss: {loss.item()}')

for epoch in range(epoch):
    train(ANN1, device, trainloader, optimizer, epoch)


correct, total = 0,0
with t.no_grad():
    for batch_idx, (data, target) in enumerate(valloader):
        data, target = data.to(device), target.to(device)
        output = ANN1(data.view(-1, 3*32*32))
        for idx, i in enumerate(output):
            if t.argmax(i) == target[idx]:
                correct +=1
            total +=1

natancnost = correct/total
print(f'accuracy: {round(natancnost, 3)}')      
if natancnost > najboljsa:
        t.save(ANN1.state_dict(), "D:/seminar/koda/utezi_ANN_cifar")

print("ANN")   
print(natancnost)
