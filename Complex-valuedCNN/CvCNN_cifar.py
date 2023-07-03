import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from complexPyTorch.complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear
from complexPyTorch.complexFunctions import complex_relu, complex_max_pool2d
import time

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = datasets.CIFAR10('../data', train=True, transform=trans, download=True)
valset = datasets.CIFAR10('../data', train=False, transform=trans, download=True)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

najboljsa = 0

class CvCNN(nn.Module):
    
    def __init__(self):
        super(CvCNN, self).__init__()
        self.conv1 = ComplexConv2d(3, 64, 3, 1)
        self.bn  = ComplexBatchNorm2d(30)
        self.conv2 = ComplexConv2d(64, 64, 2, 1)
        self.fc1 = ComplexLinear(3136,128)
        self.fc2 = ComplexLinear(128, 10)
             
    def forward(self,x):
        x = self.conv1(x)
        x = complex_relu(x)
        x = complex_max_pool2d(x, 2, 2)
        #x = self.bn(x)
        x = self.conv2(x)
        x = complex_relu(x)
        x = complex_max_pool2d(x, 2, 2)
        x = torch.flatten(x,1)
        x = self.fc1(x)
        x = complex_relu(x)
        x = self.fc2(x)
        x = x.abs()
        x =  F.log_softmax(x, dim=1)
        return x
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cv = CvCNN().to(device)
optimizer = torch.optim.SGD(cv.parameters(), lr=0.01, momentum=0.9)



pytorch_total_params = sum(p.numel() for p in cv.parameters() if p.requires_grad)

print(pytorch_total_params)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device).type(torch.complex64), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
    print(f'Train Epoch: {epoch} \tLoss: {loss.item()}')


# Run training on 50 epochs
for epoch in range(25):
    train(cv, device, trainloader, optimizer, epoch)


correct, total = 0,0
with torch.no_grad():
    for batch_idx, (data, target) in enumerate(valloader):
        data, target = data.to(device).type(torch.complex64), target.to(device)
        output = cv(data)
        for idx, i in enumerate(output):
            if torch.argmax(i) == target[idx]:
                correct +=1
            total +=1

natancnost = correct/total
 
print("CvCNN")			
print(f'accuracy: {round(correct/total, 3)}')     
if natancnost > najboljsa:
        torch.save(cv.state_dict(), "D:/seminar/koda/utezi_CvCNN_cifar")