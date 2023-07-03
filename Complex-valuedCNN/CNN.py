import numpy as np
import torch as t
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import time

transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

trainset = datasets.MNIST('../data', download=True, train=True, transform=transform)
valset = datasets.MNIST('../data', download=True, train=False, transform=transform)
trainloader = t.utils.data.DataLoader(trainset, batch_size=200, shuffle=True)
valloader = t.utils.data.DataLoader(valset, batch_size=200, shuffle=True)


dataiter = iter(trainloader)
images, labels = dataiter.next()

    
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(1,60,5,1)
        self.bn = nn.BatchNorm2d(30)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(60,30,5,1)
        self.fc1 = nn.Linear(480,128)
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
print('CNN')
print(pytorch_total_params)


optimizer = t.optim.Adam(CNN1.parameters(), lr=0.001) #e-1
najboljsa = 0.9902
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
    
    #print(f'Train Epoch: {epoch} \tLoss: {loss.item()}')

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
        t.save(CNN1.state_dict(), "D:/seminar/koda/utezi_CNN")

print("CNN")   
print(natancnost)



"""
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype('float32')
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32')


x_train = x_train / 255
x_test = x_test / 255


y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


def CNN():
	# create model
	model = Sequential()
	model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu'))
	model.add(MaxPooling2D())
	model.add(Conv2D(15, (3, 3), activation='relu'))
	model.add(MaxPooling2D())
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(10, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

model = CNN()

history = model.fit(x_train,y_train, epochs= 10, batch_size=200, validation_split=0.1, shuffle=True ,verbose=2)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='validation')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.show()

scores = model.evaluate(x_test, y_test, verbose=2)
print("konvolucijska mreža")
print(scores)
"""