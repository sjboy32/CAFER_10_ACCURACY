import torchvision
import torch
# from keras.src.metrics.accuracy_metrics import accuracy
# from PyQt5.QtGui.QRawFont import leading
# from openpyxl.styles.builtins import output
from torch import nn
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_data_set = torchvision.datasets.CIFAR10(root='dataset',train=True,transform=torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(size=(32,32),padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean = [0.5,0.5,0.5],std=[0.5,0.5,0.5])

]),download=True)
test_data_set = torchvision.datasets.CIFAR10(root='dataset',train=True,transform=torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(size=(32,32),padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean = [0.5,0.5,0.5],std=[0.5,0.5,0.5])

]),download=True)
train_data_load = DataLoader(dataset=train_data_set,batch_size=64,shuffle=True,drop_last=True)
test_data_load = DataLoader(dataset=test_data_set,batch_size=64,shuffle=True,drop_last=True)

train_data_size=len(train_data_set)
test_data_size=len(test_data_set)
print(f'the train size is {train_data_size}')
print(f'the test size is {test_data_size}')
class Mynet(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.BatchNorm2d(num_features=32),

        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.BatchNorm2d(num_features=64),

        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.BatchNorm2d(128),
    )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256,10),
        )

    def forward(self,x):
        return self.fc(self.main(x))
mynet = Mynet()
mynet = mynet.to(device)
print(mynet)
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)
#
learning_rate = 1e-3
optim = torch.optim.Adam(mynet.parameters(),lr = learning_rate)
train_step = 0
epoch = 20

#
if __name__ == '__main__':
    for i in range(epoch):
        print(f'第{i+1}轮训练begin')
        mynet.train()
        for j , (imgs,targets) in enumerate(train_data_load):
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = mynet(imgs)
            loss = loss_fn(outputs,targets)
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_step += 1
            if train_step % 100 == 0:
                print(f'THE {train_step} TIMES TO TRAIN,LOSS = {loss}')
        mynet.eval()
        accuracy = 0
        accuracy_total = 0
        with torch.no_grad():
            for j, (imgs,targets) in enumerate(test_data_load):
                imgs = imgs.to(device)
                targets = targets.to(device)
                outputs = mynet(imgs)
                accuracy = (outputs.argmax(axis=1)==targets).sum()
                accuracy_total += accuracy
            print(f'the {i+1} times train, the 准确率 is {accuracy_total/test_data_size}')
            torch.save(mynet,f'CIAFR_10_{i+1}_acc_{accuracy_total/test_data_size}.pth')


