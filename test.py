import os

import torch
import torchvision.transforms
from PIL import Image
from conda.exports import root_dir
from openpyxl.styles.builtins import output
from torch import nn
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

tragets_idx = {
    0 :'飞机',
    1 : 'car',
    2 : 'bird',
    3 : 'cat',
    4 : 'lu',
    5 : 'dog',
    6 : 'qinwa',
    7 : 'house',
    8 : 'aircraft',
    9 : 'firecar'

}
root_dir = 'test_10'
obj_dir = 'test4.png'
img_dir = os.path.join(root_dir,obj_dir)
img = Image.open(img_dir)
tran_pose = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(32,32)),
    torchvision.transforms.ToTensor()
])
mynet = torch.load('CIAFR_10_20_acc_0.8727800250053406.pth',map_location=torch.device('cpu'))
img = tran_pose(img)
img = torch.reshape(img,(1,3,32,32))
output = mynet(img)
print(tragets_idx[output.argmax(axis=1).item()])

