import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch import autograd
use_GPU = torch.cuda.is_available()

fps = 8
time = 3 #second
joints = 18
dimension = 2
person=2
classes=8
input_shape=[2,fps*time,joints*2]

latent_dim=8
DIM=8

def get_TD(x):
    x_TD=x.clone()
    for i in range(1,x.size()[2]):
        x_TD[:,:,i,:]=x[:,:,i,:]-x[:,:,i-1,:]
    size=list(x.size())
    size.remove(size[2])
    x_TD[:,:,0,:]=Variable(torch.zeros(size))
    return x_TD

class Discriminator_simple(nn.Module):
    def __init__(self,in_dim=input_shape,person=person,out=classes+1):
        super(Discriminator_simple,self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(72,48),
            nn.LeakyReLU(),
        )
        self.SPFL=nn.Sequential(
            nn.Conv2d(1,32,(3,3),padding=(1,0)),
            nn.LeakyReLU(),
            nn.Conv2d(32,64,(3,3),stride=2,padding=(1,0)),
            nn.LeakyReLU(),
            nn.Conv2d(64,128,(3,3),stride=2),
            nn.LeakyReLU()
        )
        self.fc2=nn.Sequential(
            nn.Linear(5*10*128,512),
            nn.LeakyReLU(),
            nn.Dropout(0.55)
        )
        self.D=nn.Linear(512,1)
        self.C=nn.Linear(512,out)
            
    def forward(self,x):
        x = self.fc(x.view(-1,1,24,72))
        x = self.SPFL(x)
        x = self.fc2(x.view(-1,128*5*10))
        return self.D(x),F.softmax(self.C(x),dim=1)

class CNN_interaction(nn.Module):
    def __init__(self,in_dim=input_shape,person=person,out=classes+1):
        super(CNN_interaction,self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(72,48),
            nn.LeakyReLU(),
            nn.Dropout(0.3)
        )
        self.SPFL=nn.Sequential(
            nn.Conv2d(1,32,(3,3),padding=(1,0)),
            nn.LeakyReLU(),
            nn.Conv2d(32,64,(3,3),stride=2,padding=(1,0)),
            nn.LeakyReLU(),
            nn.Conv2d(64,128,(3,3),stride=2),
            nn.LeakyReLU()
        )
        self.fc2=nn.Sequential(
            nn.Linear(5*10*128,512),
            nn.LeakyReLU(),
            nn.Dropout(0.55)
        )
        self.D=nn.Linear(512,1)
        self.C=nn.Linear(512,out)
            
    def forward(self,x):
        x = self.fc(x.view(-1,1,24,72))
        x = self.SPFL(x)
        x = self.fc2(x.view(-1,128*5*10))
        return F.log_softmax(self.C(x),dim=1)
