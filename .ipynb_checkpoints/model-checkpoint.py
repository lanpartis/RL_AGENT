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
class Discriminator(nn.Module):
    def __init__(self,in_dim=input_shape,person=person,out=classes+1):
        super(Discriminator,self).__init__()
        self.SPFL=nn.Sequential(
            nn.Conv2d(in_dim[0],64,(1,1)),
            nn.LeakyReLU(),
            nn.Conv2d(64,32,(3,1),padding=(1,0)),
            nn.LeakyReLU()
        )
        self.SPFL_TD=nn.Sequential(
            nn.Conv2d(2,64,(1,1)),
            nn.LeakyReLU(),
            nn.Conv2d(64,32,(3,1),padding=(1,0)),
            nn.LeakyReLU()
        )
        self.SPFL_2=nn.Sequential(
            nn.Conv2d(18,32,(3,3),padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d((3,3),stride=2),
            nn.Conv2d(32,64,(3,3),padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d((3,3),stride=2)
        )
        self.SPFL_2_TD=nn.Sequential(
            nn.Conv2d(18,32,(3,3),padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d((3,3),stride=2),
            nn.Conv2d(32,64,(3,3),padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d((3,3),stride=2)
        )
        self.subnet2=nn.Sequential(
            nn.Conv2d(128,128,(3,3),padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d((3,3),stride=2),
            nn.Conv2d(128,256,(3,3),padding=1),
            nn.LeakyReLU()
        )
        self.fc=nn.Sequential(
            nn.Linear(2*256*2*3,256),
            nn.Dropout()
        )
        self.D=nn.Linear(256,1)
        self.C=nn.Linear(256,out)
            
    def forward(self,x):
        p1=x[:,:,0]
        p1=p1.permute(0,3,1,2)
        p1_TD = get_TD(p1)
        p2=x[:,:,1]
        p2=p2.permute(0,3,1,2)
        p2_TD = get_TD(p2)
        #subnet one
        p1=self.SPFL(p1)
        p2=self.SPFL(p2)
        p1_TD=self.SPFL_TD(p1_TD)
        p2_TD=self.SPFL_TD(p2_TD)
        p1 = p1.permute(0,3,2,1)
        p2 = p2.permute(0,3,2,1)
        p1_TD = p1_TD.permute(0,3,2,1)
        p2_TD = p2_TD.permute(0,3,2,1)
        p1= self.SPFL_2(p1)
        p2= self.SPFL_2(p2)
        p1_TD= self.SPFL_2_TD(p1_TD)
        p2_TD= self.SPFL_2_TD(p2_TD)
        p1 = torch.cat((p1,p1_TD),dim=1)
        p2=torch.cat((p2,p2_TD),dim=1)
        #subnet two
        res1 = self.subnet2(p1)
        res2 = self.subnet2(p2)
        #concatenate two person
        res = torch.cat((res1,res2),dim=1)
        res=F.leaky_relu(self.fc(res.view(-1,2*256*2*3)))
        D_res=F.sigmoid(self.D(res))
        C_res=F.softmax(self.C(res),dim=1)
        return D_res,C_res
       
def get_TD(x):
    x_TD=x.clone()
    for i in range(1,x.size()[2]):
        x_TD[:,:,i,:]=x[:,:,i,:]-x[:,:,i-1,:]
    size=list(x.size())
    size.remove(size[2])
    x_TD[:,:,0,:]=Variable(torch.zeros(size))
    return x_TD
class Generator(nn.Module):
    def __init__(self,in_dim=latent_dim,out_dim=[fps*time,joints*person*dimension]):
        super(Generator,self).__init__()
        self.fc= nn.Linear(in_dim,36*DIM)
        self.bnorm1 = nn.BatchNorm1d(36*DIM)
        self.ct1 = nn.ConvTranspose2d(DIM,32,4,padding=1,stride=2)
        self.bnorm2 = nn.BatchNorm2d(32)
        self.ct2 = nn.ConvTranspose2d(32,1,4,padding=1,stride=2)
        self.bnorm3 = nn.BatchNorm2d(1)
        self.fc2 = nn.Linear(24,out_dim[1])
        
    
    def forward(self,x):
        x = self.bnorm1(F.relu(self.fc(x)))
        x = x.view(-1,DIM,6,6)
        x = self.bnorm2(F.relu(self.ct1(x)))
        x= self.bnorm3(F.relu(self.ct2(x)))
        x = F.tanh(self.fc2(x))
        return x
def getNoise(batch_size):
    return torch.randn(batch_size,latent_dim)


class Discriminator_simple(nn.Module):
    def __init__(self,in_dim=input_shape,person=person,out=classes+1):
        super(Discriminator_simple,self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(72,48),
            nn.LeakyReLU(),
            nn.Dropout(0.5)
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
            nn.Linear(5*10*128,1024),
            nn.LeakyReLU(),
            nn.Dropout(0.5)
        )
        self.D=nn.Linear(1024,1)
        self.C=nn.Linear(1024,out)
            
    def forward(self,x):
        x = self.fc(x.view(-1,1,24,72))
        x = self.SPFL(x)
        x = self.fc2(x.view(-1,128*5*10))
        return F.sigmoid(self.D(x)),F.softmax(self.C(x),dim=1)

class Discriminator_W(nn.Module):
    def __init__(self,in_dim=input_shape,person=person,out=classes+1):
        super(Discriminator_W,self).__init__()
        self.SPFL=nn.Sequential(
            nn.Conv2d(in_dim[0],64,(1,1)),
            nn.LeakyReLU(),
            nn.Conv2d(64,32,(3,1),padding=(1,0)),
            nn.LeakyReLU()
        )
        self.SPFL_TD=nn.Sequential(
            nn.Conv2d(2,64,(1,1)),
            nn.LeakyReLU(),
            nn.Conv2d(64,32,(3,1),padding=(1,0)),
            nn.LeakyReLU()
        )
        self.SPFL_2=nn.Sequential(
            nn.Conv2d(18,32,(3,3),padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d((3,3),stride=2),
            nn.Conv2d(32,64,(3,3),padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d((3,3),stride=2)
        )
        self.SPFL_2_TD=nn.Sequential(
            nn.Conv2d(18,32,(3,3),padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d((3,3),stride=2),
            nn.Conv2d(32,64,(3,3),padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d((3,3),stride=2)
        )
        self.subnet2=nn.Sequential(
            nn.Conv2d(128,128,(3,3),padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d((3,3),stride=2),
            nn.Conv2d(128,256,(3,3),padding=1),
            nn.LeakyReLU()
        )
        self.fc=nn.Sequential(
            nn.Linear(2*256*2*3,256),
            nn.Dropout()
        )
        self.D=nn.Linear(256,1)
        self.C=nn.Linear(256,out)
            
    def forward(self,x):
        p1=x[:,:,0]
        p1=p1.permute(0,3,1,2)
        p1_TD = get_TD(p1)
        p2=x[:,:,1]
        p2=p2.permute(0,3,1,2)
        p2_TD = get_TD(p2)
        #subnet one
        p1=self.SPFL(p1)
        p2=self.SPFL(p2)
        p1_TD=self.SPFL_TD(p1_TD)
        p2_TD=self.SPFL_TD(p2_TD)
        p1 = p1.permute(0,3,2,1)
        p2 = p2.permute(0,3,2,1)
        p1_TD = p1_TD.permute(0,3,2,1)
        p2_TD = p2_TD.permute(0,3,2,1)
        p1= self.SPFL_2(p1)
        p2= self.SPFL_2(p2)
        p1_TD= self.SPFL_2_TD(p1_TD)
        p2_TD= self.SPFL_2_TD(p2_TD)
        p1 = torch.cat((p1,p1_TD),dim=1)
        p2=torch.cat((p2,p2_TD),dim=1)
        #subnet two
        res1 = self.subnet2(p1)
        res2 = self.subnet2(p2)
        #concatenate two person
        res = torch.cat((res1,res2),dim=1)
        res=F.leaky_relu(self.fc(res.view(-1,2*256*2*3)))
        D_res=self.D(res)
        C_res=F.softmax(self.C(res),dim=1)
        return D_res,C_res