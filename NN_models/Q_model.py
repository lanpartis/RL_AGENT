import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import autograd
use_GPU = torch.cuda.is_available()

input_shape = [24,480,640]
output_shape=6
units =[16,32,64,256]
ker_size = [9,5,5]
st_size = [3,2,1]
p_s=2 #pooling_size,pooling_strides
fss=4*5 #final state size

class DQN(nn.Module):
    def __init__(self,input_shape=input_shape,output_shape=output_shape):
        super(DQN,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_shape[0],units[0],kernel_size=ker_size[0],stride=st_size[0]),
            # nn.BatchNorm2d(n_s[0]),
            nn.ReLU(),
            nn.MaxPool2d(p_s,p_s),
        )

        self.conv2 =nn.Sequential(
            nn.Conv2d(units[0],units[1],kernel_size=ker_size[1],stride = st_size[1]),
            # nn.BatchNorm2d(n_s[1]),
            nn.ReLU(),
            nn.MaxPool2d(p_s,p_s),
        )
        self.conv3 =nn.Sequential(
            nn.Conv2d(units[1],units[2],kernel_size=ker_size[1],stride = st_size[1]),
            # nn.BatchNorm2d(n_s[2]),
            nn.ReLU(),
            nn.MaxPool2d(p_s,p_s),
        )
        self.L1 = nn.Linear(units[2]*fss,units[3])
        self.drop = nn.Dropout(p=0.5)
        self.out = nn.Linear(units[3],output_shape)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0),-1)
        x = F.relu(self.L1(x))
        x = self.drop(x)
        return self.out(x)