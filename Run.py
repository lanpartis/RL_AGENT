from Q_agent import DQNAgent
from robot import Cam
from robot import Pepper
import thread
import time
from PIL import Image
import torch
from torchvision.transforms import Compose,ToTensor,Resize
import numpy as np
Image_size=[240,240]
ip='192.168.1.121'
robot = Pepper.Pepper(ip=ip)
cam = Cam.Cam()
Q_Agent = DQNAgent()
ep_file='RL_DATA/EP.data'
time_unit = 3
camera = Cam.fishcam
actions=robot.action_list
ep = int(open(ep_file).read())
time_stamp = 0
def trans(imgs):
    res=list()
    for img in imgs:
        img=Image.fromarray(img)
        img=img.resize(Image_size)
        img=np.array(img)
        res.append(img)
    res=np.stack(res)
    return res
def to_tensor(imgs):
    res = list()
    for img in imgs:
        img = Image.fromarray(img)
        img = ToTensor()(img)
        res.append(img)
    return torch.cat(res,dim=0)

def proc_state(state):
    s_tensor=[]
    for s in state:
        s_tensor.append(trans(s))
    s_tensor=torch.cat(s_tensor,dim=0)
    return s_tensor
def resample(state):
    return state[-24:]
robot.perform_act('wait')
state = cam.get_state(camera,time_unit)
state = trans(resample(state))

while True:
    s = time.time()
    state_tensor = to_tensor(state)
    act_num = Q_Agent.get_action_eps(state_tensor)
    print 'action is %s'%actions[act_num]
    time.sleep(1)
    thread.start_new_thread(robot.perform_act,(actions[act_num],))
    time.sleep(0.3)
    n_state = cam.get_state(camera,time_unit)
    n_state = trans(resample(n_state))
    Q_Agent.memorize(state,act_num,n_state,ep,time_stamp)
    state = n_state
    time_stamp+=1
    print time.time()-s
