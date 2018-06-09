import random
from NN_models.Q_model import DQN as Q_model
from torch.optim import Adam
from torch.autograd import Variable
import torch
import thread
import os
import numpy as np
from PIL import Image
import glob
use_GPU = False
PILMODE='L'
class DQNAgent:
    model_path='DQN_models'
    memory_path='RL_DATA/'
    state_file=memory_path+'STATE'
    action_file=memory_path+'ACTION'
    ep_reward_file=memory_path+'ep_reward.dat'
    batch_size = 25
    epsilon = 1
    epsilon_decay = 0.99
    epsilon_final = 0.1
    epsilon_endtime = 30000
    action_size = 5
    discount_factor = 0.7
    n_replay = 1 #replay per learning step
    learn_start = 3000
    replay_memory = 30000
    clip_delta = 1

    def __init__(self,episode=0):
        self.ep = episode
        if episode==0:
            self.F_model = Q_model()
            self.target_F_model = Q_model()
        else:
            self.load_model(episode)
        if use_GPU:
            self.F_model.cuda()
            self.target_F_model.cuda()
#         self.Y_optimizer = torch.optim.Adam(self.Y_model.parameters(),1e-4)
    
    def get_action(self,state):
        nsize = [1,]
        nsize.extend(state.shape)
        state = state.reshape(nsize)
        state = Variable(torch.Tensor(state))
        state = torch.Tensor(state)
        if use_GPU:
            state = state.cuda()
        res = self.Y_model(state)
        _,act = res.max()
        return max
    def get_action_eps(self,state):
        if random.random() <= self.epsilon:
            #if self.epsilon > self.epsilon_final:
                #self.epsilon*=self.epsilon_decay
            return random.randint(0,self.action_size-1)
        return self.get_action(state)
            
    
    def memorize(self,state,action,n_state,ep,ts):
        thread.start_new_thread(save_state,(state,ep,ts))
        thread.start_new_thread(save_action,(action,ep,ts))

    def save_model(self,tag=None):
        name='episode_%d_model'%self.ep
        if tag is not None:
            name += tag
        torch.save(self.target_F_model,name+'.model')
        
    def load_model(self):
        name='episode_%d_model'%(self.ep-1)
        self.F_model = torch.load(name)
        self.target_F_model=torch.load(name)
    
    def update_target_model(self):
        self.target_F_model.load_state_dict(self.F_model.state_dict())
        
    
def save_state(state,ep,time_stamp):
    s_dir='RL_DATA/EP%d/STATE/s%04d'%(ep,time_stamp)
    if not os.path.isdir(s_dir):
        os.makedirs(s_dir)
    for i in range(state.shape[0]):
        frame=Image.fromarray(state[i],PILMODE)
        frame.save(s_dir+'/%04d.png'%i)
def save_action(action,ep,time_stamp):
    s_dir='RL_DATA/EP%d/ACTION'%ep
    if not os.path.isdir(s_dir):
        os.makedirs(s_dir)
    actfil=open(s_dir+'/a%04d.data'%time_stamp,'w')
    actfil.write(str(action))
    actfil.close()