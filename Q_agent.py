import random
from NN_models.Q_model import DQN as Q_model
from torch.optim import Adam
from torch.autograd import Variable
from torch import nn
import torch
import thread
import os
import json
import numpy as np
from PIL import Image
import glob
use_GPU = torch.cuda.is_available()
PILMODE='L'
Image_Size=[240,240]
input_shape = [24,240,240]
actionsize=6
loss_func=nn.MSELoss()
if use_GPU:
    loss_func = loss_func.cuda()
class DQNAgent:
    model_path='DQN_models'
    memory_path='RL_DATA/'
    state_file=memory_path+'STATE'
    action_file=memory_path+'ACTION'
    batch_size = 25
    epsilon = 1
    epsilon_decay = 0.99
    epsilon_final = 0.1
    epsilon_endtime = 30000
    action_size = 6
    discount_factor = 0.9
    n_replay = 1 #replay per learning step
    clip_delta = 1

    def __init__(self,episode=1):
        self.ep = episode
        if episode==1:
            self.F_model = Q_model(output_shape=actionsize)
            self.target_F_model = Q_model(output_shape=actionsize)
        else:
            self.load_model(episode)
        if use_GPU:
            self.F_model.cuda()
            self.target_F_model.cuda()
        self.F_optimizer = Adam(self.F_model.parameters(),1e-4)
    
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
        
    def experience_replay(self,N=10,batchsize=25):
        memory = load_memory_of_episode(self.ep)
        for i in range(int(N*len(memory)/batchsize)):
            minibatch = random.sample(memory,batchsize)
            shape=[batchsize,]
            shape.extend(input_shape)
            update_input = np.zeros(shape)
            update_target = np.zeros((batchsize, self.action_size))
            for j in range(batchsize):
                state = minibatch[j]['state']
                action = int(minibatch[j]['action'])
                reward = minibatch[j]['reward']
                n_state = minibatch[j]['n_state']
                
                ystate = Variable(torch.FloatTensor(state))
                nstate = Variable(torch.FloatTensor(n_state))
                nshape=[1,]
                nshape.extend(ystate.size())
                ystate = ystate.view(nshape)
                nstate = nstate.view(nshape)
                if use_GPU:
                    ystate = ystate.cuda()
                    nstate = nstate.cuda()
                target = self.F_model.forward(ystate)[0].cpu().data.numpy()
                q_2 =self.target_F_model.forward(nstate)
                q_2_max = torch.max(q_2).cpu().data.numpy()
                target[action] = reward + self.discount_factor*q_2_max

                update_input[i] = state
                update_target[i] = target
            
            update_input=Variable(torch.FloatTensor(update_input))
            update_target=Variable(torch.FloatTensor(update_target))
            if use_GPU:
                update_input=update_input.cuda()
                update_target=update_target.cuda()
            prediction=self.F_model.forward(update_input)
            loss = loss_func(prediction,update_target)

            self.F_model.zero_grad()
            loss.backward()
            
            for param in self.F_model.parameters():
                param.grad.data.clamp_(-1,1)
            self.F_optimizer.step()
        self.update_target_model()
    
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