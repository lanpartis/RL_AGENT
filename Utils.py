import os
import glob
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import pandas as pd
from sklearn import linear_model 
from xml.etree import ElementTree
from PIL import Image,ImageDraw
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats, signal, interpolate
OpenPose_LABEL = ["Nose","Neck","RShoulder","RElbow","RWrist","LShoulder","LElbow","LWrist","RHip","RKnee","RAnkle","LHip","LKnee","LAnkle","REye","LEye","REar","LEar","Background"]
KINECT_LABEL = ['HEAD','NECK','TORSO','RIGHT_SHOULDER','RIGHT_ELBOW','RIGHT_HAND','LEFT_SHOULDER','LEFT_ELBOW','LEFT_HAND','RIGHT_HIP','RIGHT_KNEE','RIGHT_FOOT','LEFT_HIP','LEFT_KNEE','LEFT_FOOT']
Image_Size=[240,240]
UImage_Size = [736,736]# [Width, Heighth]
fig=plt.gcf()
fig.set_size_inches(20, 10)
ori_categories=['wave_hand',
 'call_people',
 'bow',
 'hand_paper',
 'approach',
 'leave',
 'nod',
 'point']
selected = [0,1,2,3,4,5,6,7]
sel_label=[ori_categories[i] for i in selected]
NUM_OF_EACH=2*52
video_length=24
# data utility
def openpose2kinect(Pair):#[N,Person_id,keypoint_id,x or y]
    for n in range(Pair.shape[1]):
        Person = Pair[:,n]
        d = Person[:,:Person.shape[1]-4,:]
        x = (d[:,1,0]+d[:,8,0]+d[:,11,0])/3.0
        y = (d[:,1,1]+d[:,8,1]+d[:,11,1])/3.0
        TORSO = np.array([x,y]).T
        d1 = np.concatenate([d[:,:2,:],TORSO[:,None,:]],axis=1)
        d = np.concatenate([d1,d[:,2:]],axis=1)
        if 'result' in locals():
            result = np.concatenate([result,d[:,None,:,:]],axis=1)
        else:
            result = d
            result = d[:,None,:,:]
    return result

def reform(data):
    xl=data[:,:54:3]
    yl=data[:,1:55:3]
    xr=data[:,54::3]
    yr=data[:,55::3]
    add_dim = lambda x:x.reshape(x.shape[0],1,-1,1)
    xl=add_dim(xl)
    xr=add_dim(xr)
    yl=add_dim(yl)
    yr=add_dim(yr)
    l = np.concatenate([xl,yl],3)
    r=np.concatenate([xr,yr],3)
    score=data[:,2::3]
    score=score.reshape(score.shape[0],2,-1)
    return np.concatenate((l,r),1),score

def json_read(file):
    f = open(file,"r")
    data = json.load(f)["people"]
    _min,_max = Image_Size[0],0
    p = []
    Center = Image_Size[0]/2.
    nan = float("nan")
    for n in range(len(data)):# decide mait two person
        Neck_x = data[n]["pose_keypoints_2d"][3]
        if Neck_x == 0:
           p.append(nan)
        else:
           p.append(Neck_x)
    try:
        left_id,right_id = p.index(min(p)),p.index(max(p))
        Left_Person,L_Score = reform(data[left_id]["pose_keypoints_2d"])
        Right_Person,R_Score = reform(data[right_id]["pose_keypoints_2d"])
        data = np.concatenate([Left_Person[None,:],Right_Person[None,:]],axis=0)# ["keypoint index:0~14"]["x:0 or y:1"] => ["Left:0 or Right:1"]["keypoint index:0~17"]["x:0 or y:1"]
        score = np.concatenate([L_Score[None,:],R_Score[None,:]],axis=0)
    except:
        data = np.zeros([2,18,2])# In case of no detecting any person by openpose
        score = np.zeros([2,18])
    return data,score

def make_time_data(data,Time=8):# data:(N,feature_num) out:(N,t,feature_num)
        N = len(data)
        data = data.reshape([N,-1])
        Time = 8
        out = [data[s:s+8] for s in range(N-Time+1)]
        out = np.array(out)
        return out
# 描画関連            
def comx(x):
            return float(x)
def comy(y):
            return 1. - (float(y))
def line(X,Y,s,e,f=True):
            idx = int(len(X)/2.)
            if f==True:
               X,Y = [comx(X[s+idx]),comx(X[e+idx])],[comy(Y[s+idx]),comy(Y[e+idx])]
            else:
               X,Y = [comx(X[s]),comx(X[e])],[comy(Y[s]),comy(Y[e])]
            if (0 in X) or (0 in Y):
                return 0
            plt.plot(X, Y,'o-',color=cm.gray(0.2),ms=6,lw=3,mfc='royalblue')

def update(i,X,Y):
            plt.cla()
            plt.xlim([0,2.])
            plt.ylim([0,1.])
            x,y = [],[]
            '''
            if i<8:
               #plt.title(str(n),fontsize=18)
               plt.text(1150,1800,"STOP",fontsize=18)
            else:
               #plt.title("PLAY",fontsize=18)
               plt.text(1150,1800,"PLAY",fontsize=18)
            '''
            position = []
            if int(X.shape[1]/2.) == 15:
                #   0       1      2            3               4            5            6               7           8            9           10           11          12         13          14   
                #['HEAD','NECK','TORSO','RIGHT_SHOULDER','RIGHT_ELBOW','RIGHT_HAND','LEFT_SHOULDER','LEFT_ELBOW','LEFT_HAND','RIGHT_HIP','RIGHT_KNEE','RIGHT_FOOT','LEFT_HIP','LEFT_KNEE','LEFT_FOOT']
                position = [(0,1),(1,2),(3,1),(4,3),(5,4),(6,1),(7,6),(8,7),(9,2),(10,9),(11,10),(12,2),(13,12),(14,13)]
            elif int(X.shape[1]/2.) == 18:
                #    0      1        2          3        4         5          6        7       8      9       10      11      12      13      14     15     16     17                               
                #["Nose","Neck","RShoulder","RElbow","RWrist","LShoulder","LElbow","LWrist","RHip","RKnee","RAnkle","LHip","LKnee","LAnkle","REye","LEye","REar","LEar"]
                position = [(0,1),(1,2),(2,3),(3,4),(1,5),(5,6),(6,7),(1,8),(8,9),(9,10),(1,11),(11,12),(12,13),(0,14),(14,16),(0,15),(15,17)]
            for f in [False,True]:
                for s,e in position:
                    line(X[i],Y[i],s,e,f)
            plt.tick_params(labelbottom='off')
            plt.tick_params(labelleft='off')
            plt.tick_params(color='white')
            
def draw(data,f,test = False,fps=8,ext='.mp4'):# [N,feature points:{P1([x1,...,x15,y1,...,y15]),P2([x1,...,x15,y1,...,y15])}]
            fig = plt.figure()
            idx = int(data.shape[1]/4.)
            #ini = np.ones([8,60])*data[0]
            #data = np.r_[ini,data]
            X,Y = np.c_[data[:,:idx],data[:,2*idx:3*idx]],np.c_[data[:,idx:2*idx],data[:,3*idx:]]
            f = f+ext
            print("saving "+f+"...")
            if ext == '.gif':
                    Writer = animation.writers["imagemagic"]
            else:
                    Writer = animation.writers["ffmpeg"]
            writer = Writer(fps=fps)
            ani = animation.FuncAnimation(fig, update, fargs = (X, Y),blit=False,frames = len(X),interval = int(1000./fps))
            ani.save(f,writer=writer)
            plt.close()

def Pair2draw(Pair,f,fps=8):# [N, person_id, keypoint, x or y] -> [N,feature_num]
            Pair=np.copy(Pair)
            Pair[:,0,:,0]+=0.5
            Pair[:,1,:,0]+=1.5
            Pair[:,0,:,1]+=0.5
            Pair[:,1,:,1]+=0.5
            draw_src = np.concatenate([Pair[:,:,:,0],Pair[:,:,:,1]],axis=2)# [N, person_id, keypoint, x or y] -> [N, person_id, (x0,x1,..x15,y1,...,y15)]
            draw_src = np.concatenate([draw_src[:,0,:],draw_src[:,1,:]],axis=1)
            draw(draw_src,f,fps=fps)
            
# 前処理関連
def normalization(Pair):
    Pair[:,:,:,0],Pair[:,:,:,1] = Pair[:,:,:,0]/Image_Size[0] - 0.5,Pair[:,:,:,1]/UImage_Size[1]-0.5
    return Pair

def replace_outlier(df,under=0.1,top=0.9):
    nan = float("nan")
    #外れ値の基準点
    _min = df.quantile(under)
    _max = df.quantile(top)

    df[df < _min] = nan
    df[df > _max] = nan
    return df

# 4分位数による外れ値除去
def preprocessing(data):
    nan = float("nan")
    for n in range(data.shape[1]):# Number of persons
        P = "A" if n == 0 else "B"
        for k in range(data.shape[2]):# Number of keypoints
            plt.figure()
            fps = 8.
            t = np.arange(0, float(data.shape[0]/fps), 1./fps)
            
            x,y = data[:,n,k,0],data[:,n,k,1]
            sx,sy = "Person"+P+"_"+OpenPose_LABEL[k]+"_X","Person"+P+"_"+OpenPose_LABEL[k]+"_Y"
            df = pd.DataFrame({sx:x,sy:y})
            
            #　前処理
            df = df.where(df > 0., nan)
            df[sx] = replace_outlier(df[sx],under=0.1,top=0.9)
            df[sy] = replace_outlier(df[sy],under=0.1,top=0.9)
            
            # 線形補間
            kind = "linear"
            df = df.interpolate(method=kind, axis=0).bfill()# 補間を実行
            x,y = np.array(df[sx]),np.array(df[sy])
            
            #plt.plot(t,data[:,n,k,0])
            if not "Person" in locals():
                Person = np.concatenate([x[:,None,None,None],y[:,None,None,None]],axis=3)
                #plt.plot(t,Person[:,0,0,0])
            else:
                _Person = np.concatenate([x[:,None,None,None],y[:,None,None,None]],axis=3)
                Person = np.concatenate([Person,_Person],axis=2)
                #plt.plot(t,Person[:,0,k,0])
        if not "Pair" in locals():
            Pair = Person
        else:
            Pair = np.concatenate([Pair,Person],axis=1)
        del Person
    return Pair

# score による外れ値除去
def preprocessing2(data,score,criteria=0.05):
    score = np.array(score)
    nan = float("nan")
    for n in range(data.shape[1]):# Number of persons
        P = "A" if n == 0 else "B"
        for k in range(data.shape[2]):# Number of keypoints
            plt.figure()
            fps = 24.
            t = np.arange(0, float(data.shape[0]/fps), 1./fps)
            
            x,y = data[:,n,k,0],data[:,n,k,1]
            sx,sy = "Person"+P+"_"+OpenPose_LABEL[k]+"_X","Person"+P+"_"+OpenPose_LABEL[k]+"_Y"
            df = pd.DataFrame({sx:x,sy:y})
            
            #　前処理
            df = df.where(df > 0., nan)
            idx = np.where(score[:,n,k] < criteria)
            for i in idx:
                df[sx][i],df[sy][i] = nan,nan
            #print(df.isnull().sum()/len(data)*100)# display ratio of outvalue
            # 線形補間
            kind = "linear"
            df = df.interpolate(method=kind, axis=0)# 補間を実行
            x,y = np.array(df[sx]),np.array(df[sy])
            
            #plt.plot(t,data[:,n,k,0])
            if not "Person" in locals():
                Person = np.concatenate([x[:,None,None,None],y[:,None,None,None]],axis=3)
                #plt.plot(t,Person[:,0,0,0])
            else:
                _Person = np.concatenate([x[:,None,None,None],y[:,None,None,None]],axis=3)
                Person = np.concatenate([Person,_Person],axis=2)
                #plt.plot(t,Person[:,0,k,0])
        if not "Pair" in locals():
            Pair = Person
        else:
            Pair = np.concatenate([Pair,Person],axis=1)
        del Person
    return Pair


#data augmentation
# selected label

def smoothing_filter(seq):
    for j in range(3,seq.shape[0]-2):
#         -3 12 17 12 -3
        seq[j]=(-3*seq[j-2] + 12*seq[j-1] +17*seq[j] +12*seq[j+1] -3*seq[j+2])/35
    return seq
# selected label

def shape_data(dataset):#from(N,72) to (N,2,18,2)
    n_dataset=[]
    for data in dataset:
        ldata=data[:,:36]
        rdata=data[:,36:]
        ldata=ldata.reshape(ldata.shape[0],1,18,2)
        rdata=rdata.reshape(rdata.shape[0],1,18,2)
        n_dataset.append(np.concatenate([ldata,rdata],1))
    return n_dataset
        
def sel_train_data(dataset,labels,selected=[3,4,5,7]):
    n_dataset=[]
    n_labels=[]
    for i in range(len(dataset)):
        if labels[i] in selected:
            n_dataset.append(dataset[i])
            n_labels.append(selected.index(labels[i]))
    return n_dataset,n_labels


def data_balance(dataset,labels,sel_label=sel_label,size=NUM_OF_EACH,mix=False):
    count={}
    for i in labels:
        count[i] = count.get(i,0)+1
    highest = max(count.values())
    ex_dataset=[]
    ex_labels=[]
    l=len(sel_label)
    if mix:
        l+=1
    for i in range(l):
        ex_dataset.extend(infuse_data(dataset,labels,i,size))
        ex_labels.extend(np.ones(size)*i)
    return ex_dataset,ex_labels
    
def infuse_data(dataset,labels,label,total):
    n_dataset=[]
    idx = [i for i in range(len(labels)) if labels[i] == label]
    start=0
    while True:
        for i in idx:
            n_data,res=video_clip(dataset[i],start=start)
            if res == None:
                continue
            n_dataset.append(n_data)
            total-=1
            if total ==0:
                return n_dataset
        start+=1
            
def video_clip(data,length=video_length,start=0):
    full = len(data)
    if start and start+length>full:
        return None,None
    return data[start:start+length],start
def rand_clip(data,length=video_length,used=[]):
    full = len(data)
    start=np.random.randint(0,full-length+1)
    if len(used) >= full-length:
        return None,None
    while(start in used):
        start=np.random.randint(0,full-length+1)
    return data[start:start+length],start

def add_mirror_data(dataset,labels):
    size=len(dataset)
    for i in range(size):
        dataset.append(people_exchange(dataset[i]))
        labels.append(labels[i])
    return dataset,labels

def people_exchange(data):
    N_data = np.copy(data)
    N_data[:,0]=data[:,1]
    N_data[:,1]=data[:,0]
    return N_data

def make_unlabeled_dataset(unlabeled_dataset,size=5000):
    n_dataset=[]
    while True:
        for i in range(len(unlabeled_dataset)):
            n_data,start=rand_clip(unlabeled_dataset[i])
            if start == None:
                continue
            n_dataset.append(n_data)
            size-=1
            if size ==0:
                return n_dataset

def data_augment(dataset,labels,nolabels,size=NUM_OF_EACH,sel=True,mix=False):
    if sel:
        dataset,labels=sel_train_data(dataset,labels)
    if mix:
        dataset,labels=mix_with_unlabeled(dataset,labels,nolabels)
    for i in dataset:
        smoothing_filter(i)
    train_data,train_labels,test_data,test_labels=data_split(dataset,labels)
    train_data,train_labels=data_balance(train_data,train_labels,size=16*6,mix=mix)
    test_data,test_labels=data_balance(test_data,test_labels,size=16*2,mix=mix)
    train_data,train_labels=add_mirror_data(train_data,train_labels)
    test_data,test_labels=add_mirror_data(test_data,test_labels)
    return train_data,train_labels,test_data,test_labels
def data_split(dataset,labels,split=0.7):
    train_data=[]
    train_labels=[]
    test_data=[]
    test_labels=[]
    count={}
    for i in labels:
        count[i]=count.get(i,0)+1
    for key in list(count.keys()):
        idx = [i for i in range(len(labels)) if labels[i] == key]
        for i in range(int(len(idx)*split)):
            train_data.append(dataset[idx[i]])
            train_labels.append(labels[idx[i]])
        for i in range(int(len(idx)*split),len(idx)):
            test_data.append(dataset[idx[i]])
            test_labels.append(labels[idx[i]])
            
    return train_data,train_labels,test_data,test_labels  
def mix_with_unlabeled(dataset,labels,nolabels):
    unlabeled=[]
    for i in nolabels:
        clipnum=int(len(i)/32)
        for j in range(clipnum):
            unlabeled.append(i[j*32:(j+1)*32])
    n_labels=[len(sel_label) for i in range(len(unlabeled))]
    dataset.extend(unlabeled)
    labels.extend(n_labels)
    return dataset,labels