import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as funct

import random,time

from model import linear_network
from icm_model import Icm

#************************************************************
#************************************************************
class Agent():
    def __init__(self,insize,hidden_size,outsize,randoms,epsilon,gamma):
        # super().__init__()
        self.n_games=0
        self.lrate=.001
        self.max_mem=500000
        self.batch_size=1000
        self.epsilon=epsilon
        self.gamma=gamma
        self.memory=[]
        
        self.insize=insize
        self.hidden=hidden_size
        self.outsize=outsize
        
        
        
        self.model=linear_network(self.insize,self.hidden,self.outsize,self.gamma,self.lrate)
        
        self.max_rands=randoms
        
        self.random_count=0
        self.predicted_count=0
        
        
    def get_ratios(self):
        total=self.random_count+self.predicted_count
        r_rat=round(self.random_count/total,4)
        p_rat=round(self.predicted_count/total,4)
        return 'random {} : predicted {}'.format(r_rat,p_rat)
        
    def reset_ratios(self):
        self.random_count=0
        self.predicted_count=0
        
    def remember(self,state, action, reward,next_state,done):
        info=[state,action,reward,next_state,done]
        if len(self.memory)>=self.max_mem:
            self.memory.pop(0)
        self.memory.append(info)
        
    def train_long_memory(self):
        msize=len(self.memory)
        if msize<self.batch_size:
            batch=self.memory
        
        else:
            start=random.randint(0,msize-self.batch_size)
            batch=self.memory[start:start+self.batch_size]
            
        states,actions,rewards,next_states,dones=zip(*batch)
        
        isize=len(states)
        
        for i in range(isize):
            self.model.train_step([states[i]],[actions[i]],[rewards[i]],[next_states[i]],[dones[i]])
        
        
        
    
    def train_short_memory(self,state,action,reward,next_state,done):
        self.model.train_step([state],[action],[reward],[next_state],[done])
        
    def get_action(self,state):
        
        r_limit=self.max_rands-self.n_games
        
        if r_limit<round(self.max_rands*self.epsilon):
            r_limit=round(self.max_rands*self.epsilon)
        #    
        final_move=torch.FloatTensor([0 for i in range(self.outsize)])  
        
        
        
        
        if random.randint(0,self.max_rands)<r_limit:
            move=random.randint(0,self.outsize-1)
            final_move[move]=1
            self.random_count+=1
            
        else:
            state0=torch.FloatTensor(state).to(self.model.device)
            pred=self.model(state0)
            
            # 
            # for action in (self.outsize):
            #     self.icm.get_intrinsic_reward()
            
            
            move=torch.argmax(pred).item()
            
            final_move=torch.FloatTensor([i.item() for i in pred])
            self.predicted_count+=1
            
            state0.cpu()
            
            
        
        
        return final_move,torch.argmax(final_move).item()
        
        
#************************************************************
#************************************************************        



    
    
    
    
    
    
    
        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    