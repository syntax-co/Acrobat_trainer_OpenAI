import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as funct

from model import linear_network,ICM_network,inverse_ICM

import random

class Icm():
    def __init__(self,input_size,hidden_size,output_size,gamma,learning_rate):
        
        self.icm=ICM_network(input_size+1,hidden_size,input_size,gamma,learning_rate)
        self.inverse_ICM=inverse_ICM(input_size*2,hidden_size,output_size,gamma,learning_rate)
        
        self.memory=[]
        self.stored=0
        self.max_memory=500000
        self.batch_size=1000
        
    
    def remember(self,state,action,next_state):
        self.memory.append([state,action,next_state])
        self.stored+=1
        if self.stored>self.max_memory:
            self.memory=self.memory[self.stored-self.max_memory:self.stored]
        
    
    def convert_states(self,states): #tuple/list
        state_holder=[[k for k in i] for i in states]
        return state_holder
    
    
    
    def get_intrinsic_reward(self,state,action,new_state):
        
        
        cstates=self.convert_states([state,new_state])
        current_state=cstates[0]
        next_state=torch.FloatTensor(cstates[1]).to(self.icm.device)
        
        current_state+=[torch.argmax(action).item()]
        current_state=torch.FloatTensor(current_state).to(self.icm.device)
        
        icm_run=self.icm.forward(current_state)
        
        intrinsic_reward=self.icm.criterion(icm_run,next_state)
        
        
        next_state.to('cpu')
        current_state.to('cpu')
        
        torch.cuda.empty_cache()
        
        return intrinsic_reward
        
    
    def get_invserse(self,state,new_state):
        cstates=self.convert_states([state,new_state])
        inverse_input=torch.FloatTensor(cstates[0]+cstates[1]).to(self.inverse_ICM.device)
        inverse_run=self.inverse_ICM(inverse_input)
        
        if self.icm.device=='cuda' or self.inverse_ICM.device=='cuda':
            inverse_input.to('cpu')
            torch.cuda.empty_cache()
        
        return inverse_run
        
        
    def train_long_memory(self):
        
        if self.stored<self.batch_size:
            batch=self.memory
            size=self.stored
        else:
            start=random.randint(0,self.stored-self.batch_size)
            print(start,start+self.batch_size)
            batch=self.memory[start:start+self.batch_size]
            size=self.batch_size
        
        states,actions,new_states=zip(*batch)
        
        
        
        for i in range(size):
            self.train_models(states[i],actions[i],new_states[i])
    
    
        
    def train_models(self,state,action,next_state):
        dex=torch.argmax(action)
        
        cstates=self.convert_states([state,next_state])
        conv_state=cstates[0]
        conv_nstate=cstates[1]
        
        
        icm_input=conv_state.copy()
        icm_input+=[dex]
        icm_input=torch.FloatTensor(icm_input).to(self.icm.device)
        inverse_input=torch.FloatTensor(conv_state+conv_nstate).to(self.inverse_ICM.device)
        
        
        predicted_state=self.icm.forward(icm_input)
        predicted_action=self.inverse_ICM.forward(inverse_input)
        # print(action)
        actual_state=torch.FloatTensor(conv_nstate).to(self.icm.device)
        actual_action=torch.FloatTensor(action).to(self.inverse_ICM.device)
        
        
        self.icm.optimizer.zero_grad()
        self.inverse_ICM.optimizer.zero_grad()
        
        icm_reward=self.icm.criterion(predicted_state,actual_state)
        inverse_loss=self.inverse_ICM.criterion(predicted_action,actual_action)
        
        icm_reward.backward()
        inverse_loss.backward()
        
        self.icm.optimizer.step()
        self.inverse_ICM.optimizer.step()
        
        
        #tensor removal
        
        if self.icm.device=='cuda' or self.inverse_ICM.device=='cuda':
            icm_input.to('cpu')
            inverse_input.to('cpu')
            actual_state.to('cpu')
            actual_action.to('cpu')
            
            torch.cuda.empty_cache()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    