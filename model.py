import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as funct





class inverse_ICM(nn.Module):
    def __init__(self,insize,hidden_size,outsize,gamma,lr):
        super().__init__()
        
        self.device='cuda'
        
        self.inlayer=nn.Linear(insize,hidden_size).to(self.device)
        self.hid_layer1=nn.Linear(hidden_size,hidden_size).to(self.device)
        self.hid_layer2=nn.Linear(hidden_size,hidden_size).to(self.device)
        self.hid_layer3=nn.Linear(hidden_size,hidden_size).to(self.device)
        self.hid_layer4=nn.Linear(hidden_size,hidden_size).to(self.device)
        self.outlayer=nn.Linear(hidden_size,outsize).to(self.device)
        
        
        
        self.lr=lr
        self.gamma=gamma
        
        self.optimizer=optim.Adam(self.parameters(),lr=self.lr)
        self.criterion=nn.MSELoss().to(self.device)
    
    
        
    def forward(self,x):
        
        x=torch.relu(self.inlayer(x))
        x=torch.relu(self.hid_layer1(x))
        x=torch.relu(self.hid_layer2(x))
        x=torch.relu(self.hid_layer3(x))
        x=torch.relu(self.hid_layer4(x))
        
        x=self.outlayer(x)
        return x



class ICM_network(nn.Module):
    def __init__(self,insize,hidden_size,outsize,gamma,lr):
        super().__init__()
        
        self.device='cuda'
        
        self.inlayer=nn.Linear(insize,hidden_size).to(self.device)
        self.hid_layer1=nn.Linear(hidden_size,hidden_size).to(self.device)
        self.hid_layer2=nn.Linear(hidden_size,hidden_size).to(self.device)
        self.hid_layer3=nn.Linear(hidden_size,hidden_size).to(self.device)
        self.hid_layer4=nn.Linear(hidden_size,hidden_size).to(self.device)
        self.outlayer=nn.Linear(hidden_size,outsize).to(self.device)
        
        
        
        self.lr=lr
        self.gamma=gamma
        
        self.optimizer=optim.Adam(self.parameters(),lr=self.lr)
        self.criterion=nn.MSELoss().to(self.device)
    
    
        
    def forward(self,x):
        
        x=torch.relu(self.inlayer(x))
        x=torch.relu(self.hid_layer1(x))
        x=torch.relu(self.hid_layer2(x))
        x=torch.relu(self.hid_layer3(x))
        x=torch.relu(self.hid_layer4(x))
        
        x=self.outlayer(x)
        return x
    
    
    


class linear_network(nn.Module):
    def __init__(self,insize,hidden_size,outsize,gamma,lr):
        super().__init__()
        
        self.device='cuda'
        
        self.inlayer=nn.Linear(insize,hidden_size).to(self.device)
        self.hid_layer1=nn.Linear(hidden_size,hidden_size).to(self.device)
        self.hid_layer2=nn.Linear(hidden_size,hidden_size).to(self.device)
        self.hid_layer3=nn.Linear(hidden_size,hidden_size).to(self.device)
        self.hid_layer4=nn.Linear(hidden_size,hidden_size).to(self.device)
        self.outlayer=nn.Linear(hidden_size,outsize).to(self.device)
        
        self.lr=lr
        self.gamma=gamma
        
        self.optimizer=optim.Adam(self.parameters(),lr=self.lr)
        self.criterion=nn.MSELoss().to(self.device)
        
        
        
    def forward(self,x):
        
        x=torch.relu(self.inlayer(x))
        # x=funct.dropout(x,p=0.1)
        
        x=torch.relu(self.hid_layer1(x))
        # x=funct.dropout(x,p=.1)
        
        x=torch.relu(self.hid_layer2(x))
        # x=funct.dropout(x,p=0.1)
        
        x=torch.relu(self.hid_layer3(x))
        # x=funct.dropout(x,p=0.1)
        
        x=torch.relu(self.hid_layer4(x))
        # x=funct.dropout(x,p=0.1)
        
        
        x=self.outlayer(x)
        return x
        
    def train_step(self,state,action,reward,state_new,done):
        train_size=len(state)
        
        
        state=torch.FloatTensor(state).to(self.device)
        next_state=torch.FloatTensor(state_new).to(self.device)
        action=action[0].to(self.device)
        reward=torch.FloatTensor(reward).to(self.device)
        done=done[0]
        
        
        
        prediction=self.forward(state)[0]
        
        target=torch.FloatTensor([i.item() for i in prediction]).to(self.device)
        
        
        
        if done:
            q_new=reward
        else:
            q_new=reward+(self.gamma*torch.max(self.forward(next_state)))
            
            
        target[torch.argmax(action).item()]=q_new
        
        self.optimizer.zero_grad()
        loss=self.criterion(prediction,target)
        loss.backward()
        
        self.optimizer.step()
        
        
        #%%%%%%%%%%%%%%%%%%%%%%%
        # memory remover
        if self.device=='cuda':
            action.to('cpu')
            state.to('cpu')
            next_state.to('cpu')
            action.to('cpu')
            reward.to('cpu')
            target.to('cpu')
        
            torch.cuda.empty_cache()
        

        
    
        
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        