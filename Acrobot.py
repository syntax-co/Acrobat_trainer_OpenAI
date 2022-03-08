import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as funct

from live_graph import Live_graph
import multiprocessing as mp


from agent import Agent
from icm_model import Icm

import gym
import numpy as np


import time

game=gym.make('Acrobot-v1')

input_size=game.observation_space.shape[0]
output_size=game.action_space.n
gamma=.9
epsilon=.1

agent=Agent(input_size,200,output_size,20,epsilon,gamma)
icm=Icm(input_size,200,output_size,gamma,.1)



# lgraph=Live_graph()
# lgraph.add_new_object('icm_model')


action_skips=2


past_rewards=[]
past_size=20


while True:
    state=np.array(game.reset())
    total_rewards=0
    
    for i in range(10000):

        game.render()
        action,dex=agent.get_action(state)

        new_state,reward,done,_ = game.step(dex)
        
        intrinsic_reward=icm.get_intrinsic_reward(state,action,new_state)
        
        total_rewards+=reward
        
        reward=reward+intrinsic_reward.item()
        
        
        
        

        # time.sleep(.05)


        if done or i == 500:
            # lgraph.add_new_info(['icm_model'],[reward])
            # total_rewards-=5

            # reward=-1

            agent.n_games+=1


            agent.remember(state,action,reward,new_state,done)
            agent.train_short_memory(state,action,reward,new_state,done)
            
            icm.remember(state,action,new_state)
            icm.train_models(state,action,new_state)

            past_rewards.append(total_rewards)
            if len(past_rewards)>past_size:
                past_rewards.pop(0)

            print('*'*50)
            print(agent.get_ratios(),'- game#: ',agent.n_games, '- total reward: ',total_rewards)
            print('average ',round(sum(past_rewards)/len(past_rewards),4),'-','reward: ',reward)
            print('chosen: ',dex,'- actions: ',[i.item() for i in action])
            print('*'*50)

            agent.reset_ratios()
            break

        else:

            if i%action_skips==0:
                agent.remember(state,action,reward,new_state,done)
                agent.train_short_memory(state,action,reward,new_state,done)
                
                icm.remember(state,action,new_state)
                icm.train_models(state,action,new_state)
                
                

            state=new_state

    agent.train_long_memory()
    # icm.train_long_memory()

    if agent.n_games>=10000:
        break




game.close() 














