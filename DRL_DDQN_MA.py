import math, random

from planta import Planta
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F

from IPython.display import clear_output
import matplotlib.pyplot as plt

from collections import deque

import os
import datetime
import csv


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
            
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done
    
    def __len__(self):
        return len(self.buffer)

class DuelingDQN(nn.Module):
    def __init__(self, num_inputs, num_outputs,NN_size = 32):
        super(DuelingDQN, self).__init__()

        self.NN_size = NN_size
        self.num_outputs = num_outputs
        
        
        self.feature = nn.Sequential(
            nn.Linear(num_inputs, self.NN_size),
            nn.ReLU()
        )
        
        self.advantage = nn.Sequential(
            nn.Linear(self.NN_size, self.NN_size),
            nn.ReLU(),
            nn.Linear(self.NN_size, num_outputs)
        )
        
        self.value = nn.Sequential(
            nn.Linear(self.NN_size, self.NN_size),
            nn.ReLU(),
            nn.Linear(self.NN_size, 1)
        )
        
    def forward(self, x):
        x = self.feature(x)
        advantage = self.advantage(x)
        value     = self.value(x)
        return value + advantage  - advantage.mean()
    
    def act(self, state, epsilon):
        if random.random() > epsilon:
            state   = torch.FloatTensor(state).unsqueeze(0)
            q_value = self.forward(state)
            action  = q_value.max(1)[1].data[0]
            action = action.item()
        else:
            action = random.randrange(self.num_outputs)
        return action

    def get_act(self,state):
        state   = torch.FloatTensor(state).unsqueeze(0)
        q_value = self.forward(state)
        action  = q_value.max(1)[1].data[0]
        action = action.item()
        return action

class DDQN_Agent:
    def __init__(self, num_inputs, num_outputs,NN_size = 32,batch_size =32):

        self.buffer_replay = ReplayBuffer(5000)
        self.current_model = DuelingDQN(num_inputs, 3,NN_size)
        self.target_model  = DuelingDQN(num_inputs, 3,NN_size) # env.observation_space.shape[0]
        self.optimizer     = optim.Adam(self.current_model.parameters())
        self.batch_size    = batch_size

    def train(self):
        self.current_model.train()
        self.target_model.train()

    def eval(self):
        self.current_model.eval()
        self.target_model.eval()


    def compute_td_loss(self):
       state, action, reward, next_state, done = self.buffer_replay.sample(self.batch_size)

       state      = (torch.FloatTensor(np.float32(state)))
       next_state = (torch.FloatTensor(np.float32(next_state)))
       action     = (torch.LongTensor(action))
       reward     = (torch.FloatTensor(reward))
       done       = (torch.FloatTensor(done))

       q_values      = self.current_model(state)
       next_q_values = self.target_model(next_state)

       q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
       next_q_value     = next_q_values.max(1)[0]
       expected_q_value = reward + gamma * next_q_value * (1 - done)
    
       loss = (q_value - expected_q_value.detach()).pow(2).mean()
        
       self.optimizer.zero_grad()
       loss.backward()
       self.optimizer.step()
    
       return loss

    def act(self,state, epsilon):
        return self.current_model.act(state, epsilon)

    def get_act(self,state):
        return self.current_model.get_act(state)

    def update_target(self):
        self.target_model.load_state_dict(self.current_model.state_dict())

def evaluate(env, policy):

    n_agentes = len(policy)
    for i_ag in range(n_agentes):
        policy[i_ag].eval()
    
    rewards = []
    done = False
    episode_reward = 0

    state,_ = env.reset()

    while not done:
        arr_ac = []
        for i_ag in range(n_agentes):
            action = policy[i_ag].get_act(state)
            arr_ac.append(action)
                
        state, reward,_, done, prod_test = env.step(arr_ac)

        episode_reward += reward
        
    return episode_reward,prod_test


if __name__ == "__main__":

    HIDDEN_DIM = [32,64,128]
    agent_type = 'DDQN'
    agent_tech = 'MA'
    n_maq = 2
    MAX_EPISODES = 50
    N_TRIALS = 25
    REWARD_THRESHOLD =95.0
    REWARD_THRESHOLD_EVAL = 80.0
    PRINT_EVERY = 10
    TEST_EVERY = 5
    gamma = 0.9

    for dim in HIDDEN_DIM:

        current_time = datetime.datetime.now()
        pasta = str(current_time.year)+str(current_time.month)+str(current_time.day)+'_'+str(current_time.hour)+str(current_time.minute)+str(current_time.second)

        newpath = '/media/leandro/EXTRA/DEV/DRL/Resultados/'+agent_type+'/'+agent_tech+'/'+str(n_maq)+'M/'+str(dim)+'/'+pasta
        print(newpath) 
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        run_name = newpath+'/'+agent_type+'_'+agent_tech+'_'+str(dim)+'_'+pasta
        print(run_name)
        train_env =  Planta(cfg_file ='line_'+str(n_maq)+'M.cfg',log_dir=run_name+'_T',mode=1)
        #envE = Planta(cfg_file ='line_'+str(n_maq)+'M.cfg',log_dir=run_name+'_E',mode=1)

        epsilon_start = 1.0
        epsilon_final = 0.01
        epsilon_decay = 500

        epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

        n_agentes = train_env.n_maquinas

        agentes = []
        batch_size = 32
        for i_ag in range(n_agentes):
            agentes.append(DDQN_Agent(train_env.observation_space.shape[0], 3,dim,batch_size))

    #if USE_CUDA:
    #    current_model = current_model.cuda()
    #    target_model  = target_model.cuda()

        train_rewards = []
        test_rewards = []
        mean_test_rewards = 0
        train_reward = 0

        episode_reward = 0

        state,_ = train_env.reset()
        cont_eps = 0

        test_perc_prod = []
        mean_test_perc_prod = 0.0
        mean_train_prod = 0.0

        for episode in range(1, MAX_EPISODES+1):

            state,_ = train_env.reset()
            episode_reward = 0
            done = False

            for i_ag in range(n_agentes):
                agentes[i_ag].train()

            while not done:

                epsilon = epsilon_by_frame(cont_eps/10.0)
                arr_ac = []
                for i_ag in range(n_agentes):
                    action = agentes[i_ag].act(state, epsilon)
                    arr_ac.append(action)

                next_state, reward,rc_loc, done, perc_prod_train = train_env.step(arr_ac)
                mean_train_prod = mean_train_prod + ( (perc_prod_train-mean_train_prod)/episode)

                for i_ag in range(n_agentes):
                    agentes[i_ag].buffer_replay.push(state, arr_ac[i_ag], rc_loc[i_ag], next_state, done)

                state = next_state
                episode_reward += reward
                cont_eps+=1

                if len(agentes[0].buffer_replay) > batch_size:
                    for i_ag in range(n_agentes):
                        loss = agentes[i_ag].compute_td_loss()
                        train_env.Plot_Var('info/value_loss_'+str(i_ag),loss.item())    
                if (cont_eps) % 100 == 0:
                    for i_ag in range(n_agentes):
                        agentes[i_ag].update_target()

            train_reward =episode_reward

            train_rewards.append(train_reward)
            mean_train_rewards = np.mean(train_rewards[-N_TRIALS:])

            train_env.Plot_Var('info/Reward mean',mean_train_rewards)
            train_env.Plot_Var('info/EPs',episode)

            if episode % TEST_EVERY == 0:#perc_prod_train > REWARD_THRESHOLD_EVAL:
                test_env =  Planta(cfg_file ='line_'+str(n_maq)+'M.cfg',log_dir=run_name+'_E_'+str(episode),mode = 1)
                test_reward, perc_prod_test= evaluate(test_env, agentes)
                test_rewards.append(test_reward)
                test_perc_prod.append(perc_prod_test)
                mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])
                mean_test_perc_prod = np.mean(test_perc_prod[-N_TRIALS:])
                test_env.Plot_Var('info/Reward mean',mean_test_rewards)

            if episode % PRINT_EVERY == 0:
                print(f'| Episode: {episode:3} | Mean Train Rewards: {mean_train_rewards:5.1f} | Mean Test Rewards: {mean_test_rewards:5.1f} | Prod: {perc_prod_train:5.1f}')

            if mean_test_perc_prod >= REWARD_THRESHOLD:
                print(f'| Episode FIM: {episode:3} | Mean Train Rewards: {mean_train_rewards:5.1f} | Mean Test Rewards: {mean_test_rewards:5.1f} | Prod: {mean_test_perc_prod:5.1f}')
                print(f'Reached reward threshold in {episode} episodes')
                for i_ag in range(n_agentes):
                    torch.save(agentes[i_ag].current_model.state_dict(), run_name+'_'+str(i_ag)+'.pt')
                break
                  #ALG       TIP        maq   TAM MRW                INT                 EPS     PTR             TRW               PTE                 DTI          DTF
        resRun = [agent_type,agent_tech,n_maq,dim,mean_train_rewards,train_env.itc_total,episode,mean_train_prod,mean_test_rewards,mean_test_perc_prod,current_time,datetime.datetime.now(),run_name,MAX_EPISODES]

            
        fileRes = "Tab_Res.csv"
        with open(fileRes, 'a') as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(resRun)
            csvfile.close()