import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions

import matplotlib.pyplot as plt
import numpy as np
from planta import Planta

import os
import datetime 
import csv

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout = 0.5):
        super(MLP,self).__init__()

        self.fc_1 = nn.Linear(input_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc_1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc_2(x)
        return x

class ActorCritic(nn.Module):
    def __init__(self, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM,DISCOUNT_FACTOR):
        super(ActorCritic,self).__init__()

        self.actor = MLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
        self.critic = MLP(INPUT_DIM, HIDDEN_DIM, 1)

        self.values = []
        self.rewards = []
        self.actions = []
        self.states = []
        self.log_prob_actions = []

        self.apply(init_weights)

        LEARNING_RATE = 0.01

        self.optimizer = optim.Adam(self.parameters(), lr = LEARNING_RATE)

        PPO_STEPS = 5
        PPO_CLIP = 0.2

        self.ppo_steps = PPO_STEPS
        self.ppo_clip = PPO_CLIP

        self.discount_factor = DISCOUNT_FACTOR
        
    def forward(self, state):
        
        action_pred = self.actor(state)
        value_pred = self.critic(state)
        
        return action_pred, value_pred

    def reset(self):
        self.values = []
        self.rewards = []
        self.actions = []
        self.states = []
        self.log_prob_actions = []

    def calculate_returns(self,rewards, discount_factor, normalize = True):
    
        returns = []
        R = 0
        
        for r in reversed(rewards):
            R = r + R * discount_factor
            returns.insert(0, R)
            
        returns = torch.tensor(returns)
        
        if normalize:
            returns = (returns - returns.mean()) / returns.std()
            
        return returns

    def calculate_advantages(self,returns, values, normalize = True):
        
        advantages = returns - values
        
        if normalize:
            
            advantages = (advantages - advantages.mean()) / advantages.std()
            
        return advantages

    def update_policy(self, states, actions, log_prob_actions, advantages, returns):
    
        total_policy_loss = 0 
        total_value_loss = 0
        
        advantages = advantages.detach()
        log_prob_actions = log_prob_actions.detach()
        actions = actions.detach()
        
        for _ in range(self.ppo_steps):
                    
            #get new log prob of actions for all input states
            action_pred, value_pred = self(states)
            value_pred = value_pred.squeeze(-1)
            action_prob = F.softmax(action_pred, dim = -1)
            dist = distributions.Categorical(action_prob)
            
            #new log prob using old actions
            new_log_prob_actions = dist.log_prob(actions)
            
            policy_ratio = (new_log_prob_actions - log_prob_actions).exp()
                    
            policy_loss_1 = policy_ratio * advantages
            policy_loss_2 = torch.clamp(policy_ratio, min = 1.0 - self.ppo_clip, max = 1.0 + self.ppo_clip) * advantages
            
            policy_loss = - torch.min(policy_loss_1, policy_loss_2).sum()
            
            value_loss = F.smooth_l1_loss(returns, value_pred).sum()
        
            self.optimizer.zero_grad()

            policy_loss.backward()
            value_loss.backward()

            self.optimizer.step()
        
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
        
        return total_policy_loss / self.ppo_steps, total_value_loss / self.ppo_steps

    def update_train(self,device = torch.device('cpu')):

        states = torch.cat(self.states).to(device)
        actions = torch.cat(self.actions).to(device)    
        log_prob_actions = torch.cat(self.log_prob_actions).to(device)
        values = torch.cat(self.values).squeeze(-1).to(device)
        
        returns = self.calculate_returns(self.rewards, self.discount_factor).to(device)
        advantages = self.calculate_advantages(returns, values).to(device)
        
        policy_loss, value_loss = self.update_policy(states, actions, log_prob_actions, advantages, returns)

        return policy_loss, value_loss




def train(env, policy,device = torch.device('cpu')):
        
    done = False
    episode_reward = 0

    state,_ = env.reset()
    n_agentes = len(policy)
    for i_ag in range(n_agentes):
        policy[i_ag].reset()
        policy[i_ag].train()

    while not done:

        state = torch.FloatTensor(state).unsqueeze(0).to(device)

        arr_ac = []
        for i_ag in range(n_agentes):
            policy[i_ag].states.append(state) #append state here, not after we get the next state from env.step()

            action_pred, value_pred = policy[i_ag](state)     
            action_prob = F.softmax(action_pred, dim = -1)
            action_pred.to(device)
            value_pred.to(device)
            action_prob.to(device)

            dist = distributions.Categorical(action_prob)
        
            action = dist.sample()
        
            log_prob_action = (dist.log_prob(action))

            arr_ac.append(action.item())

            policy[i_ag].actions.append(action)
            policy[i_ag].log_prob_actions.append(log_prob_action)
            policy[i_ag].values.append(value_pred)
        
        state, reward,rc_loc, done, perc_prod = env.step(arr_ac)

        for i_ag in range(n_agentes):  
            policy[i_ag].rewards.append(rc_loc[i_ag])
        
        episode_reward += reward

    policy_loss = []
    value_loss = []
    for i_ag in range(n_agentes):
        policy_loss.append(0.0)
        value_loss.append(0.0)
        policy_loss[i_ag], value_loss[i_ag] = policy[i_ag].update_train(device)

    return policy_loss, value_loss, episode_reward, perc_prod



def evaluate(env, policy):
    
    rewards = []
    done = False
    episode_reward = 0

    n_agentes = len(policy)
    for i_ag in range(n_agentes):
        policy[i_ag].to(torch.device('cpu'))
        policy[i_ag].reset()
        policy[i_ag].eval()

    state,_= env.reset()

    while not done:

        state = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            arr_ac = []
            for i_ag in range(n_agentes):
                action_pred, _ = policy[i_ag](state)

                action_prob = F.softmax(action_pred, dim = -1)
                
                action = torch.argmax(action_prob, dim = -1)

                arr_ac.append(action.item())
                
        state, reward,_, done, prod_test = env.step(arr_ac)

        episode_reward += reward
        
    return episode_reward, prod_test


if __name__ == "__main__":


    HIDDEN_DIM = [32,64,128]
    agent_type = 'PPO'
    agent_tech = 'MA_1p5BP_newS'
    n_maq = 2
    MAX_EPISODES = 50
    N_TRIALS = 25
    REWARD_THRESHOLD = 95.0
    REWARD_THRESHOLD_EVAL = 80.0
    PRINT_EVERY = 10
    DISCOUNT_FACTOR = 0.99
    TEST_EVERY = 5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Rodando em '+str(device))

    for dim in HIDDEN_DIM:
        current_time = datetime.datetime.now()
        pasta = str(current_time.year)+str(current_time.month)+str(current_time.day)+'_'+str(current_time.hour)+str(current_time.minute)+str(current_time.second)

        newpath = '/media/leandro/EXTRA/DEV/DRL/Resultados/'+agent_type+'/'+agent_tech+'/'+str(n_maq)+'M/'+str(dim)+'/'+pasta
        print(newpath) 
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        run_name = newpath+'/'+agent_type+'_'+agent_tech+'_'+str(dim)+'_'+pasta
        print(run_name)
        train_env = Planta(cfg_file ='line_'+str(n_maq)+'M.cfg',log_dir=run_name+'_T',mode = 1)# 8 ='st', 'wp', 'fp', 'bp', 'wt':, 'sp', 'ot' + buffer Anterior e proximo
        #test_env =  Planta(cfg_file ='line_'+str(n_maq)+'M.cfg',log_dir=run_name+'_E',mode = 1)

        INPUT_DIM = train_env.observation_space.shape[0]
        OUTPUT_DIM = 3#train_env.action_space.n
        

        n_agentes = train_env.n_maquinas

        policy = []
        for i_ag in range(n_agentes):
            policy.append(ActorCritic(INPUT_DIM, dim, OUTPUT_DIM,DISCOUNT_FACTOR).to(device))


        train_rewards = []
        test_rewards = []
        mean_test_rewards = 0
        test_perc_prod = []
        mean_test_perc_prod = 0
        mean_train_prod = 0.0

        for episode in range(1, MAX_EPISODES+1):
            
            policy_loss, value_loss, train_reward, perc_prod_train = train(train_env, policy,device)
            mean_train_prod = mean_train_prod + ( (perc_prod_train-mean_train_prod)/episode)

            for i_ag in range(n_agentes):
                train_env.Plot_Var('info/policy_loss_'+str(i_ag),policy_loss[i_ag])
                train_env.Plot_Var('info/value_loss_'+str(i_ag),value_loss[i_ag])

            if episode % TEST_EVERY == 0:#perc_prod_train > REWARD_THRESHOLD_EVAL: # a cada mil testar, arquivos distintos, sem th rw min
                    test_env =  Planta(cfg_file ='line_'+str(n_maq)+'M.cfg',log_dir=run_name+'_E_'+str(episode),mode = 1)
                    test_reward,perc_prod_test = evaluate(test_env,policy)
                    test_rewards.append(test_reward)
                    test_perc_prod.append(perc_prod_test)
                    mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])
                    mean_test_perc_prod = np.mean(test_perc_prod[-N_TRIALS:])
                    test_env.Plot_Var('info/Reward mean',mean_test_rewards)
                    for i_ag in range(n_agentes):
                        policy[i_ag].to(device)
            
            
            train_rewards.append(train_reward)
            mean_train_rewards = np.mean(train_rewards[-N_TRIALS:])
            train_env.Plot_Var('info/RE_mean',mean_train_rewards)
            
            if episode % PRINT_EVERY == 0:
                print(f'| Episode: {episode:3} | Mean Train Rewards: {mean_train_rewards:5.1f} | Mean Test Rewards: {mean_test_rewards:5.1f} | Prod: {perc_prod_train:5.1f}')
            
            if mean_test_perc_prod >= REWARD_THRESHOLD:
                print(f'| Episode FIM: {episode:3} | Mean Train Rewards: {mean_train_rewards:5.1f} | Mean Test Rewards: {mean_test_rewards:5.1f} | Prod: {mean_test_perc_prod:5.1f}')
                print(f'Reached reward threshold in {episode} episodes')
                for i_ag in range(n_agentes):
                    torch.save(policy[i_ag].state_dict(), run_name+'_95'+str(i_ag)+'.pt')
                #break

        for i_ag in range(n_agentes):
            torch.save(policy[i_ag].state_dict(), run_name+'_'+str(i_ag)+'.pt')

        resRun = [agent_type,agent_tech,n_maq,dim,mean_train_rewards,mean_train_prod,mean_test_rewards,mean_test_perc_prod,train_env.itc_total,\
                  str(current_time)[:-7],str(datetime.datetime.now())[:-7],run_name,episode,MAX_EPISODES]

            
        fileRes = "Tab_Res.csv"
        with open(fileRes, 'a') as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(resRun)
            csvfile.close()