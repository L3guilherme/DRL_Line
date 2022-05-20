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

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout = 0.5):
        super().__init__()

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
    def __init__(self, actor, critic):
        super().__init__()
        
        self.actor = actor
        self.critic = critic
        
    def forward(self, state):
        
        action_pred = self.actor(state)
        value_pred = self.critic(state)
        
        return action_pred, value_pred


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)

def train(env, policy, optimizer, discount_factor, ppo_steps, ppo_clip):
        
    policy.train()
        
    states = []
    actions = []
    log_prob_actions = []
    values = []
    rewards = []
    done = False
    episode_reward = 0

    state = env.reset()

    while not done:

        state = torch.FloatTensor(state).unsqueeze(0)

        #append state here, not after we get the next state from env.step()
        states.append(state)
        
        action_pred, value_pred = policy(state)
                
        action_prob = F.softmax(action_pred, dim = -1)
                
        dist = distributions.Categorical(action_prob)
        
        action = dist.sample()
        
        log_prob_action = dist.log_prob(action)
        
        state, reward, done, perc_prod = env.step(action.item())

        train_env.Plot_Var('info/action',action.item())

        actions.append(action)
        log_prob_actions.append(log_prob_action)
        values.append(value_pred)
        rewards.append(reward)
        
        episode_reward += reward
    
    states = torch.cat(states)
    actions = torch.cat(actions)    
    log_prob_actions = torch.cat(log_prob_actions)
    values = torch.cat(values).squeeze(-1)
    
    returns = calculate_returns(rewards, discount_factor)
    advantages = calculate_advantages(returns, values)
    
    policy_loss, value_loss = update_policy(policy, states, actions, log_prob_actions, advantages, returns, optimizer, ppo_steps, ppo_clip)

    return policy_loss, value_loss, episode_reward,perc_prod


def calculate_returns(rewards, discount_factor, normalize = True):
    
    returns = []
    R = 0
    
    for r in reversed(rewards):
        R = r + R * discount_factor
        returns.insert(0, R)
        
    returns = torch.tensor(returns)
    
    if normalize:
        returns = (returns - returns.mean()) / returns.std()
        
    return returns


def calculate_advantages(returns, values, normalize = True):
    
    advantages = returns - values
    
    if normalize:
        
        advantages = (advantages - advantages.mean()) / advantages.std()
        
    return advantages

def update_policy(policy, states, actions, log_prob_actions, advantages, returns, optimizer, ppo_steps, ppo_clip):
    
    total_policy_loss = 0 
    total_value_loss = 0
    
    advantages = advantages.detach()
    log_prob_actions = log_prob_actions.detach()
    actions = actions.detach()
    
    for _ in range(ppo_steps):
                
        #get new log prob of actions for all input states
        action_pred, value_pred = policy(states)
        value_pred = value_pred.squeeze(-1)
        action_prob = F.softmax(action_pred, dim = -1)
        dist = distributions.Categorical(action_prob)
        
        #new log prob using old actions
        new_log_prob_actions = dist.log_prob(actions)
        
        policy_ratio = (new_log_prob_actions - log_prob_actions).exp()
                
        policy_loss_1 = policy_ratio * advantages
        policy_loss_2 = torch.clamp(policy_ratio, min = 1.0 - ppo_clip, max = 1.0 + ppo_clip) * advantages
        
        policy_loss = - torch.min(policy_loss_1, policy_loss_2).sum()
        
        value_loss = F.smooth_l1_loss(returns, value_pred).sum()
    
        optimizer.zero_grad()

        policy_loss.backward()
        value_loss.backward()

        optimizer.step()
    
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
    
    return total_policy_loss / ppo_steps, total_value_loss / ppo_steps

def evaluate(env, policy):
    
    policy.eval()
    
    rewards = []
    done = False
    episode_reward = 0

    state = env.reset()

    while not done:

        state = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
        
            action_pred, _ = policy(state)

            action_prob = F.softmax(action_pred, dim = -1)
                
        action = torch.argmax(action_prob, dim = -1)
                
        state, reward, done, perc_prod = env.step(action.item())

        episode_reward += reward
        
    return episode_reward,perc_prod


if __name__ == "__main__":

    HIDDEN_DIM = [32,64,128]
    agent_type = 'PPO'
    agent_tech = 'SA'
    n_maq = 3
    MAX_EPISODES = 50
    DISCOUNT_FACTOR = 0.99
    N_TRIALS = 25
    REWARD_THRESHOLD = 93.0
    REWARD_THRESHOLD_EVAL = 80.0
    PRINT_EVERY = 10
    TEST_EVERY = 5
    PPO_STEPS = 5
    PPO_CLIP = 0.2
    LEARNING_RATE = 0.01

    for dim in HIDDEN_DIM:

        current_time = datetime.datetime.now()
        pasta = str(current_time.year)+str(current_time.month)+str(current_time.day)+'_'+str(current_time.hour)+str(current_time.minute)+str(current_time.second)

        newpath = '/media/leandro/EXTRA/DEV/DRL/Resultados/'+agent_type+'/'+agent_tech+'/'+str(n_maq)+'M/'+str(dim)+'/'+pasta
        print(newpath) 
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        run_name = newpath+'/'+agent_type+'_'+agent_tech+'_'+str(dim)+'_'+pasta
        print(run_name)

        train_env = Planta(cfg_file ='line_'+str(n_maq)+'M.cfg',log_dir=run_name+'_T',mode = 0)# 8 ='st', 'wp', 'fp', 'bp', 'wt':, 'sp', 'ot' + buffer Anterior e proximo

        INPUT_DIM = train_env.observation_space.shape[0]
        OUTPUT_DIM = train_env.action_space.n

        actor = MLP(INPUT_DIM, dim, OUTPUT_DIM)
        critic = MLP(INPUT_DIM, dim, 1)

        policy = ActorCritic(actor, critic)

        optimizer = optim.Adam(policy.parameters(), lr = LEARNING_RATE)

        train_rewards = []
        test_rewards = []
        mean_test_rewards = 0.0
        test_perc_prod = []
        mean_test_perc_prod = 0.0
        mean_train_rewards = 0.0
        mean_train_prod =0.0

        for episode in range(1, MAX_EPISODES+1):
            
            policy_loss, value_loss, train_reward,perc_prod_train = train(train_env, policy, optimizer, DISCOUNT_FACTOR, PPO_STEPS, PPO_CLIP)
            mean_train_prod = mean_train_prod + ( (perc_prod_train-mean_train_prod)/episode)
            train_env.Plot_Var('info/policy_loss ',policy_loss)
            train_env.Plot_Var('info/value_loss ',value_loss)

            if episode % TEST_EVERY == 0:#perc_prod_train > REWARD_THRESHOLD_EVAL:
                test_env =  Planta(cfg_file ='line_'+str(n_maq)+'M.cfg',log_dir=run_name+'_E_'+str(episode),mode = 0)
                test_reward,perc_prod_test = evaluate(test_env, policy)
                test_rewards.append(test_reward)
                test_perc_prod.append(perc_prod_test)
                mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])
                mean_test_perc_prod = np.mean(test_perc_prod[-N_TRIALS:])
                test_env.Plot_Var('info/Reward mean',mean_test_rewards)
            
            train_rewards.append(train_reward)
            mean_train_rewards = np.mean(train_rewards[-N_TRIALS:])
            train_env.Plot_Var('info/RE_mean',mean_train_rewards)
            train_env.Plot_Var('info/EPs',episode)
            
            if episode % PRINT_EVERY == 0:
            
                print(f'| Episode: {episode:3} | Mean Train Rewards: {mean_train_rewards:5.1f} | Mean Test Rewards: {mean_test_rewards:5.1f} | Prod: {perc_prod_train:5.1f}')
            
            if mean_test_perc_prod >= REWARD_THRESHOLD:
                
                print(f'| Episode FIM: {episode:3} | Mean Train Rewards: {mean_train_rewards:5.1f} | Mean Test Rewards: {mean_test_rewards:5.1f} | Prod: {mean_test_perc_prod:5.1f}')
                print(f'Reached reward threshold in {episode} episodes')
                torch.save(policy.state_dict(), run_name+'.pt')
                
                break

        resRun = [agent_type,agent_tech,n_maq,dim,mean_train_rewards,train_env.itc_total,episode,mean_train_prod,mean_test_rewards,mean_test_perc_prod,current_time,datetime.datetime.now(),MAX_EPISODES]

            
        fileRes = "Tab_Res.csv"
        with open(fileRes, 'a') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(resRun)
            csvfile.close() 