# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 10:07:56 2017

@author: HYU
"""

import numpy as np
import config
import matplotlib.pyplot as plt
import time

class Environment:
    def __init__(self, number_actions):
        self.reward_mean = np.floor(np.random.uniform(size=[number_actions])*1000)*0.001
    
    def get_reward(self, action):
        if np.random.uniform() < self.reward_mean[action]:
            return 1        
        return 0
    
    def reward(self):
        return self.reward_mean

# estimation class
class Estimator:
    snapshot_step2 = 100.0
    #initial_reward = 0.0
    def __init__(self, number_actions):
        self.cnt_reward = np.full([number_actions],1)
        self.cnt_trial = np.full([number_actions],1)
        self.count = 0
        self.snapshot = []
        self.steps = np.floor(np.exp(np.arange(int(np.log(100000)))))
        self.max_time = dict.fromkeys(range(number_actions),0)

    def optimal_action(self):        
        return np.argmax(self.cnt_reward/self.cnt_trial)

    def update(self, reward):             
#        if self.count % Estimator.snapshot_step2 == 0:        
        if self.count in self.steps:
            for a in range(len(self.cnt_reward)):
                self.snapshot.append((self.count,a,self.cnt_reward[a]/self.cnt_trial[a]))
        self.count += 1
        self.cnt_reward[action]+= reward
        self.cnt_trial[action] += 1  
        
        self.max_time[self.optimal_action()] = self.count            
    
    def get(self):
        return self.cnt_reward/self.cnt_trial
    
    def history(self):
        return self.snapshot
    
    def optimal_since(self):
        return max(self.max_time[a] for a in self.max_time if a != self.optimal_action())
    
#policy class
class Policy:
    def __init__(self, num_action, prob_explore):
        self.num_action = num_action
        self.prob_explore = prob_explore
        self.action_frequency = dict.fromkeys(range(num_action),0)
    
    def get_action(self, reward_estimator):
        
        # simple exploration
        action = -1        
        
        if np.random.uniform() >= self.prob_explore:
            #exploitation
            action = int(np.argmax(reward_estimator.get()))
        else:
            action = int(np.random.choice(self.num_action))
        self.action_frequency[action] += 1
        return action
    
    def get_action_frequency(self, action):
        return self.action_frequency[action]