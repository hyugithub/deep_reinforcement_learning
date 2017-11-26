# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 11:10:59 2017

@author: HYU
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import component_simple as rl
import config


ts = time.time()

# basic simulation configuration
np.random.seed(config.initial_seed)
optimal_since = []

for run in range(config.num_simulation):
    number_actions = config.number_actions
    number_try = config.number_try
    prob_exploration = config.prob_exploration    
    
    #binary reward
    env = rl.Environment(number_actions)
    #reward_est = rl.Estimator(number_actions)
    policy = rl.Policy(number_actions, prob_exploration)
    
    #sequential events
    for t in np.arange(number_try):
        # select an action    
        action, explore = policy.get_action()
        #print(action)
        
        # observe reward
        reward = env.get_reward(action)
    
        # learning
        policy.learn(action, explore, reward)    
          
    #for s in reward_est.history():
    #    print("step=%d action=%d reward estimate=%.3f"%s)
    optimal_since.append(policy.get_learner().optimal_since())
        
    if 0:
        final_estimate = policy.get_learner().get()
    
        for a in sorted(range(len(env.reward())), key=lambda x: env.reward()[x], reverse=True):
            print(a,policy.get_action_frequency(a), env.reward()[a])        
        print("optimal policy = %d"%policy.get_learner().optimal_action())
        print("the policy is optimal since %d"%policy.get_learner().optimal_since())
        for a in range(number_actions):
            x,y = zip(*[(s[0],s[2]) for s in policy.get_learner().history() if s[1] == a])
            #y = [s[2] for s in reward_est.history() if s[1] == a]
            plt.plot(x, y, '-', label=str(a))
        plt.legend(loc='best')
        plt.show()    

#print(optimal_since)

if 1:
    print("policy converge mean = %d"%np.mean(optimal_since))
    print("policy converge std = %d"%np.std(optimal_since))
    plt.hist(optimal_since, bins='auto')
    plt.gca().set_xscale("log")
    plt.show()

print("programming running time = %.3f"%(time.time()-ts))
