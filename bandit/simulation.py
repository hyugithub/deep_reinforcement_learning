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
from scipy.stats import norm

ts = time.time()

#in verbose model, all debugging information is printed
verbose = False

# basic simulation configuration
np.random.seed(config.initial_seed)
# when is the optimal policy stable
optimal_since = []
# at this point, does statistical significance hold
error_prob = []

#for run in range(config.num_simulation):
for run in range(10000):
    if verbose:
        print("\n starting run =", run)
    number_actions = config.number_actions
    number_try = config.number_try
    prob_exploration = config.prob_exploration    
    
    #binary reward
    env = rl.Environment(number_actions)
    #reward_est = rl.Estimator(number_actions)
    policy = rl.Policy(number_actions, prob_exploration)
    
    reward_action_history = []
       
    #sequential events
    for t in np.arange(number_try):        
        
        # select an action    
        action, explore = policy.get_action()
        #print(action)
        
        # observe reward
        reward = env.get_reward(action)
        
        reward_action_history.append((action,reward))
   
        # learning
        policy.learn(action, explore, reward)    
          
    #for s in reward_est.history():
    #    print("step=%d action=%d reward estimate=%.3f"%s)
    
    optimal_since.append(policy.get_learner().optimal_since())    
    
    #calculate probability optimal policy is better than others
    opt = policy.get_learner().optimal_action()
    threshold = max(policy.get_learner().optimal_since(),1)
    reward_action_history = reward_action_history[0:threshold]
    
    hist_opt = [r for a,r in reward_action_history if a == opt]
    mean_opt = (config.reward_bayesian_prior+sum(hist_opt)) \
               / (config.trial_bayesian_prior+len(hist_opt))
    var_opt = (1.0-mean_opt)*mean_opt \
               /(config.trial_bayesian_prior+len(hist_opt))

    if verbose:
        print("optimal action = %d"%opt)
        print("the action is optimal since %d"%threshold)         
        print("recommendation = ", len(hist_opt)," reward= ", sum(hist_opt))
        print("estimate for optimal action mean=%.3f"%mean_opt
              , " std=%.3f"%np.sqrt(var_opt))

    prob_error = 0    
    for alt in range(number_actions):
        if alt != opt:
            hist_alt = [r for a,r in reward_action_history if a == alt]
            # mean and variance derived from binomial distribution
            mean_alt = (config.reward_bayesian_prior+sum(hist_alt)) \
                       / (config.trial_bayesian_prior+len(hist_alt))
            var_alt = (1.0-mean_alt)*mean_opt \
                       /(config.trial_bayesian_prior+len(hist_alt))            
                       
            # if by this time data is extremely sparse
            # essentially we don't have data to determine
            # the chance of error is 50/50
            prob_error = 0.5
            
            if mean_opt > 1e-8 and mean_opt < 0.9999999 and var_opt > 1e-8 \
                and mean_alt > 1e-8 and mean_alt < 0.9999999 \
                and var_alt > 1e-8:
                # assuming an (approximated) normal distribution
                # we can calculate probability for opt >= alt  
                prob_error = max(0, norm.cdf(0.0
                                             , loc=mean_opt-mean_alt
                                             , scale=np.sqrt(var_opt+var_alt)))
            if verbose:
                print("alt action ="
                      , alt
                      , " Pr(alt>opt) = %.3f" % prob_error)                      
                print("estimate for alt action mean=%.3f"%mean_alt
                      , " std=%.3f"%np.sqrt(var_alt))
    error_prob.append(prob_error)
    
    if verbose:
        final_estimate = policy.get_learner().get()
    
        for a in sorted(range(len(env.reward()))
                        , key=lambda x: env.reward()[x]
                        , reverse=True):
            print(a,policy.get_action_frequency(a), env.reward()[a])        
        print("optimal action = %d"%policy.get_learner().optimal_action())
        print("the action is optimal since %d"
              %policy.get_learner().optimal_since())
        for a in range(number_actions):
            x,y = zip(*[(s[0],s[2]) for s in policy.get_learner().history() 
                                if s[1] == a])
            #y = [s[2] for s in reward_est.history() if s[1] == a]
            plt.plot(x, y, '-', label=str(a))
        plt.legend(loc='best')
        plt.show()    

#print(optimal_since)

print("policy converge mean = %d" % np.mean(optimal_since)
        , "std = %d" %np.std(optimal_since)
        , "median= %d" % np.median(optimal_since))
print(np.percentile(optimal_since, np.arange(start=0,stop=100,step=10)))
plt.hist(optimal_since, bins='auto')
plt.gca().set_xscale("log")
plt.show()

np.set_printoptions(precision=3, suppress=True)
print("distribution of error probabiilty when optimal")
print("mean = %.3f" % np.mean(error_prob)
         , "std = %.3f" % np.std(error_prob)
         , "median = %.3f" % np.median(error_prob))

print("percentile:")
print(np.percentile(error_prob, np.arange(start=0,stop=100,step=10)))

print("programming running time = %.3f"%(time.time()-ts))
