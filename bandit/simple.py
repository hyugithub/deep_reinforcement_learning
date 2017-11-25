#test git checkin capability
#test git checkin capability AGAIN

# configuration of actions
import numpy as np
import config
import matplotlib.pyplot as plt
import time

ts = time.time()

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

# basic simulation configuration
np.random.seed(config.initial_seed)
optimal_since = []

for run in range(config.num_simulation):
    number_actions = config.number_actions
    number_try = config.number_try
    prob_exploration = config.prob_exploration    
    
    #binary reward
    env = Environment(number_actions)
    reward_est = Estimator(number_actions)
    policy = Policy(number_actions, prob_exploration)
    
    #book keeping
    for t in np.arange(number_try):
        # select an action    
        action = policy.get_action(reward_est)
        #print(action)
        
        # observe reward
        reward = env.get_reward(action)
    
        # update estimate    
        reward_est.update(reward)    
          
    #for s in reward_est.history():
    #    print("step=%d action=%d reward estimate=%.3f"%s)
    optimal_since.append(reward_est.optimal_since())
        
    if 0:
        final_estimate = reward_est.get()
    
        for a in sorted(range(len(env.reward())), key=lambda x: env.reward()[x], reverse=True):
            print(a,policy.get_action_frequency(a), env.reward()[a])        
        print("optimal policy = %d"%reward_est.optimal_action())
        print("the policy is optimal since %d"%reward_est.optimal_since())
        for a in range(number_actions):
            x,y = zip(*[(s[0],s[2]) for s in reward_est.history() if s[1] == a])
            #y = [s[2] for s in reward_est.history() if s[1] == a]
            plt.plot(x, y, '-', label=str(a))
        plt.legend(loc='best')
        plt.show()    

#print(optimal_since)

print("policy converge mean = %d"%np.mean(optimal_since))
print("policy converge std = %d"%np.std(optimal_since))
plt.hist(optimal_since, bins='auto')
plt.gca().set_xscale("log")
plt.show()
print("programming running time = %.3f"%(time.time()-ts))