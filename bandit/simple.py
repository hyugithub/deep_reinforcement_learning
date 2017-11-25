#test git checkin capability
#test git checkin capability AGAIN

# configuration of actions
import numpy as np
import config

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

    def update(self, reward):             
#        if self.count % Estimator.snapshot_step2 == 0:        
        if self.count in self.steps:
            for a in range(len(self.cnt_reward)):
                self.snapshot.append((self.count,a,self.cnt_reward[a]/self.cnt_trial[a]))
        self.count += 1
        self.cnt_reward[action]+= reward
        self.cnt_trial[action] += 1  
    
    def get(self):
        return self.cnt_reward/self.cnt_trial
    
    def history(self):
        return self.snapshot
    
#policy class
class Policy:
    def __init__(self, num_action, prob_explore):
        self.num_action = num_action
        self.prob_explore = prob_explore
    
    def get_action(self, reward_estimator):
        if np.random.uniform() >= self.prob_explore:
            #exploitation
            return int(np.argmax(reward_estimator.get()))
        # simple exploration
        return int(np.random.choice(self.num_action))    

# basic simulation configuration
number_actions = config.number_actions
number_try = config.number_try
prob_exploration = config.prob_exploration
np.random.seed(config.initial_seed)

#binary reward
reward_mean = np.floor(np.random.uniform(size=[number_actions])*1000)*0.001
reward_est = Estimator(number_actions)
policy = Policy(number_actions, prob_exploration)

#book keeping
action_frequency = {}

for t in np.arange(number_try):
    # select an action    
    action = policy.get_action(reward_est)
    #print(action)
    
    if action in action_frequency:
        action_frequency[action] += 1
    else:
        action_frequency[action] = 1
    
    # observe reward
    reward = (np.random.uniform() < reward_mean[action])

    # update estimate    
    reward_est.update(reward)    
    
final_estimate = reward_est.get()

for a in sorted(range(len(reward_mean)), key=lambda x: reward_mean[x], reverse=True):
    print(a,action_frequency[a], reward_mean[a])

#for s in reward_est.history():
#    print("step=%d action=%d reward estimate=%.3f"%s)
    
import matplotlib.pyplot as plt

for a in range(len(reward_mean)):
    x = [s[0] for s in reward_est.history() if s[1] == a]
    y = [s[2] for s in reward_est.history() if s[1] == a]
    plt.plot(x, y, '-', label=str(a))
plt.legend(loc='best')
plt.show()    
    