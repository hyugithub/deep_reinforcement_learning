
# coding: utf-8

# In[55]:


# configuration of actions
import numpy as np
np.random.seed(234)

number_actions = 5

#binary reward
reward_mean = np.floor(np.random.uniform(size=[number_actions])*1000)*0.001

# let us also assume reward is bernoulli
# so we need to estimate the probability

cnt_reward = np.full([number_actions],1)
cnt_trial = np.full([number_actions],1)



# In[56]:


class select_action:
    def __init__(self, num_action, prob_explore):
        self.num_action = num_action
        self.prob_explore = prob_explore
    
    def get_action(self, estimate_mean):
        if np.random.uniform() >= self.prob_explore:
            #exploitation
            return int(np.argmax(estimate_mean))
        # simple exploration
        return int(np.random.choice(self.num_action))


# In[60]:


number_try = 10000
prob_exploration = 0.5

sel_action = select_action(number_actions, prob_exploration)

selected = {}

for t in np.arange(number_try):
    # select an action    
    action = sel_action.get_action(cnt_reward/cnt_trial)
    #print(action)
    
    if action in selected:
        selected[action] += 1
    else:
        selected[action] = 1
    
    # observe reward
    reward = (np.random.uniform() < reward_mean[action])
    
    cnt_reward[action]+= reward
    cnt_trial[action] += 1
    
    # update estimate
    
final_estimate = cnt_reward/cnt_trial    

for a in sorted(range(len(reward_mean)), key=lambda x: reward_mean[x]):
    print(a,selected[a], reward_mean[a])

