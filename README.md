# Deep Reinforcement Learning

## Introduction

The purpose of this code base is to develop and test algorithms in Reinforcement Learning and Deep Reinforcement Learning. Reinforcement Learning traced its roots back to dynamic programming and Markov Decision Processes, work developed by Bellman et al back in 1950s. It lost momentum when computational challenges such as curse of dimensionality discouraged large-scale applications. Recently, the rapid development in deep learning has motivated people to re-evaluate the potential on reinforcement learning. Basically, what Bertsekas and Tsitsiklis wrote in Neuro-Dynamic Programming becomes achievable. Meanwhile, such new potential and applications puts the fundamental RL architecture (namely the state, the action, the reward, and the environment) to tough test. Warren Powell pointed out that, for example, the state definition is not polished. To examine the flexibility and limitations of RL/DRL algorithms, we focus on marketing, sales, and personalization applications.

## Open Questions

Here we list some open questions in RL to motivate our study:

### Performance of Stochastic Gradient Descent and Bellman Error Minimization
https://www.sigmetrics.org/sigmetrics2017/MI_Jordan_sigmetrics2017.pdf and other related papers

reports-archive.adm.cs.cmu.edu/anon/anon/1999/CMU-CS-99-132.pdf

### Practical Bandit
Convergence; delayed reward; reward mismatch; correlated bandit; strategy

### Contextual Bandit
Thompson sampling

### Holistic state-space estimation and optimization
DQN-like structure

### Forecasting versus Optimization
Supervised learning can be viewed as a prediction problem, with the goal to minimize generalized error on testing dataset. At the same time, Reinforcement Learning is not a prediction problem, but an optimization one. What are the pros and cons when incorporating SL in RL?

### Statistical View on Exploration vs. Exploitation
Discussion on Thompson Sampling, UCB, and more formal treatment.

### Warm Start
