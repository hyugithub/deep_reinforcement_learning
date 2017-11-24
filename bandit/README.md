# Bandit

In bandit we solve one-step decision making problems. Arguably this is simpler than sequential decision making. But the scale and complexity in real-world bandit problem make things hard. We explore such complexity here.

Even though bandit is not sequential decision making, it provides a benchmark for other algorithms.

We will likely discuss contextual bandit in a separate chapter.

## Key questions

### Scalability 
Performance when number of actions increases

Computational cost

### Rare event 
In many real-world bandit problems, a binary reward happens rarely. For example, on certain websites only 1% of users click the recommended content. So the average reward is very low (0.01). Rare events present additional challenges for RL.

### Convergence
We are interested in two questions: a) how fast does reward estimate converge? b) how fast does optimal control policy converge?

### Correlation

### Non-binary reward 
Continuous distribution or mixture
