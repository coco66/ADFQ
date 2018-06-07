Author: Heejin Chloe Jeong (heejinj@seas.upenn.edu)

Affiliation: University of Pennsylvania, Philadelphia, PA

Here I share some of my publically available codes for ADFQ.

* Bayesian Q-learning with Assumed Density Filtering (https://arxiv.org/abs/1712.03333)
	- NIPS workshop on Advances on Approximate Bayesian Inference, NIPS 2017, Long Beach, CA
	- AAAI Spring Symposium Series, Data-Efficient Reinforcement Learning, Palo Alto, CA 

## Example for running ADFQ algorithm
In order to run ADFQ algorithm, for example,
```
import brl
scene = 'loop'
loop = brl.adfq(scene, discount = 0.9, means = 0.0, variances = 100.0, init_policy=True)
loop.learning('Approx','egreedy',0.05, eval_greedy=True) # ADFQ-Approx algorithm with epsilon greedy with epsilon 0.05. Semi-evaluation during learning.
total_cumulative_reward = loop.get_total_reward()
import matplotlib.pyplot as plt
plt.plot(loop.Q_err); plt.show() # To see RMSE of Q values
plt.plot([x[1] for x in loop.eval]) # To see Averaged cumulative reward during learning
```

## Citing
If you use this repo in your research, you can cite it as follows:
```bibtex
@misc{ADFQrepo,
    author = {Heejin Jeong},
    title = {ADFQ_public},
    year = {2018},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/coco66/ADFQ_public.git}},
}



