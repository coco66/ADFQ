The code will be updated soon!!

Author: \\
Heejin Chloe Jeong* (heejinj@seas.upenn.edu)\\
Clark Zhang (clarkz@seas.upenn.edu)\\
Daniel D. Lee (ddlee@seas.upenn.edu)\\
George J. Pappas (pappasg@seas.upenn.edu)\\

* Assumed Density Filtering Q-learning (https://arxiv.org/abs/1712.03333)

## Requirement 
The ADFQ codes for the finite state and action spaces (directly under the ADFQ directory) work both in python 2.7.x and python 3. For continuous state space, you can use Deep ADFQ (ADFQ/DeepADFQ).\\
Clone the repo
```
git clone https://github.com/coco66/ADFQ.git
```
In order to run DeepADFQ, you need python 3 (>=3.5) and tensorflow-gpu.
We include some codes from OpenAI baselines (due to the repository stability issue, we are not directly using the OpenAI baseline git repo). You may need some packages mentioned in the installation guidelines at https://github.com/openai/baselines. 

You need to export python paths:
```
export PYTHONPATH=PATH_TO_ADFQ:$PYTHONPATH
export PYTHONPATH=PATH_TO_ADFQ/DeepADFQ:$PYTHONPATH
```

## Example for running ADFQ algorithm

```
python run_brl.py --env loop
```
And running ADFQ in Cartpole-v0
```
python run_mlp.py
```
set callback=None in line 78 if you don't want it to end its training after reaching a maximum time step of a task (e.g. 199 for CartPole).\\
Running ADFQ in an atari game, for example, Asterix-v0
```
python run_atari.py --env AsterixNoFrameskip-v4 --act_policy bayesian
```
## Citing
If you use this repo in your research, you can cite it as follows:
```bibtex
@misc{ADFQrepo,
    author = {Heejin Jeong, Clark Zhang, Daniel D. Lee, George J. Pappas},
    title = {ADFQ_public},
    year = {2018},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/coco66/ADFQ.git}},
}

```
