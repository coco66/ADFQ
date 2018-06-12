Author: 
Heejin Chloe Jeong* (heejinj@seas.upenn.edu)
Clark Zhang (clarkz@seas.upenn.edu)
Daniel D. Lee (ddlee@seas.upenn.edu)

* Assumed Density Filtering Q-learning (https://arxiv.org/abs/1712.03333)

## Requirement 
The ADFQ codes for the finite state and action spaces (directly under the ADFQ directory) work both in python 2.7.x and python 3.
Clone the repo
```
git clone https://github.com/coco66/ADFQ.git
```
In order to run DeepADFQ, you need OpenAI baselines and its requirements -- python 3 (>=3.5) and tensorflow 1.x (>= 1.3).
Visit https://github.com/openai/baselines and install OpenAI baselines following the instruction.
Then, copy the codes in deepq_add to the baselines repository if you would like to use the evaluation methods presented in the paper, Assumed Density Filtering Q-learning:
```
cd ADFQ/DeepADFQ
scp build_graph.py /path/to/baselines/baselines/deepq/
scp simple.py /path/to/baselines/baselines/deepq/
scp run_mlp.py /path/to/baselines/baselines/deepq/experiments/
scp run_atari.py /path/to/baselines/baselines/deepq/experiments/
```

## Example for running ADFQ algorithm

```
python run_brl.py --env loop
```
And running ADFQ in Cartpole-v0
```
python run_mlp.py
```
set callback=None in line 78 if you don't want it to end its training after reaching 199.
Running ADFQ in an atari game, for example, Asterix-v0
```
python run_mlp.py --env AsterixDeterministic-v4 --act_policy bayesian
```

## Citing
If you use this repo in your research, you can cite it as follows:
```bibtex
@misc{ADFQrepo,
    author = {Heejin Jeong, Clark Zhang, Daniel D. Lee},
    title = {ADFQ_public},
    year = {2018},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/coco66/ADFQ.git}},
}



