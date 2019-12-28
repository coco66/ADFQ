import baselines0.deepq.run_atari as run_atari
import pickle
import os, pdb
dirpath = 'results/atari/ADFQ-bayes/'
fnames = [a for a in os.listdir(dirpath) if a!='.DS_Store']
for f in fnames:
    ifpass = False
    for file in os.listdir(os.path.join(dirpath,f)):
        if '.png' in file:
            ifpass = True
            break
    if not ifpass:
        print(f)
        x = pickle.load(open(os.path.join(os.path.join(dirpath,f),'records.pkl'),'rb'))
        run_atari.plot(x, os.path.join(dirpath,f))