### switching state space model

#### Environment:
first install ssm
```
git clone https://github.com/lindermanlab/ssm
cd ssm
pip install numpy cython
pip install -e .
```
then install other necessary packages
```
pip install scipy pandas matplotlib
```

#### Important scripts:
```
sssm.py: runs a switching state space model given a config file specifying 
hyperparameters and dataset. calls data_preprocess.py to preprocess data.

simulation_agent.py: runs edge tracking simulation given parameters of a 
switching state space model. calls simulation_envs.py for edge structure.
```

### demo files:
```
'./files/configs/' contains an example config file for sssm.py, it specifies 
the example fly trajectory in './files/et_45/'
```


