import time
import os
import sys
sys.path.append("../include")
sys.path.append("../src")

import numpy             as np
import matplotlib        as mpl
import matplotlib.pyplot as plt
import pandas            as pd

from   math              import sqrt
from   scipy.io          import savemat, loadmat
from   scipy.stats       import iqr
from   scipy.integrate   import simps
from   pearson           import logpearson, dlogpearson
from   load_dataset      import load_data
from   optim             import bfgs_log_kernel, bfgs_log_kernel_w1
import numpy as np
import doce

# define the doce environnment

# define the experiment
experiment = doce.Experiment(
  name = 'timbre-metric-learning',
  purpose = 'Timbre study',
  author = 'Mathieu Lagrange',
  address = 'mathieu.lagrange@ls2n.fr',
)
# set acces paths (here only storage is needed)
experiment.set_path('output', '../../../drive/experiments/data/'+experiment.name+'/')

# set the plan (factor : modalities)
experiment.add_plan('plan',
  dataset = ['Grey1977', 'Grey1978', 'Iverson1993_Whole', 'Iverson1993_Onset', 'Iverson1993_Remainder', 'McAdams1995', 'Lakatos2000_Harm', 'Lakatos2000_Perc', 'Lakatos2000_Comb', 'Barthet2010', 'Patil2012_A3', 'Patil2012_DX4', 'Patil2012_GD4', 'Siedenburg2016_e2set1', 'Siedenburg2016_e2set2', 'Siedenburg2016_e2set3', 'Siedenburg2016_e3'],
  embedding  = ['strf', 'stft', 'spectrum', 'scattering', 'clap', 'encodec', 'mert', 'mertcat'],
  method = ['direct', 'learn'],
  warm = [0, 1]
)

experiment.add_plan('mert',
  dataset = ['Grey1977', 'Grey1978', 'Iverson1993_Whole', 'Iverson1993_Onset', 'Iverson1993_Remainder', 'McAdams1995', 'Lakatos2000_Harm', 'Lakatos2000_Perc', 'Lakatos2000_Comb', 'Barthet2010', 'Patil2012_A3', 'Patil2012_DX4', 'Patil2012_GD4', 'Siedenburg2016_e2set1', 'Siedenburg2016_e2set2', 'Siedenburg2016_e2set3', 'Siedenburg2016_e3'],
  embedding  = ['mertcut'],
  index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
  method = ['direct', 'learn'],
  warm = [0, 1]
)

experiment._plan.default('warm', 1)

experiment.set_metric(
  name = 'pearson',
  percent=True,
  higher_the_better= True,
  significance = True,
  precision = 10
  )

def step(setting, experiment):
 
    if setting.embedding == 'mertcut':
        emb = 'mertcat'
    else:
        emb = setting.embedding
 
    r,D,d   = load_data(setting.dataset,emb, '../')
    
    if setting.embedding == 'mertcut':
        r = r[setting.index*768:(setting.index+1)*768]

    weight = np.ones(r.shape[0])
    if setting.method == 'learn':
        if setting.warm:
            opt     = bfgs_log_kernel_w1(r,d)
        else:
            opt     = bfgs_log_kernel(r,d)
        weight = opt.x

    pearson = (-logpearson(weight,r,d))**2

    np.save(experiment.path.output+setting.identifier()+experiment.metric_delimiter+'weight.npy', weight)
    np.save(experiment.path.output+setting.identifier()+experiment.metric_delimiter+'pearson.npy', pearson)
 
# invoke the command line management of the doce package
if __name__ == "__main__":
  doce.cli.main(experiment = experiment,
                func = step
                )
