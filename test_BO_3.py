#Import Modules

#GPyOpt - Cases are important, for some reason
import GPyOpt
from GPyOpt.methods import BayesianOptimization

#numpy
import numpy as np
from numpy.random import multivariate_normal #For later example

import pandas as pd

#Plotting tools
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from numpy.random import multivariate_normal


#Define the objective function
def obj_func(x):
    out = x**4 + 2*x**3 -12*x**2 - 2*x + 6
    return(out)



#Plot the function 
x = pd.Series(np.linspace(-5,4,1000))
f_x = pd.Series.apply(x, obj_func)

plt.plot(x, f_x, 'b-')
plt.show()


domain = [{'name': 'var_1', 'type': 'continuous', 'domain': (-5,4)}]


myBopt_1d = BayesianOptimization(f=obj_func, domain=domain)
myBopt_1d.run_optimization(max_iter=5)
myBopt_1d.plot_acquisition()