# Bayesian Optimization For Robot Chemistry Lab
This repository contains implementations of bayesian optimization, data pre-processing, and dimension reduction as functions or classes in Python. The goals are to provide the following implementations such that they are suitable for both individual use and automation in a robot chemistry lab setting:
1. Function to preprocess data collected by the Rolfes Group.
2. Single-Objective and Multi-Objective optimizers using Bayesian Optimization.
3. Dimension reduction implementation which is compatible with the output of the pre-processed function and the input or output of the Single-Objective optimizer.<p>

We also intend to evaluate the performance of our full implementation on a highly dimensional mixed categorical-continuous space representative of the Rolfes Group use case.<p>
## Contents

1. preprocessor function
2. BayesianOptimization class
3. Cost aware acquisition functions as python classes
4. BODi
5. Experiment emulator class
6. Experiment setup and plots
7. Weighted VAEs and regular VAEs.

We recommend that the reader check out the example_use folder for examples on how to use the preprocessor, BayesianOptimization class and the experiment emulator. Our experiment setup and results are contained in the experiments folder.
