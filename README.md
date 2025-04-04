# Bayesian Optimization For Robot Chemistry Lab
This repository contains implementations of bayesian optimization, data pre-processing, and dimension reduction in Python. The goals are to provide the following implementations such that they are suitable for both individual use and automation in a robot chemistry lab setting:
1. Function to preprocess data collected by the Rolfes Group.
2. Single-Objective and Multi-Objective optimizers using Bayesian Optimization.
3. Dimension reduction implementation which is compatible with the output of the pre-processed function and the input or output of the Single-Objective optimizer.
We also intend to test the performance of our full implementation on a highly dimensional mixed categorical-continuous space representative of the Rolfes Group use case.
##