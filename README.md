# Bayesian Optimization For Robot Chemistry Lab
This repository contains implementations of bayesian optimization, data pre-processing, and dimension reduction as functions or classes in Python. The goals are to provide the following implementations such that they are suitable for both individual use and automation in a robot chemistry lab setting:
1. Function to preprocess data collected by the Rolfes Group.
2. Single-Objective and Multi-Objective optimizers using Bayesian Optimization.
3. Dimension reduction implementation which is compatible with the output of the pre-processed function and the input or output of the Single-Objective optimizer.<p>
We also intend to evaluate the performance of our full implementation on a highly dimensional mixed categorical-continuous space representative of the Rolfes Group use case.<p>
## Context
### Fontys Robotlab Research
The RobotLab program ‘Big Chemistry’ has received over 90 million euros from the National Growth Fund to position the Netherlands as a global leader in chemical robotics combined with artificial intelligence. By building an autonomous ‘RobotLab’, large numbers of experiments can be carried out, yielding large datasets on properties of molecular systems. The aim is to train new algorithms to predict the properties of molecular systems, e.g. solubility, phase separation, critical micelle concentration, smell, toxicity, and reaction rates. <p>
Fontys contributes to this program through practice oriented research. Central to the research plan is the construction of a robot lab, combining chemical research, high technology (robotics) and artificial intelligence (big data + self-learning systems). Fixed and mobile robots will be developed to automate the lab, if not completely, to a large degree. The data collected during chemical expert experiments will help to perform new expert experiments faster and better (recommender system).
### Experiment planning through Bayesian optimization
In a lot of applications and fundamental research, chemists aim to optimize one or more physical quantities that are a result of a chemical reaction/process. For example, the yield of a certain substance is to be maximised in the least amount of time, consuming the least amount of chemicals. The design of experiments in this context is to be constructed as the way to find the (unique) combination of parameters that may be varied in a chemical experiment (e.g. pH, concentration and nature of a buffer or co-colvent, temperature, etc.) that optimizes some quantity of interest.<p>
Mathematically, this problem may be thought of as a problem of optimizating an expensive to evaluate function (i.e. doing the experiment) in a (typically) high-dimensional space. Bayesian optimization (BO) is emerging as a standard approach to the design of experiments in this context. In BO the quantity of interest, which is regarded as a function of the (large number of) chemical parameters, is modeled as a Gaussian process, giving information about the expected values of, as well as a measure of the uncertainty about the quantity for which the experiments are to be optimized. The first step in BO is to perform a Gaussian process regression to all available  experimental data. Once this is complete, a so-called acquisition function, which can be constructed from the Gaussian process, is to be optimized as this function codifies for which combination of chemical parameters it is either possible or likely (exploration vs. exploitation) that the optimum will be reached.
## Our Approach
We define our dataset as an excel file consisting of two different sheets. The first sheet should show
|Time   |T° Read|A_1|...|A_n|
|-------|-------|---|---|---|
|t_1    |       |   |   |   |
|\vdots |       |   |   |   |
|t_m    |...    |...|...|...|
##