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
We reiterate the task of maximizing the yield of some substance while minimizing the amount of ingredients and the time it would take to synthethize that much of that given substance.<p>
In the upcoming sections we will cover the following topics:
- what we consider as features and targets.
- How we pre-process the Rolfes data.
- How we implement Single and Multiple output Gaussian processes in Python.
- Which acquisition functions we use and the optimizer for it in Python.
- The manner in which we implement the minimization of the number of ingredients.
- How we implement Single and Multiple Objective Bayesian Optimization in Python.

### Pre-processing
We mentioned that Bayesian optimization would be used to optimize these values, but we must first properly define our samples, features and targets. In this section we will cover this while explaining how we pre-process data for the Rolfes Group.<p>We define our dataset as an excel file consisting of two different sheets. The first sheet should contain the measurement of some substance for $n$ wells in a given well-plate at $m$ different times, as well as the recorded temperature at those times. An example is shown below.
|Time|T° Read|$A_1$|...|$A_n$|
|-------|-------|---|---|---|
|$t_1$|$T_1$|$A_{1,1}^{\text{time}}$|$\cdots$|$A_{1,n}^{\text{time}}$|
|$\vdots$|$\vdots$|$\vdots$|$\ddots$|$\vdots$|
|$t_m$|$T_m$|$A_{m,1}^{\text{time}}$|$\cdots$|$A_{m,n}^{\text{time}}$|

We note that although the temperature is recorded, we do not use it.<p>The second sheet should be a table of $p$ parameters and their recorded value for each well as depicted in the table below. 
|well|$A_1$|$\cdots$|$A_n$|
|-------|---|---|---|
|$x_1$|$A_{1,1}^{x}$|$\cdots$|$A_{m,1}^{x}$|
|$\vdots$|$\vdots$|$\ddots$|$\vdots$|
|$x_p$|$A_{p,1}^{x}$|$\cdots$|$A_{p,n}^{x}$|

The function `preprocessor` (Not implemented yet) finds the interval of the onset of the reaction rate for all wells. It then calculates $v$ values which capture the gradient information of that interval. A table like the one above can be constructed analogously.<p>
`preprocessor` then returns our preprocessed data as a `DataFrame` with $v+p+1$ columns and $n$ rows. An example can be seen below. 
|well|$\tau_1$|$\cdots$|$\tau_v$|$x_1$|$\cdots$|$x_p$|yield|
|-|-|-|-|-|-|-|-|
|$A_1$|$A_{1,1}^{\tau}$|$\cdots$|$A_{1,n}^{\tau}$|$A_{1,1}^{x}$|$\cdots$|$A_{1,n}^{x}$|$A_{m,1}^{\text{time}}$|
|$\vdots$|$\vdots$|$\ddots$|$\vdots$|$\vdots$|$\ddots$|$\vdots$|$\vdots$|
|$A_n$|$A_{p,1}^{\tau}$|$\cdots$|$A_{p,n}^{\tau}$|$A_{p,1}^{x}$|$\cdots$|$A_{p,n}^{x}$|$A_{m,n}^{\text{time}}$|

To conclude on what we set out to define, Our samples are given by $A_1,...,A_n$.We have $x_1,...,x_p$ as features and our yield as a target.<p>
We can choose to interpret $\tau_1,...,\tau_v$ as targets replacing the yield or as additional targets. It is also possible to exclude them entirely if one does not wish to optimize for time.<p>We chose to allow for $v$ time components to be able to research which methods are most effective in using the yield gradient for single or multi objective optimization and prediction purposes.

### Gaussian Process Regression
After pre-processing, we must differentiate between single-objective and multi-objective optimization. In this section we will explain how we implement single and multiple output gaussian processes.<p>We will change the notation from the previous section to increase clarity.<p>Assume we have $d_\text{in}$ features to predict with and $d_\text{out}$ targets. We will restrict ourselves to some interval for both our features and targets. We denote $I\subseteq\R^{d_\text{in}}$ as our feature-space and $J\subseteq\R^{d_\text{out}}$ as our target-space.<p>We would like to approximate some black box function $f:I\to J$. We do this so that we can optimize this surrogate of $f$. In the context of the Rolfes group, $f$ takes the values of our chemicals as input and outputs either the final yield or information about the yield vs. time graph.<p>
A valid input to $f$ would be $x:=\begin{bmatrix}x_{1} \\\vdots \\x_{d_{\text{in}}}\end{bmatrix}\in I$, where $f(x):=\begin{bmatrix}f_1(x) \\\vdots \\f_{d_{\text{out}}}(x)\end{bmatrix}$