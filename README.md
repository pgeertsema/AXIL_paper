# AXIL_paper

This repository contains code for the paper "*Instance-based Explanations for Gradient Boosting Machine Predictions with AXIL Weights*" Geertsema & Lu (2023)

In a nutshell, AXIL weights allows any LightGBM regression prediction to be expressed as a sum of the products of the AXIL weights and the training data target instances. So $y_{j}^{pred}= k_{j} \cdot y_{train}$ where $k_{j}$ is the vector of AXIL weights for instance j and $y_{train}$ is the vector of training data targets.

[axil.py](axil.py) contains functionality for fitting to a LightGBM model and training data (Explainer.fit()), and constructing AXIL weights for a data set (Explainer.transform())

[axil_benchmarks.py](axil_benchmarks.py) contains code that benchmarks AXIL against k-NN and five other algorithms on 12 datasets.

[axil_performance.py](axil_performance.py) contains code that measures the execution time of AXIL on synthetics datasets with a vayring number of instances and GBM trees

[axil_test.py](axil_test.py) contains code that validates the AXIL predicted value against the GBM predicted value

[wdi.py](wdi.py) illustrates the application of AXIL weights in the context of predicting smoking prevalence by country (see the paper for details)

NOTE: This is experimental code for research purposes. Functionality may change without warning.

Feel free to send any comments / suggestions to p.geertsema at auckland.ac.nz

*Image below: AXIL weights for countries in relation to predicting smoking prevalence (see paper)*

![image](https://user-images.githubusercontent.com/78324985/205521898-85c37c94-d3a8-4f1f-a101-f57f2e62c1e8.png)

