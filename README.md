# AXIL_paper

This repository contains code for the paper "*Instance-based Explanations for Gradient Boosting Machine Predictions with AXIL Weights*" Geertsema & Lu (2022)

axil.py contains functionality for fitting to a LightGBM model and training data (Explainer.fit()), and constructing AXIL weights for a data set (Explainer.transform())

wdi.py illustrates the application of AXIL weights in the context of predicting smoking prevalence by country (see the paper for details)

NOTE: This is experimental code for research purposes. Functionality may change without warning.

Feel free to send any comments / suggestions to p.geertsema at auckland.ac.nz

AXIL weights for countries in relation to predicting smoking prevalence (see paper)

![image](https://user-images.githubusercontent.com/78324985/205521898-85c37c94-d3a8-4f1f-a101-f57f2e62c1e8.png)

