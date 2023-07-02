# AXIL - Additive eXplanations using Instance Loadings
# Copyright (C) Paul Geertsema 2022, 2023
# Python code to represent LightGBM regression predictions as a linear combination of training data target values
# See "Instance-based Explanations for Gradient Boosting Machine Predictions" by Geertsema & Lu

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.


#---------------------------------------------------------------
# imports
#---------------------------------------------------------------
from sklearn.datasets import make_friedman1
import numpy as np
import lightgbm as lgb
import axil

#---------------------------------------------------------------
# model settings
#---------------------------------------------------------------

OBSERVATIONS = 100
FEATURES = 16
LEARNING_RATE = 0.01
TREES = 52
LEAVES = 14
TOLERANCE = 1e-6

#---------------------------------------------------------------
# data
#---------------------------------------------------------------

# non-linear synthetic data
X, y = make_friedman1(n_samples = OBSERVATIONS, n_features = FEATURES, noise = 0, random_state= 42)
print(X, y)

#---------------------------------------------------------------
# model parameters
#---------------------------------------------------------------

params = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": LEAVES,
    "verbose": 1,
    "min_data": 2,
    "learning_rate": LEARNING_RATE,    
}

#---------------------------------------------------------------
# train model
#---------------------------------------------------------------

lgb_data = lgb.Dataset(X, label=y)

# the sample average mean is treated in our analysis as a tree with one leaf
model = lgb.train(params, lgb_data, num_boost_round=TREES-1)

# model predictions

y_hat = model.predict(X)
v = y.reshape(OBSERVATIONS,1).copy()

#---------------------------------------------------------------
# In-sample test
#---------------------------------------------------------------

explainer = axil.Explainer(model, learning_rate=LEARNING_RATE)
explainer.fit(X)

K = explainer.transform(X)
K.shape

k = v.T @ K
k.shape

diff_pred = k - y_hat
diff_pred

print("===CHECK TRAIN===: difference in GBM predicted ", np.round(diff_pred,6))
assert(np.allclose(diff_pred, np.zeros((1,OBSERVATIONS)), atol=TOLERANCE))

#---------------------------------------------------------------
# Out-of-sample test
#---------------------------------------------------------------

TEST_OBSERVATIONS = 200

X_test, y_test = make_friedman1(n_samples = TEST_OBSERVATIONS, n_features = FEATURES, noise = 0, random_state= 123)

print(X_test, y_test)

S = len(X_test)

K_test = explainer.transform(X_test)
K_test.shape

y_test_hat = model.predict(X_test)
y_test_hat.shape

k_test = v.T @ K_test
k_test.shape
k_test

diff_pred_test = k_test - y_test_hat
diff_pred_test

print("===CHECK TEST===: difference in GBM predicted ", np.round(diff_pred_test,6))
assert(np.allclose(diff_pred_test, np.zeros((1,S)), atol=TOLERANCE))