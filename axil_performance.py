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
import timeit
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

#---------------------------------------------------------------
# model settings
#---------------------------------------------------------------

OBSERVATIONS = 100
FEATURES = 16
LEARNING_RATE = 0.01
TREES = 52
LEAVES = 14
TOLERANCE = 1e-6

# set seed for reproduceability
np.random.seed(42)

#---------------------------------------------------------------
# model parameters
#---------------------------------------------------------------

params = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": LEAVES,
    "verbose": -1,
    "min_data": 2,
    "learning_rate": LEARNING_RATE,    
}

#---------------------------------------------------------------
# Performance evaluation
#---------------------------------------------------------------

# We consider the time complexity when fitting n instances with a GBM model with m
# trees. Fitting fixed m trees with fixed q separate n × n matrix multiplications (each
# O(nw) with w ≈ 2.8 (Strassen's algorithm)
# O(m × q × O(n))

# so linear in # trees and n^2.8-ish in n observations

results = pd.DataFrame(columns=['Tree Count', 'Instance Count', 'Execution Time'])

#warmup
tree_count = 16
instance_count = 100
X, y = make_friedman1(n_samples = instance_count, n_features = FEATURES, noise = 0, random_state= 42)
lgb_data = lgb.Dataset(X, label=y)
model = lgb.train(params, lgb_data, num_boost_round=tree_count-1)
explainer = axil.Explainer(model, learning_rate=LEARNING_RATE)

explainer = axil.Explainer(model, learning_rate=LEARNING_RATE)
start_time = timeit.default_timer()
#--- timed
explainer.fit(X)
#--- timed
end_time = timeit.default_timer()
execution_time = end_time - start_time
print(f"Performance evaluation: {tree_count} trees and {instance_count} instances took {execution_time} seconds.")

# record performance

for tree_count in [8, 16, 32, 64]:
    for instance_count in [250, 500, 750, 1000, 1500, 2000, 3000, 5000, 10000]:

        # non-linear synthetic data
        X, y = make_friedman1(n_samples = instance_count, n_features = FEATURES, noise = 0, random_state= 42)
        lgb_data = lgb.Dataset(X, label=y)
        model = lgb.train(params, lgb_data, num_boost_round=tree_count-1)
        explainer = axil.Explainer(model, learning_rate=LEARNING_RATE)

        explainer = axil.Explainer(model, learning_rate=LEARNING_RATE)
        start_time = timeit.default_timer()
        #--- timed
        explainer.fit(X)
        #--- timed
        end_time = timeit.default_timer()
        execution_time = end_time - start_time
        print(f"Performance evaluation: {tree_count} trees and {instance_count} instances took {execution_time} seconds.")

        results = pd.concat([results, pd.DataFrame([{'Tree Count': tree_count, 'Instance Count': instance_count, 'Execution Time': execution_time}])], ignore_index=True)


# functional form for fitted curves
def func(x, a, b):
    return a * x ** b

# define colors for the lines (make sure to have as many colors as you have unique tree counts)
colors = ['b', 'g', 'r', 'c']

# Now we plot the results
for i, tree_count in enumerate(results['Tree Count'].unique()):
    temp_df = results[results['Tree Count'] == tree_count]

    # fit the function to the data
    params, params_covariance = curve_fit(func, temp_df['Instance Count'], temp_df['Execution Time'])

    print(f"Tree count {tree_count} fitted parameters: {params}")

    plt.semilogy(temp_df['Instance Count'], temp_df['Execution Time'], 'o', color=colors[i], label=f'Tree Count {tree_count} data')
    plt.semilogy(temp_df['Instance Count'], func(temp_df['Instance Count'], params[0], params[1]), color=colors[i], label=f'Tree Count {tree_count} fitted')

    # annotate the parameters on the graph
    plt.annotate(f'a={params[0]:.2f}, b={params[1]:.2f}', (temp_df['Instance Count'].max(), func(temp_df['Instance Count'].max(), params[0], params[1])), textcoords="offset points", xytext=(-10,-10), ha='center')

plt.xlabel('Instance Count')
plt.ylabel('Execution Time (seconds, log scale)')
plt.legend()
#plt.show()
plt.savefig(r'C:\Paul\Dropbox\AXIL\results\performance_evaluation.pdf', format='pdf')

# pivot the DataFrame
pivot_df = results.pivot(index='Instance Count', columns='Tree Count', values='Execution Time')
pivot_df

# output results in latex
with open(r'C:\Paul\Dropbox\AXIL\results\performance_evaluation.tex', 'w') as tf:
    tf.write(pivot_df.to_latex(index=True, header=True))


