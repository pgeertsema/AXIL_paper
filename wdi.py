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

import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
import seaborn as sns
import networkx as nx
from sklearn.metrics import mean_squared_error, r2_score
import scipy
import axil
import os
import shap

SOURCE = r"D:\Data\AXIL"
RESULTS = r"C:\Paul\Dropbox\AXIL\results"
os.chdir(RESULTS)

#---------------------------------------------------------------
# Data
#---------------------------------------------------------------

TARGET = "smoking_rate"

data = pd.read_stata(SOURCE+"\\final_data.dta", index_col="Country")
data = data.sort_values([TARGET], ascending=False).reset_index()
data["gdp_per_capita"]

N = len(data)

# use full sample
data_train = data
data_test = data

X_train = data_train.drop(columns=[TARGET, "Country", "countrycode"])
y_train = data_train[TARGET]

X_test = data_test.drop(columns=[TARGET, "Country", "countrycode"])
y_test = data_test[TARGET]

plt.close()
plt.scatter(data_train.gdp_per_capita, data_train.smoking_rate)
plt.xlabel("gdp_per_capita")
plt.ylabel("smoking_rate") 
plt.savefig("scatter_"+TARGET+"_vs_GDP.pdf")
plt.close()


#---------------------------------------------------------------
# Model parameters
#---------------------------------------------------------------

TREES = 3
LEAVES = 6
LR = 0.1

params = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": LEAVES,
    "verbose": 1,
    "min_data": 4,
    "learning_rate": LR,    
}

#---------------------------------------------------------------
# Train model
#---------------------------------------------------------------

np.random.seed(42)

lgb_data = lgb.Dataset(X_train, label=y_train)
model = lgb.train(params, lgb_data, num_boost_round=TREES)

y_train_hat = model.predict(X_train)
y_test_hat = model.predict(X_test)

y_hat_pd = pd.DataFrame(y_train_hat, columns=["y_hat_pd"])
result = pd.concat([data["Country"], y_train, y_hat_pd], axis=1)
result

plt.scatter(y_train, y_train_hat)
plt.xlabel("actual "+TARGET)
plt.ylabel("model predicted "+TARGET)
plt.savefig("actual_vs_predicted_"+TARGET+".pdf")
plt.close()

print(f"Train RMSE = {mean_squared_error(y_train, y_train_hat)}")
print(f"Train R2 = {r2_score(y_train, y_train_hat)}")

print(f"Test RMSE = {mean_squared_error(y_test, y_test_hat)}")
print(f"Test R2 = {r2_score(y_test, y_test_hat)}")

#---------------------------------------------------------------
# SHAP explanations
#---------------------------------------------------------------

SHAP_explainer = shap.Explainer(model)
shap_values = SHAP_explainer(X_test)
fig=plt.gcf()
shap.plots.bar(shap_values, show=False)
fig.savefig("SHAP_magnitudes_"+TARGET+".pdf", bbox_inches="tight")

#---------------------------------------------------------------
# Chosen example with SHAP
#---------------------------------------------------------------

chosen_index = data_test.Country.to_list().index("South Africa")

chosen_shaps = pd.concat([ X_train.columns.to_series().reset_index(),  pd.Series(shap_values[chosen_index].values).reset_index()], axis=1)
chosen_shaps = chosen_shaps.iloc[:,[1,3]]
chosen_shaps.columns =["Country","SHAP"]
chosen_shaps.sort_values("SHAP", inplace=True, ascending=False)

# check
diff = y_train_hat[chosen_index] - (y_train_hat.mean() + chosen_shaps["SHAP"].sum())
print("Chosen_diff = ", diff)

num = 2 
mean_list = [("mean_of_predictions", y_train_hat.mean())]
start_list = [(a,b) for (a,b) in zip(chosen_shaps.head(num).Country, chosen_shaps.head(2).SHAP)] 
end_list = [(a,b) for (a,b) in zip(chosen_shaps.tail(num).Country, chosen_shaps.tail(2).SHAP)]

omitted = y_train_hat[chosen_index] - (y_train_hat.mean() + chosen_shaps.head(2).SHAP.sum() + chosen_shaps.tail(2).SHAP.sum())
omitted_list = [("<omitted>", omitted)]
pred_list = [("model_prediction", y_train_hat[chosen_index])]

combined_list = mean_list + start_list + omitted_list + end_list + pred_list

# print abbreviated SHAP values
for a, b in combined_list:
    print(a,b)

# -------------------------------------------------------------------
# AXIL explanations
# -------------------------------------------------------------------

AXIL_explainer = axil.Explainer(model)
AXIL_explainer.fit(X_train)

# --- in-sample explanations ---

loadings_train = AXIL_explainer.transform(X_train)

# v is the training data targets with shape (N x 1)
v = y_train.to_numpy().reshape(len(y_train),1)
# (N x 1)

axil_y_train_hat = loadings_train.T @ v
axil_y_train_hat

print(axil_y_train_hat - y_train_hat.reshape(len(y_train_hat),1))

print(np.min(loadings_train))
print(np.max(loadings_train))

# --- out-of-sample explanations ---

loadings_test = AXIL_explainer.transform(X_test)
axil_y_test_hat = loadings_test.T @ v
diff = axil_y_test_hat - y_test_hat.reshape(len(y_test_hat),1)
print(diff)
S = len(X_test)
assert(np.allclose(diff, np.zeros((1,S)), atol=1e-5))

print(np.min(loadings_test))
print(np.max(loadings_test))

# AXIL weights 

AXIL_test_weights = pd.concat([data_train["Country"].reset_index(drop=True),  pd.DataFrame(loadings_test).reset_index(drop=True)], axis=1, ignore_index=True)
AXIL_test_weights.columns = ["Country"] + data_test["Country"].to_list()
AXIL_test_weights.set_index(AXIL_test_weights.columns[0], inplace=True)

# export to csv
AXIL_test_weights.to_csv("AXIL_weights.csv")

# export targets
y_test.to_csv("Target_actual.csv")
np.savetxt("Target_predicted.csv", y_test_hat)

#---------------------------------------------------------------
# Chosen example with AXIL
#---------------------------------------------------------------

# Chosen
chosen_index = data_test.Country.to_list().index("South Africa")
print(y_train @ loadings_train[:,chosen_index])

# print Chosen weights
chosen_AXIL_weights = pd.concat([data["Country"], pd.DataFrame(loadings_train[:,chosen_index]), pd.DataFrame(y_train)], axis=1)
chosen_AXIL_weights.columns = ["Country", "AXIL","y_train"]
chosen_AXIL_weights["product"] = chosen_AXIL_weights["AXIL"] * chosen_AXIL_weights["y_train"]

chosen_AXIL_weights.sort_values("AXIL", inplace=True, ascending=False)

AXIL_test_weights["South Africa"].sort_values(ascending=False).head(6)
AXIL_test_weights["South Africa"].sort_values(ascending=False).tail(6)

y_train_hat.mean()/len(y_train_hat)

# -------------------------------------------------------------------
# Graphs
# -------------------------------------------------------------------

# AXIL weights heatmap
# -------------------------------------------------------------------

plt.close()
ax = sns.heatmap(AXIL_test_weights, xticklabels=1, yticklabels=1, cmap="Blues", cbar=False)
ax.set_box_aspect(1)
ax.yaxis.tick_right()
ax.set(ylabel=None)
ax.tick_params(axis='both', which='major', labelsize=8)
plt.yticks(rotation=0)
ax.figure.tight_layout()
plt.savefig("heatmap_all_"+TARGET+".pdf")
plt.close()

# AXIL weights clustered heatmap
# -------------------------------------------------------------------

plt.close()
import scipy.spatial as sp, scipy.cluster.hierarchy as hc
linkage = hc.linkage(AXIL_test_weights, method='average')
ax = sns.clustermap(AXIL_test_weights, row_linkage=linkage, col_linkage=linkage, xticklabels=1, yticklabels=1, cmap="Blues", cbar_pos=None)
ax.figure.tight_layout()
ax.ax_heatmap.set_ylabel("")
plt.savefig("clustermap_all_"+TARGET+".pdf")
plt.close()


# AXIL weights network
# -------------------------------------------------------------------

labels = data_test["Country"].to_dict()
labels

graph = loadings_test.copy()

tr = np.percentile(graph, 85)
tr
graph[graph <= tr] = 0
graph[graph > tr] = 1
graph.min()
graph.max()

np.fill_diagonal(graph, 0)
G = nx.from_numpy_array(graph)
nx.draw_circular(G, with_labels=True, labels=labels, font_weight='light', node_color = data_test[TARGET].to_list(), cmap="Reds", edge_color="gray")
plt.savefig("network_"+TARGET+".pdf")
plt.close()