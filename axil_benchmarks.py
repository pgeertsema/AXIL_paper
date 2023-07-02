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


# import necessary packages
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from scipy.stats import pearsonr
from sklearn.datasets import make_regression
import openml
import axil
import gc
import os, psutil

# suppress k-means memory leak warning
import warnings
warnings.filterwarnings('ignore')

# data directory
DATA = r"D:\Data\AXIL"
RESULTS = r"C:\Paul\Dropbox\AXIL\results"


# constants
LR = 0.1 # default
LEAVES = 4 # to prevent overfitting in small datasets (default = 31)
TREES = 100 #default

# define the number of observations and features for synthetic data
N = 1000
M = 20

#------------------------------------------------------------------------------------------------------------------
#  Datasets
#------------------------------------------------------------------------------------------------------------------


openml_datasets = {
    "Abalone": 45033,        # (4177,7)
    "Airfoil": 44957,        # (1503,5)
    "Autos": 42372,          # (392,5)
    "Boston": 531,           # (506,22)
    "Concrete": 44959,       # (1030,8)
    "CPU": 227,              # (8192,12)
    "Diabetes": 41514,       # (442,10)
    "Forest fire": 43440,    # (517,10)
    "Grid stability": 44973, # (10000,12)
    "Red wine": 44972,       # (1599,11)
    "Titanic": 41265,        # (1307,7
    "Treasury": 42367,       # (1049,15)
}

# dictionary of datasets
datasets = {} 

openml_id = 204

for datasetname, openml_id in openml_datasets.items():
    print(datasetname, openml_id)

    # get data (will be cached automatically)
    dataset = openml.datasets.get_dataset(openml_id) 
    df_X, df_y, categorical, feature_names = dataset.get_data(dataset_format="dataframe")

    print(f"got data")

    # ensure there is a target
    if df_y is None:
        print("df_y is None, so get it from default_target_attribute")
        target = dataset.default_target_attribute
        df_y = df_X[target]
        df_X.drop([target],axis=1, inplace=True)
    else:
        print("got target from df_y")
        target = df_y.columns[0]

    print("target=",target)

    # convert categoricals to one-hot encoding
    for col, is_categorical in zip(df_X.columns, categorical):
        if is_categorical:
            dummies = pd.get_dummies(df_X[col], prefix=col).astype(int)
            df_X = pd.concat([df_X, dummies], axis=1)
            df_X.drop(col, axis=1, inplace=True)
    print("one hot encodings completed")

    # force y to be numeric (wine dataset)
    if df_y.dtype == 'object':
        df_y = pd.to_numeric(df_y, errors='coerce')
    print(f"y to numeric completed ({df_y.dtype})")
    
    # drop non-numeric cols from X
    df_X = df_X.select_dtypes(include=[np.number])
    print("drop non-numeric cols from X completed")            

    # get rid of missing values
    # -- combine df_X and df_y
    df_combined = pd.concat([df_X, df_y], axis=1)

    # -- drop rows with missing values
    df_combined = df_combined.dropna()

    # -- split df_combined back into df_X and df_y
    df_X = df_combined[df_X.columns]
    df_y = df_combined[df_y.name] 
    print("drop missing completed")         

    # convert to numpy arrays
    print(type(df_X), type(df_y))
    X = df_X.to_numpy()
    y = df_y.to_numpy()

    print("converted to ndarrays")

    # add to dictionary
    ds = {}
    ds["name"] = datasetname
    ds["X"] = X
    ds["y"] = y
    ds["description"] = dataset.description
    ds["id"] = openml_id
    ds["version"] = dataset.version
    num_instances, num_features = X.shape
    ds["instances"] = num_instances
    ds["features"] = num_features

    # add this dictionary to global dictionary of all datasets
    datasets[datasetname] = ds


#------------------------------------------------------------------------------------------------------------------
#  Algorithms
#------------------------------------------------------------------------------------------------------------------

# calculate the optimal in-sample number of clusters using silhouette score
def optimal_clusters_silhouette(X, max_clusters):
    scores = []  # silhouette scores for each k

    for k in range(2, max_clusters+1):  # silhouette score is not defined for a single cluster
        kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
        score = silhouette_score(X, kmeans.labels_)
        scores.append(score)

    optimal_k = scores.index(max(scores)) + 2  # Add 2 because k starts from 2
    return optimal_k


class BaseModel:
    """A base class for the algorithms."""
    def fit(self, X: np.array, y: np.array):
        raise NotImplementedError("Subclass must implement abstract method")

    def _calculate_K(self) -> np.array:
        raise NotImplementedError("Subclass must implement abstract method")

    def linear(self) -> np.array:
        K = self._calculate_K()  # Returns (N, N)
        
        # store original sum of elements
        orig_sum = K.sum()
        
        # set diagonal elements to 0, so implements leave-one-out prediction
        np.fill_diagonal(K, 0)
        
        # rescale so each row and column sum to same value as previously (since K is symetric)
        K = K * (orig_sum/K.sum())
        return K

class AXILClass(BaseModel):
    """
    Calculatex AXIL weights
    """
    def __init__(self, model, learning_rate) -> None:
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate

    def fit(self, X: np.array, y: np.array):  # X is (N, M), y is (N, 1)
        self.X = X
        self.explainer = axil.Explainer(self.model, learning_rate=self.learning_rate)
        self.explainer.fit(self.X)


    def _calculate_K(self) -> np.array:  # Returns (N, N)
        K = self.explainer.transform(self.X)        
        # free up memory
        self.explainer.reset()
        return K


class KAXILClass(BaseModel):
    """
    Calculate AXIL weights using only the k largest absolule instance weights 
    in each column of K generated by AXIL, rescaled to achieve the same weight in each column
    (For a more direct comparison with k-NN)
    """
    def __init__(self, model, learning_rate) -> None:
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate

    def fit(self, X: np.array, y: np.array):  # X is (N, M), y is (N, 1)
        self.X = X
        n, _ = X.shape
        # heuristic : set k = sqrt(n) where n is # of samples
        self.n_neighbors = int(np.sqrt(n))
        self.explainer = axil.Explainer(self.model, learning_rate=self.learning_rate)
        self.explainer.fit(self.X)

    def _calculate_K(self) -> np.array:  # Returns (N, N)
        K = self.explainer.transform(self.X)

        # get sum of each column, for rescaling later
        K_sum = np.sum(K, axis=0)
        
        # find the indices of the n_neighbors entries with largest absolute values for each column
        partitioned = np.argpartition(np.abs(K), -self.n_neighbors, axis=0)[-self.n_neighbors:]

        # create a mask with the same shape as K with False everywhere
        mask = np.zeros_like(K, dtype=bool)

        # set the positions of the largest absolute values (as per partitioned) to True
        for idx, col in enumerate(K.T):
            mask[partitioned[:, idx], idx] = True
        
        # apply the mask to set all non-largest values to zero
        K_masked = np.where(mask, K, 0)
        
        # rescale the columns
        K_sum_new = np.sum(K_masked, axis=0)
        K_prime = K_masked * (K_sum / K_sum_new)        

        # free up memory
        self.explainer.reset()
        # return adjusted K
        return K_prime


class KNNClass(BaseModel):
    '''
    Use k-NN to predict y from X. Then extract the nearest neighbors for each prediction i in [1,...,N] and use it to construct a vector k 
    such that each element of k is 1/#neigbours if the corresponding instance is in the cluster, and zero otherwise. Combine the column vectors k into a matrix K.
    '''
    def __init__(self):
        super().__init__()

    def fit(self, X, y): # X is (N, M), y is (N, 1)
        self.X = X
        n, _ = X.shape
        # heuristic : set k = sqrt(n) where n is # of samples
        self.n_neighbors = int(np.sqrt(n))
        self.knn = KNeighborsRegressor(n_neighbors=self.n_neighbors)
        self.knn.fit(X, y)

    def predict(self, X):
        return self.knn.predict(X)

    def _calculate_K(self):
        _, indices = self.knn.kneighbors(self.X)
        k_vectors = []
        for i in range(self.X.shape[0]):
            k = np.zeros(self.X.shape[0])
            neighbor_indices = indices[i]
            k[neighbor_indices] = 1/self.n_neighbors 
            k_vectors.append(k)
        K = np.column_stack(k_vectors)
        return K

class KMeansClass(BaseModel):
    """
    K-means clustering.. 
    k_{i} is calculated so that for instances in the same cluster as i the entry is (1/g) where g is the number of elements in that cluster, otherwise zero.
    """
    def __init__(self):
        super().__init__()

    def fit(self, X: np.array, y: np.array):  # X is (N, M), y is (N, 1)
        self.X = X
        # use silhoutte method to select optimal # of clusters
        self.clusters = optimal_clusters_silhouette(self.X, 10)
        self.kmeans = KMeans(n_clusters=self.clusters, init="k-means++", n_init="auto").fit(X)

    def _calculate_K(self) -> np.array:  # Returns (N, N)
        clusters = self.kmeans.predict(X)
        n, _ = self.X.shape
        K = np.zeros((n, n))
        for i in range(n):
            same_cluster = (clusters == clusters[i])
            K[i, same_cluster] = 1 / np.sum(same_cluster)
        return K


class HACClass(BaseModel):
    """
    Hierarchical agglomerative clustering with average linkage and Euclidean distance-based dissimilarity measure.
    k_{i} is calculated so that for instances in the same cluster as i the entry is (1/g) where g is the number of elements in that cluster, otherwise zero.
    """
    def __init__(self):
        super().__init__()

    def fit(self, X: np.array, y: np.array):  # X is (N, M), y is (N, 1)
        self.X = X        
        # use silhoutte method to select optimal # of clusters
        self.clusters = optimal_clusters_silhouette(self.X, 10)
        self.hac = AgglomerativeClustering(n_clusters=self.clusters).fit(X)

    def _calculate_K(self) -> np.array:  # Returns (N, N)
        clusters = self.hac.labels_
        n, _ = self.X.shape
        K = np.zeros((n, n))
        for i in range(n):
            same_cluster = (clusters == clusters[i])
            K[i, same_cluster] = 1 / np.sum(same_cluster)
        return K


class CSimilarityClass(BaseModel):
    """
    Cosine similarity is calculated over X.
    Each k_{i} is supposed to represent a vector of cosine similarities between instance i and every other instance.
    """

    def __init__(self):
        super().__init__()

    def fit(self, X: np.array, y: np.array):  # X is (N, M), y is (N, 1)
        self.X = X

    def _calculate_K(self) -> np.array:  # Returns (N, N)
        # scale cosine similarity to be strictly positive in range [0,1]
        # (if not done, performance is attrocious)
        K = (cosine_similarity(self.X) + 1)/2
        # rescale so rows and cols each add to 1 (convex combination)
        K = K*(K.shape[0]/K.sum())
        return K

class DistanceYClass(BaseModel):
    """
    The distance matrix D contain as each element the absolute Euclidean distance between between y_hat_{i} and y_hat_{j} for all i and j in [1,...,N].
    K=exp(-D) is calculated element by element.
    """

    def __init__(self, y_hat) -> None:
        super().__init__()
        self.y_hat = y_hat
        print("Shape", (self.y_hat.shape))

    def fit(self, X: np.array, y: np.array):  # X is (N, M), y is (N, 1)
        #self.y_hat = lgb.LGBMRegressor().fit(X, y.ravel()).predict(X)
        pass

    def _calculate_K(self) -> np.array:   # Returns (N, N)
        D = np.abs(self.y_hat[:, np.newaxis] - self.y_hat)
        # calculate kernal matrix K using gaussian radial basis function
        K = np.exp(-D)
        # rescale so rows and cols each add to 1 (convex combination)
        K = K*(K.shape[0]/K.sum())
        return K
    

class DistanceXClass(BaseModel):
    """
    The distance matrix D contain as each element the absolute Euclidean distance between X_{i} and X_{j} for all i and j in [1,...,N].
    K=exp(-D) is calculated element by element.
    """

    def __init__(self):
        super().__init__()

    def fit(self, X: np.array, y: np.array):  # X is (N, M), y is (N, 1)
        self.X = X

    def _calculate_K(self) -> np.array:  # Returns (N, N)        
        D = squareform(pdist(self.X))
        # calculate kernal matrix K using gaussian radial basis function
        K = np.exp(-D)
        # rescale so rows and cols each add to 1 (convex combination)
        K = K*(K.shape[0]/K.sum())
        return K


#------------------------------------------------------------------------------------------------------------------
#  Benchmarking
#------------------------------------------------------------------------------------------------------------------

def benchmark(X, y): 

    # train LightGBM model
    params = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": LEAVES,
        "verbose": 1,
        "min_data": 2,
        "learning_rate": LR,    
    }

    # reshape y to (N,1)
    y = y.reshape(-1, 1)

    # build GBM model
    lgb_data = lgb.Dataset(X, label=y)
    model = lgb.train(params, lgb_data, num_boost_round=TREES-1)

    # model predictions
    y_hat_GBM = model.predict(X)

    # Initialize the classes
    algorithms = {
        "AXIL": AXILClass(model, LR),
        "k-AXIL": KAXILClass(model, LR),        
        "k-NN" : KNNClass(),
        "k-Means": KMeansClass(),
        "HAC": HACClass(),
        "C-Similarity": CSimilarityClass(),
        "Distance-Y": DistanceYClass(y_hat_GBM),
        "Distance-X": DistanceXClass()
    }

    result = {}  # dictionary to store the results

    # fit and calculate K for each class
    for algo_name, algo in algorithms.items():
        algo.fit(X, y)
        K = algo.linear()

        # model the target variable y as a linear transformation of the training data
        y_hat = np.dot(K.T, y)

        # store the results in the dictionary
        #result[algo_name] = {"y_true": y, "y_hat": y_hat}
        result[algo_name] = {}

        # MSE
        rmse = np.sqrt( np.mean((y - y_hat)**2) )
        result[algo_name]["rmse"] = rmse
        
        # R^2
        corr, _ = pearsonr(y.ravel(), y_hat.ravel())
        # The R^2 measure in a univariate regression is the square of the pearson correlation between the variables        
        r2 = (corr) ** 2
        result[algo_name]["r2"] = r2

        print(f"{algo_name} rmse = {rmse} , r2 = {r2}")


    del y_hat_GBM, K, y_hat, X, y
    return result, algorithms

# set seed for reproduceability
np.random.seed(42)

# Generate synthetic dataset
X, y = make_regression(n_samples=N, n_features=M, noise=0.1)
rs, algs = benchmark(X,y)

#------------------------------------------------------------------------------------------------------------------
#  Run benchmarks
#------------------------------------------------------------------------------------------------------------------

# create matrices for storing results
rmse_results = pd.DataFrame(np.zeros((len(datasets), len(algs))), index=datasets.keys(), columns=algs.keys())
rmse_results.name = "rmse"
r2_results = pd.DataFrame(np.zeros((len(datasets), len(algs))), index=datasets.keys(), columns=algs.keys())
r2_results.name = "r2"

# iterate through datasets
for dataset_name, ds in datasets.items():
    print(dataset_name)
    # get data from dataset    
    X, y = ds["X"], ds["y"]
    # benchmark dataset
    rs, algs = benchmark(X, y)

    # free up memory
    gc.collect()

    # loop through algos
    algo_count = 0
    for algo_name, algo in algs.items():
        algo_count +=1
        # calculate metrics
        rmse = rs[algo_name]["rmse"]
        r2   = rs[algo_name]["r2"]
        print(f'=== dataset {dataset_name}, algo = {algo_name}, rmse = {rmse}, r2 = {r2}')

        # save metrics
        rmse_results.loc[dataset_name, algo_name] = rmse
        rmse_results.name = "rmse"
        r2_results.loc[dataset_name, algo_name] = r2
        r2_results.name = "r2"
    print("")

def df_to_latex(df, precision=2, cell_alignment="r", row_alignment="l", bold="none"):
    format_str = "{:." + str(precision) + "f}" # Formatting string to control precision
    num_columns = len(df.columns) + 1  # Plus one for index

    latex_str = "\\begin{tabular}{" + " ".join([row_alignment]+[cell_alignment] * (num_columns-1)) + "}\n"

    # Add empty space for index label in header
    latex_str += " & " + " & ".join(df.columns) + " \\\\\n"

    latex_str += "\\hline\n"

    # Add rows
    for index, row in df.iterrows():
        row_strs = [index]  # Start with index label
        for col in df.columns:
            val = row[col]
            if val == row.min() and bold == 'min':
                row_strs.append("\\textbf{" + format_str.format(val) + "}")  # Bold for min
            elif val == row.max() and bold == 'max':
                row_strs.append("\\textbf{" + format_str.format(val) + "}")  # Bold for max
            else:
                row_strs.append(format_str.format(val))  # Regular formatting
        latex_str += " & ".join(row_strs) + " \\\\\n"

    latex_str += "\\hline\n"
    latex_str += "\\end{tabular}"

    return latex_str


for df in [rmse_results, r2_results]:

    name = df.name
    # limit decimal places to 2
    df = df.round(3)

    latex = None
    # save results to latex    
    if name == 'rmse':
        latex = df_to_latex(df, precision=3, bold="min")
    if name == 'r2':
        latex = df_to_latex(df, precision=3, bold="max")

    # write the output to a .tex file
    with open(RESULTS+'\\benchmark_'+name+'.tex', 'w') as f:
        f.write(latex)

datasets.keys()