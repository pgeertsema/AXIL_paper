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

# module docstring 
"""AXIL - Additive eXplanations using Instance Loadings"""

#---------------------------------------------------------------
# imports
#---------------------------------------------------------------

import numpy as np
import lightgbm as lgb
import os
import psutil


#---------------------------------------------------------------
# Leaf Coincidence Matrix (LCM) utility function
#---------------------------------------------------------------


def LCM(vector1, vector2):
    '''
    utility function to create leaf coincidence matrix L from leaf membership vectors vector1 (train) and vector2 (test)
    element l_s,f in L takes value of 1 IFF observations v1 and v2 are allocated to the same leaf (for v1 in vector1 and v2 in vector2)
    that is, vector1[v1] == vector2[v2], otherwise 0

    Input arguments vector1 and vector2 are list-like, that is, support indexing and len()
    Output L is a python matrix of dimension (len(vector1), len(vector2))
    '''
    # relies on numpy broadcasting to generate self equality matrix
    vector1 = np.array(vector1)[:, None]
    vector2 = np.array(vector2)
    return vector1 == vector2

# LCM([201,893,492,131,478,653,152],[58,131,307,653,492])*1



#---------------------------------------------------------------
# functionality starts here
#---------------------------------------------------------------

class Explainer:

    # internal variables
    model = None
    learning_rate = None
    train_data = None
    P_list = None
    lm_train = []
    trained = False

    # constructor
    def __init__(self, model, learning_rate=None):

        if not isinstance(model, lgb.basic.Booster):
            print("Sorry, the model needs to be a LightGBM regression model (type 'lightgbm.basic.Booster').")
            return None
       
        self.model = model
        if learning_rate == None:

            # get from text file of model (a hack, but currently the only way)
            self.model.save_model("temp_booster.txt")
            found = False
            with open("temp_booster.txt", "r") as f:
                for line in f:
                    if "[learning_rate: " in line:
                        lr = line.replace("[learning_rate: ", "").replace("]", "")
                        try:
                            self.learning_rate = float(lr)
                        except:
                            # catch error if unable to convert lr to float
                            print("Sorry, unable to figure out the learning rate from the model provided. Please supply the 'learning_rate' parameter.")
                            return None
                        else:
                            found = True
                # delete temporary model file
                f.close()
                os.remove("temp_booster.txt")

            if not found:
                print("Sorry, unable to figure out the learning rate from the model provided. Please supply the 'learning_rate' parameter.")
                return None

        else:

            # or, if supplied by the user
            self.learning_rate = learning_rate           

        return


    # fit the model to training data
    def fit(self, X):

        if isinstance(X, lgb.basic.Dataset):
            print("Sorry, you need to provide the raw data (type lightgbm.basic.Dataset), just like for LightGBM predict()")
            return None

        # number of observations in data
        N = len(X)

        self.train_data = X

        # obtain instance leaf membership information from trained LightGBM model (argument pred_leaf=True)
        instance_leaf_membership = self.model.predict(data=X, pred_leaf=True)

        # the first "tree" mimics a single leaf, so that it effectively calculates the training data sample average
        lm = np.concatenate((np.ones((1, N)), instance_leaf_membership.T), axis = 0) + 1

        # number of trees in model
        num_trees = self.model.num_trees()

        # useful matrices
        ones = np.ones((N,N))
        I = np.identity(N)

        # Clear list of P matrices (to be used for calculating AXIL weights)
        P_list = []

        # iterations 0 model predictions (simply average of training data)
        # corresponds to "tree 0"
        P_0 = (1/N) * ones
        P_list.append(P_0)
        G_prev = P_0

        # do iterations for trees 1 to num_trees (inclusive)
        # note, LGB trees ingnores the first (training data mean) predictor, so offset by 1
        for i in range(1, num_trees+1):

            D = LCM(lm[i], lm[i])
            P = self.learning_rate * ( (D / (ones @ D)) @ (I-G_prev) )
            P_list.append(P)
            G_prev +=  P
            #process = psutil.Process()
            #print(f"Fitting tree {i}, memory used (GB):", process.memory_info().rss/1000000000)  # in mb


        self.trained = True
        self.P_list = P_list
        self.lm_train = lm
        return

    def transform(self, X):

        if not self.trained:
            print("Sorry, you first need to fit to training data. Use the fit() method.")
            return None

        # list of P matices
        P = self.P_list

        # number of instances in training data used to estimate P
        N, _ = P[0].shape

        # number of instances in this data
        S = len(X)

        # model instance membership of tree leaves 
        instance_leaf_membership = self.model.predict(data=X, pred_leaf=True)

        lm_test = np.concatenate((np.ones((1, S)), instance_leaf_membership.T), axis = 0) + 1

        # number of trees in model
        num_trees = self.model.num_trees()

        # ones matrix with same dimensions as P
        ones_P = np.ones((N, N))
        
        # ones matrix with same dimensions as L
        ones_L = np.ones((N, S))

        # first tree is just sample average
        L = ones_L
        K = (P[0].T @ (L / (ones_P @ L)))

        # execute for 1 to num_trees (inclusive)
        for i in range(1, num_trees+1):
            L = LCM(self.lm_train[i], lm_test[i])
            K += (P[i].T @ (L / (ones_P @ L)))
            #process = psutil.Process()
            #print(f"Transforming tree {i}, memory used (GB):", process.memory_info().rss/1000000000)  # in mb


        return K
    
    def reset(self):
        # reset to initialised state
        self.train_data = None
        self.P_list = None
        self.lm_train = []
        self.trained = False
