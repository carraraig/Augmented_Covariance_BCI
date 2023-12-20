import pandas as pd
import joblib
import os
import numpy as np
import warnings
from pickle import load

warnings.filterwarnings("ignore")

order_MDM = []
tau_MDM = []
order_MDOP_MDM = []
tau_MDOP_MDM = []




# BNCI2014001
path = "/home/icarrara/Documents/Project/reduced_dataset/classification_Full_nested/MOABB_lhrh/BNCI2014001"
for subject in np.arange(1, 10):
    for session in ["session_E", "session_T"]:
        name_grid = os.path.join(path, "FIX_STATE_ART_Subject_" + str(subject), "Models_WithinSession", "001-2014",
                                 str(subject),
                                 str(session), "ACM+MDM(Grid)")
        # if os.path.isdir(name_grid):
        for i in np.arange(5):
            name = os.path.join(name_grid, "fitted_model_" + str(i) + ".pkl")
            with open(name, "rb") as pickle_file:
                search = load(pickle_file)
                tau_MDM.append(search.best_params_['augmenteddataset__lag'])
                order_MDM.append(search.best_params_['augmenteddataset__order'])

for subject in np.arange(1, 10):
    for session in ["session_E", "session_T"]:
        name_grid = os.path.join(path, "FIX_MDOP_Subject_" + str(subject), "Models_WithinSession", "001-2014",
                                 str(subject),
                                 str(session), "ACM+MDM(MDOP)")
        # if os.path.isdir(name_grid):
        for i in np.arange(5):
            name = os.path.join(name_grid, "fitted_model_" + str(i) + ".pkl")
            with open(name, "rb") as pickle_file:
                search = load(pickle_file)
                tau_MDOP_MDM.append(search.steps[0][1].lag)
                order_MDOP_MDM.append(search.steps[0][1].order)


## Graph
import matplotlib.pyplot as plt

df = pd.DataFrame(list(zip(order_MDM, order_MDOP_MDM, tau_MDM, tau_MDOP_MDM)),
               columns =["Order_MDM", "Order_MDOP_MDM", "Tau_MDM", "Tau_MDOP_MDM"])

df.hist(bins=10, figsize=(15, 10))
plt.savefig("/home/icarrara/Documents/Project/reduced_dataset/classification_Full_nested/MOABB_lhrh/BNCI2014001/Figure/Grid_Search.pdf")
##
import matplotlib.pyplot as plt
fig= plt.figure(figsize=(6, 6))
plt.scatter(order_MDM, tau_MDM, c="red", label="Grid Search")
plt.scatter(order_MDOP_MDM, tau_MDOP_MDM, c="blue", label="Unified (MDOP)")
plt.xlabel("Order")
plt.ylabel("Lag")
plt.grid(False)
plt.legend()
plt.savefig("/home/icarrara/Documents/Project/reduced_dataset/classification_Full_nested/MOABB_lhrh/BNCI2014001/Figure/order_lag.pdf")
plt.savefig("/home/icarrara/Documents/Project/reduced_dataset/classification_Full_nested/MOABB_lhrh/BNCI2014001/Figure/order_lag.png")


##

