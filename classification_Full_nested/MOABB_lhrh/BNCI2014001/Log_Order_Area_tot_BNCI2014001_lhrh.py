import pandas as pd
import joblib
import os
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from pickle import load

warnings.filterwarnings("ignore")

order_MDM = []
tau_MDM = []

# BNCI2014001
path = "/home/icarrara/Documents/Project/reduced_dataset/classification_Full_nested/MOABB_lhrh/BNCI2014001"

new_data = np.zeros(shape=(10, 10))


for subject in np.arange(1, 10):
    for session in ["session_E", "session_T"]:
        name_grid = os.path.join(path, "FIX_STATE_ART_Subject_" + str(subject), "Models_WithinSession", "001-2014",
                                 str(subject),
                                 str(session), "ACM+MDM(Grid)")
        for i in np.arange(5):
            name = os.path.join(name_grid, "fitted_model_" + str(i) + ".pkl")
            with open(name, "rb") as pickle_file:
                search = load(pickle_file)
                tau_MDM.append(search.best_params_['augmenteddataset__lag'])
                order_MDM.append(search.best_params_['augmenteddataset__order'])
                best_score = search.best_score_


        pvt = pd.pivot_table(pd.DataFrame(search.cv_results_), values='mean_test_score', index='param_augmenteddataset__lag',
                             columns='param_augmenteddataset__order')

        mask = pvt < best_score*0.99

        mask_binary = np.logical_not(mask).to_numpy().astype(int)
        new_data = new_data + mask_binary

ax = sns.heatmap(data=new_data)
ax.set_xticklabels(np.arange(1, 11))
ax.set_yticklabels(np.arange(1, 11))
ax.invert_yaxis()
ax.set_xlabel("Order")
ax.set_ylabel("Lag")
plt.savefig("/home/icarrara/Documents/Project/reduced_dataset/classification_Full_nested/MOABB_lhrh/BNCI2014001/Figure/Area_Full.pdf")
plt.savefig("/home/icarrara/Documents/Project/reduced_dataset/classification_Full_nested/MOABB_lhrh/BNCI2014001/Figure/Area_Full.png")
ind = np.unravel_index(np.argmax(new_data, axis=None), new_data.shape)
print("Best Lag = ", ind[0]+1)
print("Best Order = ", ind[1]+1)
##

