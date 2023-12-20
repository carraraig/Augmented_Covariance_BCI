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


for subject in np.arange(1, 2):
    for session in ["session_E"]:
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


        pvt = pd.pivot_table(pd.DataFrame(search.cv_results_),
                             values='mean_test_score', index='param_augmenteddataset__lag',
                             columns='param_augmenteddataset__order')

        mask = pvt < best_score*0.99

        mask_binary = np.logical_not(mask).to_numpy().astype(int)
        new_df = pvt[np.logical_not(mask)]

        ax = sns.heatmap(data=pvt, mask=mask, cbar=True)
        ax.invert_yaxis()

plt.xlabel("Order")
plt.ylabel("Lag")
plt.savefig("/home/icarrara/Documents/Project/reduced_dataset/classification_Full_nested/MOABB_lhrh/BNCI2014001/Figure/Area.pdf")
plt.savefig("/home/icarrara/Documents/Project/reduced_dataset/classification_Full_nested/MOABB_lhrh/BNCI2014001/Figure/Area.png")



##

