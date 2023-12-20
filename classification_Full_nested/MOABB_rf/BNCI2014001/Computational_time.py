from datetime import datetime as dt

import numpy as np

start = dt.now()

# Set up the Directory for made it run on a server.
import sys
import os
sys.path.append(r'/home/icarrara/Documents/Project/reduced_dataset')  # Local
# sys.path.append(r'/home/icarrara/Documents/Project/autoregressive_bci')  # Server

import moabb
import mne
new_path = '/home/icarrara/Documents/Project/autoregressive_bci/Data'  # Local
# new_path = '/home/icarrara/Documents/Project/autoregressive_bci/Data'  # Server
moabb.utils.set_download_dir(new_path)

import resource
from moabb.paradigms import MotorImagery
from moabb.datasets import BNCI2014001
from pyriemann.estimation import Covariances
from sklearn.pipeline import Pipeline
from moabb.evaluations import WithinSessionEvaluation
from ACM.Augmented_Dataset import AugmentedDataset
from pyriemann.classification import MDM

sub_numb = 1

# Initialize parameter for the Band Pass filter
fmin = 8
fmax = 35
tmin = 0
tmax = None

# Select the Subject
subjects = [sub_numb]
# Load the dataset, right now you have added Nothing events to DATA using new stim channel STI
dataset = BNCI2014001()

events = ["right_hand", "feet"]
# events = ["right_hand", "left_hand"]

paradigm = MotorImagery(events=events, n_classes=len(events), fmin=fmin, fmax=fmax, tmax=tmax)

# Create a path and folder for every subject
path = os.path.join(str("STATE_ART_Subject_" + str(sub_numb)))
os.makedirs(path, exist_ok=True)
##
Time = []

for i in np.arange(1, 11):
    # Pipelines
    pipelines = {}
    # Define the different algorithm to test and assign a name in the dictionary
    pipelines["ACM+MDM(Grid)"] = Pipeline(steps=[
        ("augmenteddataset", AugmentedDataset(order=i, lag=1)),
        ("Covariances", Covariances("oas")),
        ("MDM", MDM(metric=dict(mean='riemann', distance='riemann')))
    ])

    # Evaluation For MOABB
    # ========================================================================================================
    dataset.subject_list = dataset.subject_list[int(sub_numb) - 1:int(sub_numb)]
    # Select an evaluation Within Session
    evaluation = WithinSessionEvaluation(paradigm=paradigm,
                                         datasets=dataset,
                                         overwrite=True,
                                         random_state=42,
                                         hdf5_path=path,
                                         n_jobs=-1)

    # Print the results
    result = evaluation.process(pipelines)
    Time.append(result["time"])

##
import matplotlib.pyplot as plt
order = np.arange(1, 11)
time = np.array(Time)

# np.save("classification_Full_nested/MOABB_rf/BNCI2014001/Data_Time/BNCI2014001_rf_sub1", time)
# time = np.load("classification_Full_nested/MOABB_rf/BNCI2014001/Data_Time/BNCI2014001_rf_sub1.npy")

# plt.plot(order, time[:, 0], label="Session_E")
# plt.plot(order, time[:, 1], label="Session_T")
# plt.plot(order, time[:, 0], label="Session_0")
# plt.plot(order, time[:, 1], label="Session_1")
# plt.plot(order, time[:, 2], label="Session_2")
# plt.plot(order, time[:, 3], label="Session_3")
# plt.plot(order, time[:, 4], label="Session_4")
plt.plot(order, time[:, 0], label="Session_A")
plt.plot(order, time[:, 1], label="Session_B")
plt.legend()
plt.xlabel("Order")
plt.ylabel("Time (s)")
plt.show()

