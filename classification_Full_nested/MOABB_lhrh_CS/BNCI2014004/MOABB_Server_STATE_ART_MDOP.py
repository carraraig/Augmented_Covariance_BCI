from datetime import datetime as dt
start = dt.now()

# Set up the Directory for made it run on a server.
import sys
import os
sys.path.append(r'/home/icarrara/Documents/Project/reduced_dataset')  # Local
# sys.path.append(r'/home/icarrara/Documents/Project/autoregressive_bci')  # Server

import moabb
import mne
#new_path = '/home/icarrara/Documents/Project/autoregressive_bci/Data'  # Local
new_path = "/data/athena/user/icarrara/MOABB"  # Server
moabb.utils.set_download_dir(new_path)

import resource
from moabb.paradigms import MotorImagery
from moabb.datasets import BNCI2014004
from pyriemann.estimation import Covariances
from sklearn.pipeline import Pipeline
from moabb.evaluations import CrossSessionEvaluation
from sklearn.svm import SVC
from pyriemann.tangentspace import TangentSpace
from ACM.Augmented_Dataset import AugmentedDataset
from pyriemann.classification import MDM

# Load subject from txt file
import argparse
# Create the Parser
parser = argparse.ArgumentParser()
# Add the argument
parser.add_argument('subject', nargs='+', type=int)
# Execute the parse_args() method
param = parser.parse_args()
param = param.subject

# Initialize parameter for the Band Pass filter
fmin = 8
fmax = 35
tmin = 0
tmax = None

# Select the Subject
subjects = [int(param[0])]
# Load the dataset, right now you have added Nothing events to DATA using new stim channel STI
dataset = BNCI2014004()

events = ["right_hand", "left_hand"]

paradigm = MotorImagery(events=events, n_classes=len(events), fmin=fmin, fmax=fmax, tmax=tmax)

# Create a path and folder for every subject
dataset_name = "BNCI2014004"
path_ = "/home/icarrara/Documents/Project/reduced_dataset/classification_Full_nested/MOABB_lhrh_CS/" + dataset_name
path = os.path.join(path_, "MDOP_Subject_" + str(param[0]))
os.makedirs(path, exist_ok=True)
path2 = os.path.join(path, "../Figure/")
os.makedirs(path2, exist_ok=True)

# Pipelines
pipelines = {}
# Define the different algorithm to test and assign a name in the dictionary
pipelines["ACM+TGSP+SVM(MDOP)"] = Pipeline(steps=[
    ("augmenteddataset", AugmentedDataset()),
    ("Covariances", Covariances("cov")),
    ("Tangent_Space", TangentSpace(metric="riemann")),
    ("SVM", SVC(kernel="rbf"))
])

pipelines["ACM+MDM(MDOP)"] = Pipeline(steps=[
    ("augmenteddataset", AugmentedDataset()),
    ("Covariances", Covariances("cov")),
    ("MDM", MDM(metric=dict(mean='riemann', distance='riemann')))
])

# Evaluation For MOABB
# ========================================================================================================
dataset.subject_list = dataset.subject_list[int(param[0]) - 1:int(param[0])]
# Select an evaluation Within Session
evaluation = CrossSessionEvaluation(paradigm=paradigm,
                                    datasets=dataset,
                                    overwrite=False,
                                    random_state=42,
                                    hdf5_path=path,
                                    n_jobs=-1)

# Print the results
result = evaluation.process(pipelines, nested=True, takens="MDOP")

# Close file and save the result
# =================================================================================================================
# Save the final Results
result.to_csv(os.path.join(path, "results.cvs"))

# Print Information for the system at Terminal
sys.stdout = open(os.path.join(path, "Output.txt"), "w+")
print("Dataset BNCI2014004 \n")
print(
    "#################" + "\n"
    "List of selected events: " + "\n" + str(events) + "\n"
    "#################"
)
print(r"Subject Number: {}".format(param[0]))
end = dt.now()
elapsed = end-start
print("Overall Time for programm: %02d:%02d:%02d:%02d" % (elapsed.days, elapsed.seconds // 3600, elapsed.seconds // 60 % 60, elapsed.seconds % 60))
memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0
print("Overall Memory used: {} Mb".format(memory))
sys.stdout.close()
