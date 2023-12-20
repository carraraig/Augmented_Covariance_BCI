from datetime import datetime as dt
start = dt.now()

# Set up the Directory for made it run on a server.
import sys
import os
sys.path.append(r'/home/icarrara/Documents/Project/reduced_dataset')  # Local

import moabb
import mne
new_path = '/data/athena/user/icarrara/MOABB'  # Server
# new_path = '/home/icarrara/Documents/Project/DATA'
moabb.utils.set_download_dir(new_path)

# Load the library
import resource
from moabb.datasets import BNCI2015001
from moabb.paradigms import MotorImagery
import numpy as np
from moabb.evaluations import WithinSessionEvaluation
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow import keras
from ACM.StandardScaler import StandardScaler_Epoch
import random
import absl.logging
# Avoid output Warning
absl.logging.set_verbosity(absl.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load subject from txt file
import argparse
# Create the Parser
parser = argparse.ArgumentParser()
# Add the argument
parser.add_argument('subject', nargs='+', type=int)
# Execute the parse_args() method
param = parser.parse_args()
param = param.subject

# Create a path and folder for every subject
dataset_name = "BNCI2015001"
path_ = "/home/icarrara/Documents/Project/reduced_dataset/classification_Full_nested/MOABB_rf/" + dataset_name
path = os.path.join(path_, "STATE_ART_DL_Subject_" + str(param[0]))
os.makedirs(path, exist_ok=True)
path2 = os.path.join(path, "../Figure/")
os.makedirs(path2, exist_ok=True)

# Print Information for the system at Terminal
sys.stdout = open(os.path.join(path, "Output.txt"), "w+")

# Print Information Tensorflow
print(f"Tensorflow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")

CPU = len(tf.config.list_physical_devices("CPU")) > 0
print("CPU is", "AVAILABLE" if CPU else "NOT AVAILABLE")

GPU = len(tf.config.list_physical_devices("GPU")) > 0
print("GPU is", "AVAILABLE" if GPU else "NOT AVAILABLE")

# Set up reproducibility of Tensorflow
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)  # tf cpu fix seed
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


# Use SINGLE GPU
with tf.device('/device:GPU:0'):

    # Set Reproducibility
    setup_seed(42)
    # Initialize parameter for the Band Pass filter
    # Initialize parameter for the Band Pass filter
    fmin = 8
    fmax = 30
    tmin = 0
    tmax = None

    # Load the dataset
    dataset = BNCI2015001()

    events = ["right_hand", "feet"]

    # Select the paradigm
    paradigm = MotorImagery(events=events, n_classes=len(events), fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax)

    # Select the Subject
    subjects = [int(param[0])]

    # Pipelines, we can test the classifier with different pipelines.
    pipelines = {}
    # Define the different algorithm to test and assign a name in the dictionary
    # ====================================================================================================================
    from ACM.DL_Model import *

    EPOCHS = 300
    BATCH_SIZE = 64
    VERBOSE = 0
    RANDOM_STATE = 42
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2
    PATIENCE = 75
    LOSS = 'sparse_categorical_crossentropy'
    OPTIMIZER = tf.keras.optimizers.legacy.Adam(learning_rate=LEARNING_RATE)
    CALLBACK_ES = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE)
    CALLBACK_LR = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=PATIENCE)
    HISTORY_PLOT = True

    # ====================================================================================================================
    # ShallowConvNet
    # ====================================================================================================================
    pipelines["ShallowConvNet"] = Pipeline(steps=[
        ("Resampler_Epoch", Resampler_Epoch(250)),
        ("Convert_Epoch_Array", Convert_Epoch_Array()),
        ("StandardScaler", StandardScaler_Epoch()),
        ("ShallowConvNet", ShallowConvNet(loss=LOSS,
                                          optimizer=OPTIMIZER,
                                          epochs=EPOCHS,
                                          batch_size=BATCH_SIZE,
                                          verbose=VERBOSE,
                                          random_state=RANDOM_STATE,
                                          validation_split=VALIDATION_SPLIT,
                                          callbacks=[CALLBACK_ES, CALLBACK_LR],
                                          history_plot=HISTORY_PLOT
                                          )),
    ])

    # ====================================================================================================================
    # DeepConvNet
    # ====================================================================================================================
    pipelines["DeepConvNet"] = Pipeline(steps=[
        ("Resampler_Epoch", Resampler_Epoch(250)),
        ("Convert_Epoch_Array", Convert_Epoch_Array()),
        ("StandardScaler", StandardScaler_Epoch()),
        ("DeepConvNet", DeepConvNet(loss=LOSS,
                                    optimizer=OPTIMIZER,
                                    epochs=EPOCHS,
                                    batch_size=BATCH_SIZE,
                                    verbose=VERBOSE,
                                    random_state=RANDOM_STATE,
                                    validation_split=VALIDATION_SPLIT,
                                    callbacks=[CALLBACK_ES, CALLBACK_LR],
                                    history_plot=HISTORY_PLOT
                                    )),
    ])

    # ====================================================================================================================
    # EEGNet_8_2
    # ====================================================================================================================
    pipelines["EEGNet_8_2"] = Pipeline(steps=[
        ("Resampler_Epoch", Resampler_Epoch(128)),
        ("Convert_Epoch_Array", Convert_Epoch_Array()),
        ("StandardScaler", StandardScaler_Epoch()),
        ("EEGNet_8_2", EEGNet_8_2(loss=LOSS,
                                  optimizer=OPTIMIZER,
                                  epochs=EPOCHS,
                                  batch_size=BATCH_SIZE,
                                  verbose=VERBOSE,
                                  random_state=RANDOM_STATE,
                                  validation_split=VALIDATION_SPLIT,
                                  callbacks=[CALLBACK_ES, CALLBACK_LR],
                                  history_plot=HISTORY_PLOT
                                  )),
    ])

    # Evaluation For MOABB
    # ========================================================================================================
    dataset.subject_list = dataset.subject_list[int(param[0]) - 1:int(param[0])]

    # Select an evaluation Within Session
    evaluation_online = WithinSessionEvaluation(paradigm=paradigm,
                                                datasets=dataset,
                                                overwrite=False,
                                                random_state=42,
                                                hdf5_path=path,
                                                n_jobs=1,
                                                return_epochs=True
                                                )

    # Print the results
    result = evaluation_online.process(pipelines)

# Close file and save the result
# =================================================================================================================
# Save the final Results
result.to_csv(os.path.join(path, "results.cvs"))

# Print Information for the system at Terminal
print("Dataset BNCI2015001 \n")
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
