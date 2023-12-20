# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ANALYZE THE RESULTS with MOABB_lhrh Single DATASET Multiple Subject - FUCONE plot
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os
import pandas as pd
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Compute results for all the dataset BNCI2014001
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
path = "classification_Full_nested/MOABB_lhrh/BNCI2014001"
score = "score"
results1 = pd.read_csv(os.path.join(path, "FIX_STATE_ART_Subject_1/results.cvs"))
results2 = pd.read_csv(os.path.join(path, "FIX_STATE_ART_Subject_2/results.cvs"))
results3 = pd.read_csv(os.path.join(path, "FIX_STATE_ART_Subject_3/results.cvs"))
results4 = pd.read_csv(os.path.join(path, "FIX_STATE_ART_Subject_4/results.cvs"))
results5 = pd.read_csv(os.path.join(path, "FIX_STATE_ART_Subject_5/results.cvs"))
results6 = pd.read_csv(os.path.join(path, "FIX_STATE_ART_Subject_6/results.cvs"))
results7 = pd.read_csv(os.path.join(path, "FIX_STATE_ART_Subject_7/results.cvs"))
results8 = pd.read_csv(os.path.join(path, "FIX_STATE_ART_Subject_8/results.cvs"))
results9 = pd.read_csv(os.path.join(path, "FIX_STATE_ART_Subject_9/results.cvs"))
frames = [results1, results2, results3, results4, results5, results6, results7, results8, results9]
results_ = pd.concat(frames)

results_STATE_ART = results_[(results_["pipeline"] == "ACM+MDM(Grid)") |
                             (results_["pipeline"] == "MDM") |
                             (results_["pipeline"] == "TANG+SVM") |
                             (results_["pipeline"] == "ACM+TGSP+SVM(Grid)")
                             ]

results1 = pd.read_csv(os.path.join(path, "FIX_MDOP_Subject_1/results.cvs"))
results2 = pd.read_csv(os.path.join(path, "FIX_MDOP_Subject_2/results.cvs"))
results3 = pd.read_csv(os.path.join(path, "FIX_MDOP_Subject_3/results.cvs"))
results4 = pd.read_csv(os.path.join(path, "FIX_MDOP_Subject_4/results.cvs"))
results5 = pd.read_csv(os.path.join(path, "FIX_MDOP_Subject_5/results.cvs"))
results6 = pd.read_csv(os.path.join(path, "FIX_MDOP_Subject_6/results.cvs"))
results7 = pd.read_csv(os.path.join(path, "FIX_MDOP_Subject_7/results.cvs"))
results8 = pd.read_csv(os.path.join(path, "FIX_MDOP_Subject_8/results.cvs"))
results9 = pd.read_csv(os.path.join(path, "FIX_MDOP_Subject_9/results.cvs"))
frames = [results1, results2, results3, results4, results5, results6, results7, results8, results9]
results_ = pd.concat(frames)

results_MDOP = results_[(results_["pipeline"] == "ACM+MDM(MDOP)") |
                        (results_["pipeline"] == "ACM+TGSP+SVM(MDOP)")
                        ]

frames_ = [results_STATE_ART, results_MDOP]
results_ALL = pd.concat(frames_)

path1 = os.path.join(path, "Results/")
os.makedirs(path1, exist_ok=True)

# Save the framework for all dataset in a CVS
results_ALL.to_csv(os.path.join(path, "Results/results_CompTime.cvs"))

# Compute summary statistics (Time is in seconds
time = results_ALL.groupby(['pipeline'], as_index=False)["time"].mean()
std_time = results_ALL.groupby(['pipeline'], as_index=False)["time"].std()
time['std_time'] = std_time["time"]
score_ = results_ALL.groupby(['pipeline'], as_index=False)[score].mean()
std_score = results_ALL.groupby(['pipeline'], as_index=False)[score].std()
score_['std_score'] = std_score[score]
results_pipeline = pd.merge(time, score_, on="pipeline")
print(results_pipeline)
results_pipeline.to_csv(os.path.join(path, "Results/statistic_summary_CompTime.cvs"))

# Create Folder
path2 = os.path.join(path, "Figure/")
os.makedirs(path2, exist_ok=True)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Compute the plot - ALL subject
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# To not show pictures
#matplotlib.use('Agg')

order_list = [
    "ACM+MDM(Grid)",
    "ACM+MDM(MDOP)",
    "ACM+TGSP+SVM(Grid)",
    "ACM+TGSP+SVM(MDOP)",
]



ax = sns.scatterplot(data=results_pipeline, x="time", y="score", hue="pipeline")
plt.xlabel('Time for 1-fold (s)')
plt.ylabel("ROC AUC")
ax.grid(False)
plt.savefig(os.path.join(path, "Figure/Time") + ".pdf", dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(path, "Figure/Time") + ".png", dpi=300, bbox_inches='tight')
##


# Bar Plot TOTAL
"""plt.errorbar(results_pipeline[score],
             results_pipeline["time"],
             yerr=results_pipeline["std_time"],
             xerr=results_pipeline["std_score"],
             label=results_pipeline["pipeline"],
             fmt='o')"""

plt.errorbar(results_pipeline[score],
             results_pipeline["time"],
             yerr=results_pipeline["std_time"],
             xerr=results_pipeline["std_score"],
             label=results_pipeline["pipeline"],
             fmt='o')

plt.legend(loc= "upper left")
plt.xlabel("Score")
plt.ylabel("Time")
plt.show()
##

