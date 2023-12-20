# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ANALYZE THE RESULTS with MOABB_lhrh Single DATASET Multiple Subject - FUCONE plot
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os
import moabb.analysis.plotting as moabb_plt
from moabb.analysis.meta_analysis import (
    compute_dataset_statistics,
)
import pandas as pd
from ACM.MOABB_Plot import (
    plot_results_compute_dataset_statistics,
    plot_violin,
    barplot_tot
)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Compute results for all the dataset BNCI2014001
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
path = "classification_Full_nested/MOABB_rf_CS/BNCI2015001"
score = "score"
results1 = pd.read_csv(os.path.join(path, "STATE_ART_Subject_1/results.cvs"))
results2 = pd.read_csv(os.path.join(path, "STATE_ART_Subject_2/results.cvs"))
results3 = pd.read_csv(os.path.join(path, "STATE_ART_Subject_3/results.cvs"))
results4 = pd.read_csv(os.path.join(path, "STATE_ART_Subject_4/results.cvs"))
results5 = pd.read_csv(os.path.join(path, "STATE_ART_Subject_5/results.cvs"))
results6 = pd.read_csv(os.path.join(path, "STATE_ART_Subject_6/results.cvs"))
results7 = pd.read_csv(os.path.join(path, "STATE_ART_Subject_7/results.cvs"))
results8 = pd.read_csv(os.path.join(path, "STATE_ART_Subject_8/results.cvs"))
results9 = pd.read_csv(os.path.join(path, "STATE_ART_Subject_9/results.cvs"))
results10 = pd.read_csv(os.path.join(path, "STATE_ART_Subject_10/results.cvs"))
results11 = pd.read_csv(os.path.join(path, "STATE_ART_Subject_11/results.cvs"))
results12 = pd.read_csv(os.path.join(path, "STATE_ART_Subject_12/results.cvs"))
frames = [results1, results2, results3, results4, results5, results6, results7, results8, results9,
          results10, results11, results12]
results_STATE_ART = pd.concat(frames)

results1 = pd.read_csv(os.path.join(path, "MDOP_Subject_1/results.cvs"))
results2 = pd.read_csv(os.path.join(path, "MDOP_Subject_2/results.cvs"))
results3 = pd.read_csv(os.path.join(path, "MDOP_Subject_3/results.cvs"))
results4 = pd.read_csv(os.path.join(path, "MDOP_Subject_4/results.cvs"))
results5 = pd.read_csv(os.path.join(path, "MDOP_Subject_5/results.cvs"))
results6 = pd.read_csv(os.path.join(path, "MDOP_Subject_6/results.cvs"))
results7 = pd.read_csv(os.path.join(path, "MDOP_Subject_7/results.cvs"))
results8 = pd.read_csv(os.path.join(path, "MDOP_Subject_8/results.cvs"))
results9 = pd.read_csv(os.path.join(path, "MDOP_Subject_9/results.cvs"))
results10 = pd.read_csv(os.path.join(path, "MDOP_Subject_10/results.cvs"))
results11 = pd.read_csv(os.path.join(path, "MDOP_Subject_11/results.cvs"))
results12 = pd.read_csv(os.path.join(path, "MDOP_Subject_12/results.cvs"))
frames = [results1, results2, results3, results4, results5, results6, results7, results8, results9,
          results10, results11, results12]
results_MDOP = pd.concat(frames)

results1 = pd.read_csv(os.path.join(path, "STATE_ART_DL_Subject_1/results.cvs"))
results2 = pd.read_csv(os.path.join(path, "STATE_ART_DL_Subject_2/results.cvs"))
results3 = pd.read_csv(os.path.join(path, "STATE_ART_DL_Subject_3/results.cvs"))
results4 = pd.read_csv(os.path.join(path, "STATE_ART_DL_Subject_4/results.cvs"))
results5 = pd.read_csv(os.path.join(path, "STATE_ART_DL_Subject_5/results.cvs"))
results6 = pd.read_csv(os.path.join(path, "STATE_ART_DL_Subject_6/results.cvs"))
results7 = pd.read_csv(os.path.join(path, "STATE_ART_DL_Subject_7/results.cvs"))
results8 = pd.read_csv(os.path.join(path, "STATE_ART_DL_Subject_8/results.cvs"))
results9 = pd.read_csv(os.path.join(path, "STATE_ART_DL_Subject_9/results.cvs"))
results10 = pd.read_csv(os.path.join(path, "STATE_ART_DL_Subject_10/results.cvs"))
results11 = pd.read_csv(os.path.join(path, "STATE_ART_DL_Subject_11/results.cvs"))
results12 = pd.read_csv(os.path.join(path, "STATE_ART_DL_Subject_12/results.cvs"))
frames = [results1, results2, results3, results4, results5, results6, results7, results8, results9,
          results10, results11, results12]
results_STATE_ART_DL = pd.concat(frames)

frames_ = [results_STATE_ART, results_MDOP, results_STATE_ART_DL]
results_ALL = pd.concat(frames_)

path1 = os.path.join(path, "Results/")
os.makedirs(path1, exist_ok=True)

# Save the framework for all dataset in a CVS
results_ALL.to_csv(os.path.join(path, "Results/results.cvs"))

# Compute Statistics
stats_ALL = compute_dataset_statistics(results_ALL)
stats_ALL.to_csv(os.path.join(path, "Results/statistic.cvs"))

# Compute summary statistics
results_pipeline = results_ALL.groupby(['pipeline'], as_index=False)[score].mean()
results_pipeline_std = results_ALL.groupby(['pipeline'], as_index=False)[score].std()
results_pipeline['std'] = results_pipeline_std[score]
print(results_pipeline)
results_pipeline.to_csv(os.path.join(path, "Results/statistic_summary.cvs"))

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
    "CSP+LDA",
    "FgMDM",
    "MDM",
    "Cov+EN",
    "TANG+SVM",
    "EEGNet_8_2",
    "DeepConvNet",
    "ShallowConvNet"
]

# plot dataset statistics2
sns.set_theme(context="paper", style="white", font_scale=2.5, palette="tab10")
plot_results_compute_dataset_statistics(stats_ALL, 
                                        os.path.join(path, "Figure/dataset_statistics"), 
                                        annotation=True,
                                        order_list=order_list)

# plot rain clouds
sns.set_theme(context="paper", style="white", font_scale=1, palette="tab10")
plot_violin(
    results_ALL,
    filename=os.path.join(path, "Figure/violin"),
    order_list=order_list,
    score=score
    )

# Bar Plot TOTAl
sns.set_theme(context="paper", style="white", font_scale=1, palette="tab10")
barplot_tot(results_ALL,
            filename=os.path.join(path, "Figure/score_plot3"),
            order_list=order_list,
            score=score
            )

sns.set_theme(context="paper", style="white", font_scale=1, palette="Pastel2")


# Meta analysis plot
fig = moabb_plt.meta_analysis_plot(stats_ALL, "ACM+TGSP+SVM(Grid)", "CSP+LDA")
plt.savefig(os.path.join(path, "Figure/meta_plot_TANG.pdf"), dpi=300)

# Meta analysis plot
fig = moabb_plt.meta_analysis_plot(stats_ALL, "ACM+TGSP+SVM(Grid)", "FgMDM")
plt.savefig(os.path.join(path, "Figure/meta_plot_FgMDM.pdf"), dpi=300)

# Meta analysis plot
fig = moabb_plt.meta_analysis_plot(stats_ALL, "ACM+TGSP+SVM(Grid)", "Cov+EN")
plt.savefig(os.path.join(path, "Figure/meta_plot_CovEN.pdf"), dpi=300)

# Meta analysis plot
fig = moabb_plt.meta_analysis_plot(stats_ALL, "ACM+TGSP+SVM(Grid)", "ShallowConvNet")
plt.savefig(os.path.join(path, "Figure/meta_plot_Shallow.pdf"), dpi=300)

# Meta analysis plot
fig = moabb_plt.meta_analysis_plot(stats_ALL, "ACM+TGSP+SVM(Grid)", "DeepConvNet")
plt.savefig(os.path.join(path, "Figure/meta_plot_Deep.pdf"), dpi=300)

# Meta analysis plot
fig = moabb_plt.meta_analysis_plot(stats_ALL, "ACM+TGSP+SVM(Grid)", "EEGNet_8_2")
plt.savefig(os.path.join(path, "Figure/meta_plot_EEGNet.pdf"), dpi=300)

##

