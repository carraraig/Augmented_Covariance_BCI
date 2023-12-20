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
    barplot_tot,
    score_plot_Box
)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Compute results for all the dataset BNCI2014001
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
path = "classification_Full_nested/MOABB_lhrh/"
score = "score"
results1 = pd.read_csv(os.path.join(path, 'BNCI2014001/Results/results.cvs'))
results2 = pd.read_csv(os.path.join(path, 'BNCI2014004/Results/results.cvs'))
results3 = pd.read_csv(os.path.join(path, 'Zhou2016/Results/results.cvs'))
frames = [results1, results2, results3]
results_ALL = pd.concat(frames)

# Save the framework for all dataset in a CVS
os.makedirs(os.path.join(path, "Results/"), exist_ok=True)
path1 = os.path.join(path, "Results/StateArt")
os.makedirs(path1, exist_ok=True)
results_ALL.to_csv(os.path.join(path, "Results/StateArt/results.cvs"))


# Compute Statistics
stats_ALL = compute_dataset_statistics(results_ALL)
stats_ALL.to_csv(os.path.join(path, "Results/StateArt/statistic.cvs"))

# Compute summary statistics
results_pipeline = results_ALL.groupby(['pipeline'], as_index=False)[score].mean()
results_pipeline_std = results_ALL.groupby(['pipeline'], as_index=False)[score].std()
results_pipeline['std'] = results_pipeline_std[score]
print(results_pipeline)
results_pipeline.to_csv(os.path.join(path, "Results/StateArt/statistic_summary.cvs"))

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
sns.set_theme(context="paper", style="white", font_scale=4, palette="tab10")
plot_violin(
    results_ALL,
    filename=os.path.join(path, "Figure/violin"),
    order_list=order_list,
    score=score
    )

# Box Plot Total
sns.set_theme(context="paper", style="white", font_scale=2, palette="tab10")
score_plot_Box(results_ALL,
               filename = os.path.join(path, "Figure/BoxPlot"),
               orientation="h",
               order_list=order_list,
               # legend="off",
               # title = "Right Hand vs Feet WithinSession"
               )

score_plot_Box(results_ALL,
               filename = os.path.join(path, "Figure/BoxPlot_Nolegend"),
               orientation="h",
               order_list=order_list,
               legend="off",
               # title = "Right Hand vs Feet WithinSession"
               )

score_plot_Box(results_ALL,
               filename = os.path.join(path, "Figure/BoxPlot_LegendOutside"),
               orientation="h",
               order_list=order_list,
               legend="outside",
               # title = "Right Hand vs Feet WithinSession"
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

