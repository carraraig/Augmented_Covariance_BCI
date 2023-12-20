import matplotlib.pyplot as plt
import seaborn as sns
from moabb.analysis.meta_analysis import (
    collapse_session_scores,
    compute_dataset_statistics,
    find_significant_differences,
)

from moabb.analysis.plotting import _simplify_names
import logging
log = logging.getLogger(__name__)
import pandas as pd
import numpy as np
# import ptitprince as pt
import os
import joblib
import matplotlib
# from statannotations.Annotator import Annotator


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# FUNCTION DEFINITION
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def simplify_names(x):
    if len(x) > 10:
        return x.split(" ")[0]
    else:
        return x

def plot_results_compute_dataset_statistics(stats, filename, annotation=True, order_list=None):
    P, T = find_significant_differences(stats)

    # plt.style.use("classic")
    columns = stats["pipe1"].unique()
    rows = stats["pipe2"].unique()
    if len(order_list) == len(stats["pipe1"].unique()):
        columns =order_list
        rows = order_list
    pval_heatmap = pd.DataFrame(columns=columns, index=rows, data=P)
    tval_heatmap = pd.DataFrame(columns=columns, index=rows, data=T)

    mask = np.invert((pval_heatmap < 0.05))
    # vmin = -max(abs(tval_heatmap.min().min()), abs(tval_heatmap.max().max()))
    vmax = max(abs(tval_heatmap.min().min()), abs(tval_heatmap.max().max()))
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(24, 18))
        ax = sns.heatmap(
            tval_heatmap,
            mask=mask,
            annot=annotation,
            fmt=".3f",
            cmap="plasma",
            linecolor='black',
            linewidths=1,
            vmin=0,
            vmax=vmax,
            cbar_kws={"label": "signif. t-val (p<0.05)"},
        )
        # bottom, top = ax.get_ylim()
        # ax.set_ylim(bottom + 0.6, top - 0.6)
        # ax.set_facecolor("white")
        # ax.xaxis.label.set_size(9)
        # ax.yaxis.label.set_size(9)
        plt.savefig(filename + ".pdf", dpi=300, bbox_inches='tight')
        plt.savefig(filename + ".png", dpi=300, bbox_inches='tight')


def plot_results_compute_dataset_statistics2(stats, filename, simplify=True):
    P, T = find_significant_differences(stats)

    # plt.style.use("classic")
    if simplify:
        T.columns = T.columns.map(simplify_names)
        P.columns = P.columns.map(simplify_names)
    annot_df = T.copy()
    for row in annot_df.index:
        for col in annot_df.columns:
            if T.loc[row, col] > 0:
                txt = "{:.3f}\np={:1.0e}".format(
                    T.loc[row, col], P.loc[row, col]
                )
            else:
                # we need the effect direction and p-value to coincide.
                if P.loc[row, col] < 0.05:
                    P.loc[row, col] = 1e-110
                txt = ""
            annot_df.loc[row, col] = txt
    # palette = sns.color_palette("crest", as_cmap=True)
    palette = sns.color_palette("crest", as_cmap=True)
    palette.set_under(color=[1, 1, 1])
    palette.set_over(color=[0.5, 0, 0])
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(24, 18))
        ax = sns.heatmap(
            data=-np.log(P),
            annot=annot_df,
            fmt="",
            cmap=palette,
            linewidths=1,
            linecolor="0.8",
            annot_kws={"size": 10},
            cbar=False,
            vmin=-np.log(0.05),
            vmax=-np.log(1e-100),
        )
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.6, top - 0.6)
    ax.set_facecolor("white")
    # ax.xaxis.label.set_size(9)
    # ax.yaxis.label.set_size(9)
    plt.savefig(filename + ".pdf", dpi=300, bbox_inches='tight')
    plt.savefig(filename + ".png", dpi=300, bbox_inches='tight')


def plot_violin(df_results, filename, order_list=None, score="score"):

    # sns.set_context()
    # sns.set_context("talk")
    # sns.set_theme(style="white", font_scale=2.5)

    ort = "h"
    sigma = 0.2
    dx = score
    dy = "pipeline"
    dhue = "pipeline"
    f, ax = plt.subplots(figsize=(24, 18))
    ax = sns.violinplot(
        x=dx,
        y=dy,
        hue=dhue,
        data=df_results,
        bw=sigma,
        width_viol=0.7,
        ax=ax,
        orient=ort,
        dodge=True,
        pointplot=True,
        move=0.2,
        order=order_list
    )
    plt.setp(ax.collections, alpha=.65)
    ax.get_legend().remove()
    # ax.xaxis.label.set_size(9)
    # ax.yaxis.label.set_size(9)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # plt.yticks(range(len(df_results["pipeline"].unique())))
    plt.yticks(range(len(df_results["pipeline"].unique())), rotation=45)
    # plt.xticks(fontsize=18)
    # plt.ylabel("Pipeline")
    plt.xlabel(score)
    plt.savefig(filename + ".pdf", dpi=300, bbox_inches='tight')
    plt.savefig(filename + ".png", dpi=300, bbox_inches='tight')


def score_plot_Box(data, filename, pipelines=None, order_list=None, legend=True, orientation="vertical", title=None):
    """Plot scores for all pipelines and all datasets

    Parameters
    ----------
    data: output of Results.to_dataframe()
        results on datasets
    pipelines: list of str | None
        pipelines to include in this plot
    orientation: str, default="vertical"
        plot orientation, could be ["vertical", "v", "horizontal", "h"]

    Returns
    -------
    fig: Figure
        Pyplot handle
    color_dict: dict
        Dictionary with the facecolor
    """
    data = collapse_session_scores(data)
    unique_ids = data["dataset"].apply(_simplify_names)
    if len(unique_ids) != len(set(unique_ids)):
        log.warning("Dataset names are too similar, turning off name shortening")
    else:
        data["dataset"] = unique_ids

    if pipelines is not None:
        data = data[data.pipeline.isin(pipelines)]

    if orientation in ["horizontal", "h"]:
        y, x = "dataset", "score"
        fig = plt.figure(figsize=(11, 11))
    elif orientation in ["vertical", "v"]:
        x, y = "dataset", "score"
        fig = plt.figure(figsize=(11, 11))
    else:
        raise ValueError("Invalid plot orientation selected!")

    ax = fig.add_subplot(111)
    ax = sns.boxplot(
        data=data,
        y=y,
        x=x,
        hue="pipeline",
        dodge=True,
        ax=ax,
        hue_order=order_list
    )

    # Annotation, Work with orientation "v"
    # https://levelup.gitconnected.com/statistics-on-seaborn-plots-with-statannotations-2bfce0394c00
    # pairs = [
    #    [("001-2014", "ACM+TGSP+SVM"), ("001-2014", "Cov+EN")]
    # ]
    # pvalues = [
    #    0.12500
    # ]
    # formatted_pvalues = [f'**' for pvalue in pvalues]
    # annotator = Annotator(ax, pairs, data=data, x=x, y=y, hue_order=order_list, hue="pipeline")
    # annotator.set_custom_annotations(formatted_pvalues)
    # annotator.annotate()
    # annotator.configure(test="Mann-Whitney").apply_and_annotate()
    # add_stat_annotation(ax, data=data, x=x, y=y, hue_order=order_list, hue="pipeline",
    #                    box_pairs=pairs, text_format='star', loc='outside', verbose=2, test='Mann-Whitney')
    if orientation in ["horizontal", "h"]:
        ax.set_xlim([0.4, 1])
        ax.axvline(0.5, linestyle="--", color="k", linewidth=2)
    else:
        ax.set_ylim([0.4, 1])
        ax.axhline(0.5, linestyle="--", color="k", linewidth=2)
    if title is not None:
        ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    if legend == "off":
        ax.get_legend().remove()  ## remove legend
    elif legend == "outside":
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    color_dict = {lb: h.get_facecolor()[0] for lb, h in zip(labels, handles)}
    plt.tight_layout()
    plt.savefig(filename + ".pdf", dpi=300, bbox_inches='tight')
    plt.savefig(filename + ".png", dpi=300, bbox_inches='tight')

    return fig, color_dict

"""def plot_rainclouds(df_results, filename, order_list=None, score="score"):

    # sns.set_context()
    # sns.set_context("talk")
    # sns.set_theme(style="white", font_scale=2.5)

    ort = "h"
    sigma = 0.2
    dx = "pipeline"
    dy = score
    dhue = "pipeline"
    f, ax = plt.subplots(figsize=(24, 18))
    ax = pt.RainCloud(
        x=dx,
        y=dy,
        hue=dhue,
        data=df_results,
        bw=sigma,
        width_viol=0.7,
        ax=ax,
        orient=ort,
        alpha=0.65,
        dodge=True,
        pointplot=True,
        move=0.2,
        order=order_list
    )
    ax.get_legend().remove()
    # ax.xaxis.label.set_size(9)
    # ax.yaxis.label.set_size(9)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # plt.yticks(range(len(df_results["pipeline"].unique())))
    plt.yticks(range(len(df_results["pipeline"].unique())), rotation=45)
    # plt.xticks(fontsize=18)
    # plt.ylabel("Pipeline")
    plt.xlabel(score)
    plt.savefig(filename + ".pdf", dpi=300, bbox_inches='tight')
    plt.savefig(filename + ".png", dpi=300, bbox_inches='tight')

"""

def barplot_tot(results_ALL, filename, order_list=None, score="score"):
    # Bar Plot Total
    # f, ax1 = plt.subplots(figsize=(24, 18))
    ax1 = sns.catplot(
        kind="bar",
        y=score,
        x="dataset",
        hue="pipeline",
        alpha=0.65,
        data=results_ALL,
        hue_order=order_list
    )
    plt.savefig(filename + ".pdf", dpi=300, bbox_inches='tight')
    plt.savefig(filename + ".png", dpi=300, bbox_inches='tight')



def score_plot(data, pipelines=None, score="score"):
    data = collapse_session_scores(data)
    data["dataset"] = data["dataset"].apply(simplify_names)
    if pipelines is not None:
        data = data[data.pipeline.isin(pipelines)]
    fig = plt.figure(figsize=(24, 18))
    ax = fig.add_subplot(111)
    # markers = ['o', '8', 's', 'p', '+', 'x', 'D', 'd', '>', '<', '^']
    sns.stripplot(
        data=data,
        x="dataset",
        y=score,
        jitter=0.15,
        palette='tab10',
        hue="pipeline",
        dodge=True,
        ax=ax,
        alpha=0.7,
    )
    ax.set_ylim([0, 1])
    ax.grid(axis='y')
    # ax.axvline(0.5, linestyle="--", color="k", linewidth=2)
    # ax.set_title("Scores per dataset and algorithm")
    # handles, labels = ax.get_legend_handles_labels()
    # color_dict = {lb: h.get_facecolor()[0] for lb, h in zip(labels, handles)}
    plt.legend(loc=0)
    return fig


def learning_curver(results, n_subs, filename, score="score"):
    fig, ax = plt.subplots(facecolor="white", figsize=[24, 18])

    if n_subs > 1:
        r = results.groupby(["pipeline", "subject", "data_size"]).mean().reset_index()
    else:
        r = results

    sns.pointplot(data=r, x="data_size", y=score, hue="pipeline", ax=ax, palette="Set1")

    errbar_meaning = "subjects" if n_subs > 1 else "permutations"
    title_str = f"Errorbar shows Mean-CI across {errbar_meaning}"
    ax.set_xlabel("Amount of training samples")
    ax.set_ylabel("ROC AUC")
    ax.set_title(title_str)
    fig.tight_layout()
    plt.grid()
    plt.savefig(filename + ".pdf", dpi=300, bbox_inches='tight')
    plt.savefig(filename + ".png", dpi=300, bbox_inches='tight')
