import streamlit as st
import awesome_streamlit as ast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from graphviz import Source
from IPython.display import SVG
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    MultiTaskLasso,
    PassiveAggressiveRegressor,
    Ridge,
    SGDRegressor,
)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, export_graphviz, plot_tree


def write():
    happiness_2015 = pd.read_csv("./data/2015.csv")

    variables = [
        "Economy (GDP per Capita)",
        "Family",
        "Health (Life Expectancy)",
        "Freedom",
        "Trust (Government Corruption)",
        "Generosity",
    ]

    reduced_2015 = happiness_2015[variables]

    st.header("2015 Data Analysis")
    st.subheader("Influence of each variable on the happiness")

    fig, axs = plt.subplots(
        ncols=2, nrows=len(variables) // 2, figsize=(16, 2 * len(variables))
    )

    for i, column in enumerate(variables):
        sns.regplot(
            data=happiness_2015, y="Happiness Score", x=column, ax=axs[i // 2, i % 2]
        )

    st.pyplot(fig)

    st.subheader("Correlation plots of the different variables by region")

    regions = happiness_2015["Region"].unique()

    fig, axs = plt.subplots(
        ncols=2, nrows=len(regions) // 2, figsize=(12, 2 * len(regions))
    )
    fig.tight_layout(pad=10)
    fig.autofmt_xdate(rotation=45)

    reduced_with_region = happiness_2015[variables + ["Region"]]

    for i, region in enumerate(regions):
        axs[i // 2, i % 2].set_title(f"{region} correlation plot")
        sns.heatmap(
            data=reduced_with_region[reduced_with_region["Region"] == region].corr(),
            ax=axs[i // 2, i % 2],
        )

    st.pyplot(fig)

    st.subheader("Happiness by region")

    plot = sns.catplot(
        data=happiness_2015, kind="box", x="Region", y="Happiness Score", aspect=2
    )
    plot.set_xticklabels(rotation=45)

    st.pyplot()

    st.subheader("Happiness vs family score")

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.scatterplot(
        data=happiness_2015,
        y="Happiness Score",
        x="Family",
        hue="Region",
        size="Economy (GDP per Capita)",
        ax=ax,
    )
    plt.legend(loc="center right", bbox_to_anchor=(1.65, 0.5), ncol=2)

    st.pyplot()
