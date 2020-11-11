import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
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

# Disable deprecation warnings when calling st.pyplot() without a figure
st.set_option("deprecation.showPyplotGlobalUse", False)

st.title("World Happiness analysis")

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


# Influence of each variable on the happiness

fig, axs = plt.subplots(
    ncols=2, nrows=len(variables) // 2, figsize=(16, 2 * len(variables))
)

for i, column in enumerate(variables):
    sns.regplot(
        data=happiness_2015, y="Happiness Score", x=column, ax=axs[i // 2, i % 2]
    )

st.pyplot(fig)

# Correlation plots of the different variables by region

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

# Happiness by region

plot = sns.catplot(
    data=happiness_2015, kind="box", x="Region", y="Happiness Score", aspect=2
)
plot.set_xticklabels(rotation=45)

st.pyplot()


# Happiness vs family score

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

## ML testing

### Helpers

errors = pd.DataFrame(data={"name": [], "mse": [], "r2": []})


def pred_real_plot(predictions, test_y):
    """
    predictions: prediction list from model.predict()
    test_y: ground truth

    The dotted represents a perfect result, the farther apart
    from that line the dots are, the more the model predicted
    a wrong result.
    """
    pred_real = list(zip(predictions, test_y.values.flatten()))
    g = sns.jointplot(x=predictions, y=test_y.values.flatten())
    g.ax_joint.set_xlabel("ground truth")
    g.ax_joint.set_ylabel("predictions")

    x0, x1 = g.ax_joint.get_xlim()
    y0, y1 = g.ax_joint.get_ylim()
    lims = [max(x0, y0), min(x1, y1)]
    g.ax_joint.plot(lims, lims, ":k")
    st.pyplot()


def regression(reg, train_x, train_y, test_x, test_y, name):
    """
    reg: regression model to use
    train_: training data
    test_: testing data
    name: model name

    This will print the different scores of the models

    There are currently two scoring methods used:
    - mse: mean squared error
    - r2: r squared
    Measures how close we are to the regression line
    """
    reg.fit(train_x, train_y)
    predictions = reg.predict(test_x)
    mse = mean_squared_error(test_y, predictions)
    r2 = r2_score(test_y, predictions)
    st.write(name, "error:")
    st.write("MSE:", mse)
    st.write("R^2:", r2)

    errors = {"name": name, "mse": mse, "r2": r2}

    return predictions, errors


### Split train/test
train, test = train_test_split(happiness_2015, test_size=0.2, random_state=42)

train_x = train[variables]
train_y = train[["Happiness Score"]]

test_x = test[variables]
test_y = test[["Happiness Score"]]


### Linear Regression

pred, error = regression(
    LinearRegression(), train_x, train_y, test_x, test_y, "linear regression"
)

errors = errors.append(error, ignore_index=True)

pred_real_plot(pred.flatten(), test_y)


### Ridge Regression


pred, error = regression(Ridge(), train_x, train_y, test_x, test_y, "ridge regression")

errors = errors.append(error, ignore_index=True)

pred_real_plot(pred.flatten(), test_y)


### Decision Tree Regressor

tree = DecisionTreeRegressor(max_depth=4)
pred, error = regression(
    tree, train_x, train_y, test_x, test_y, "decision tree regressor"
)

errors = errors.append(error, ignore_index=True)


graph = Source(export_graphviz(tree, out_file=None, feature_names=variables))
graph.format = "png"
graph.render("decision_tree")
from PIL import Image

image = Image.open("decision_tree.png")
st.image(image, caption="Decision Tree", use_column_width=True)

pred_real_plot(pred, test_y)


### Random Forest Regressor

pred, error = regression(
    RandomForestRegressor(max_depth=4, random_state=42),
    train_x,
    train_y.values.ravel(),
    test_x,
    test_y,
    "random forest regressor",
)

errors = errors.append(error, ignore_index=True)

pred_real_plot(pred, test_y)


### SGD

pred, error = regression(
    SGDRegressor(max_iter=10000, tol=1e-5),
    train_x,
    train_y.values.ravel(),
    test_x,
    test_y,
    "stochastic gradien descent",
)

errors = errors.append(error, ignore_index=True)

pred_real_plot(pred, test_y)


### Support Vector Regression

pred, error = regression(
    SVR(), train_x, train_y.values.ravel(), test_x, test_y, "support vector regression"
)

errors = errors.append(error, ignore_index=True)

pred_real_plot(pred, test_y)


### MLP Regressor

pred, error = regression(
    MLPRegressor(random_state=4, hidden_layer_sizes=(100,)),
    train_x,
    train_y.values.ravel(),
    test_x,
    test_y,
    "mlp regressor",
)

errors = errors.append(error, ignore_index=True)

pred_real_plot(pred, test_y)


### Voting Regressor

lr = LinearRegression()
rfr = RandomForestRegressor(max_depth=5, random_state=42)
sgd = SGDRegressor(max_iter=10000, tol=1e-5)
mlp = MLPRegressor(random_state=4)

models = [("lr", lr), ("rf", rfr), ("sgd", sgd), ("mlp", mlp)]

voting_reg = VotingRegressor(estimators=models)

pred, error = regression(
    voting_reg, train_x, train_y.values.ravel(), test_x, test_y, "voting regressor"
)

errors = errors.append(error, ignore_index=True)

pred_real_plot(pred, test_y)


## Compare ML models

errors.set_index("name", inplace=True)

st.dataframe(
    errors[["mse"]]
    .T.style.highlight_min(axis=1, color="lightgreen")
    .highlight_max(axis=1, color="lightcoral")
)

st.dataframe(
    errors[["r2"]]
    .T.style.highlight_max(axis=1, color="lightgreen")
    .highlight_min(axis=1, color="lightcoral")
)
