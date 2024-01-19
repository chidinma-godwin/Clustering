#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 17:56:06 2024

@author: Chidex
"""

import pandas as pd
import numpy as np

from matplotlib import colormaps as cm
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
import sklearn.cluster as cluster
import sklearn.metrics as skmet


def read_and_clean_data(filename):
    """
    Reads a data file into a dataframe, cleans and transposes
    the dataframe

    Parameters
    ----------
    filename : str
        The name of the data file to read into DataFrame.

    Returns
    -------
    cleaned_df : DataFrame
        The cleaned dataframe.
    cleaned_df_transposed : DataFrame
        The cleaned and transposed dataframe.

    """

    df = pd.read_csv(filename)
    df.drop(columns=["Country Code", "Series Code"], inplace=True)

    df.set_index(["Country Name", "Series Name"], inplace=True)

    # Rename columns, so that columns like "2002 [YR2002]" becomes "2002"
    df.columns = df.columns.str.split(" \[").str[0]

    df.rename(index={"Access to electricity (% of population)":
                     "Access to electricity",
                     "Control of Corruption: Percentile Rank":
                     "Control of Corruption",
                     "GNI per capita, PPP (current international $)":
                     "GNI per capita",
                     "Life expectancy at birth, total (years)":
                     "Life expectancy at birth",
                     "Political Stability and Absence of Violence/Terrorism" +
                     ": Percentile Rank": "Political Stability",
                     "Population growth (annual %)": "Population growth"},
              level=1, inplace=True)

    df_transposed = df.transpose()

    return df, df_transposed


def show_correlation(corr):
    """
    Plots heatmap to show the correlation between indicators

    Parameters
    ----------
    corr : DataFrame
        A dataframe containg the correlation coeffiecients of the indicators

    Returns
    -------
    None.

    """

    fig, ax = plt.subplots(figsize=(12, 10), layout="constrained")

    im = ax.imshow(corr, interpolation="nearest")

    # Add a colorbar to the plot
    fig.colorbar(im, orientation="vertical", fraction=0.045)

    columns_length = len(corr.columns)

    # Set the plot title
    ax.set_title("Correlation of Indicators", fontweight="bold", fontsize=25,
                 y=1.05)

    # Show all ticks and label them with the column name
    ax.set_xticks(np.arange(columns_length), labels=corr.columns, fontsize=15)
    ax.set_yticks(np.arange(columns_length), labels=corr.columns, fontsize=15)

    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=-90, ha="left")

    # The threshold for which to change the text color to ensure visibility
    threshold = im.norm(corr.to_numpy().max())/2

    # Loop over the data and create text annotations
    for i in range(columns_length):
        for j in range(columns_length):
            # Set the text color based on the color of the box
            color = ("white", "black")[
                int(im.norm(corr.to_numpy()[i, j]) > threshold)]

            ax.text(j, i, corr.to_numpy()[i, j], ha="center", va="center",
                    color=color, fontsize=20, fontweight="bold")

    plt.savefig("heatmap.png", bbox_inches="tight")

    plt.show()

    return


def make_boxplot(df, title, n_cols):
    """
    Makes a box plot from the provided dataframe

    Parameters
    ----------
    df : DataFrame
        The dataframe whose columns will be plotted.
    title : str
        The title of the boxplot.

    Returns
    -------
    None.

    """

    fig, axes = plt.subplots(1, 2, figsize=(25, 10))

    fig.suptitle(title, fontweight="bold", fontsize=30, y=0.95)

    i = 0
    # Loop through the columns and make a boxplot of each columns
    for series_name in df.columns:
        ax = axes[i]

        df[[series_name]].boxplot(ax=ax, grid=True, vert=False)

        plt.setp(ax.get_xticklabels(), fontsize=25)
        plt.setp(ax.get_yticklabels(), rotation=90, ha="center", va="center",
                 fontsize=25)

        i += 1

    plt.savefig("boxplot.png", bbox_inches="tight")

    plt.show()

    return


def get_silhoutte_score(scaled_df, num_clusters):
    """
    Computes the silhoutte score for the data based on the required
    number of clusters

    Parameters
    ----------
    scaled_df : DataFrame
        The scaled dataframe.
    num_clusters : int
        The number of clusters to create.

    Returns
    -------
    score : float
        The obtained silhoutte score.

    """
    # Set up the clusterer and set a random seed to ensure replicability
    kmeans = cluster.KMeans(n_clusters=num_clusters, n_init=20,
                            random_state=10)

    # Fit the data, the results are stored in the kmeans object
    kmeans.fit(scaled_df)

    # Calculate the silhoutte score
    score = skmet.silhouette_score(scaled_df, kmeans.labels_)

    return score


def plot_silhoute_scores(scaled_df):
    """
    Makes a line plot of the silhoutte scores for different cluster numbers

    Parameters
    ----------
    scaled_df : DataFrame
        The scaledDataFrame.

    Returns
    -------
    None.

    """
    # Calculate silhouette score for 2 to 10 clusters and plot the scores
    scores = []
    for i in range(2, 10):
        score = get_silhoutte_score(scaled_df, i)
        scores.append([i, score])

    scores_arr = np.array(scores)

    fig, ax = plt.subplots()

    ax.plot(scores_arr[:, 0], scores_arr[:, 1], marker="o")
    ax.set_xlim(2, 9)

    ax.set_xlabel("Number of cluster")
    ax.set_ylabel("Silhoutte Score")
    ax.set_title("Silhoutte Score for Different Number of Clusters")

    plt.show()


def show_clusters(original_df, scaled_df):
    """
    Creates scatter plot showing clusters of countries in the dataset

    Parameters
    ----------
    original_df : DataFrame
        The original dataframe before it was scaled.
    scaled_df : DataFrame
        The scaled dataframe.

    Returns
    -------
    cluster_labels : list of integers
        The cluster labels assigned to each data point.

    """
    # Set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=3, n_init=20, random_state=10)

    # Fit the data
    kmeans.fit(scaled_df)

    # Extract cluster labels
    cluster_labels = kmeans.labels_

    # Extract the estimated cluster centres
    cen = kmeans.cluster_centers_
    # Convert the extracted centres to the original scales
    cen = scaler.inverse_transform(cen)

    xkmeans = cen[:, 0]
    ykmeans = cen[:, 1]

    fig, ax = plt.subplots()

    # Plot the original data showing the kmeans clusters
    original_df.plot.scatter(x=0, y=1, s=10, c=cluster_labels, marker="o",
                             colormap=cm["Paired"], colorbar=False, ax=ax)

    # Show cluster centres
    ax.scatter(xkmeans, ykmeans, 45, "k", marker="d")

    ax.set_title("Cluster of Countries Data")

    plt.savefig("Cluster.png")

    plt.show()

    return cluster_labels


df_countries, df_transposed = read_and_clean_data("data.csv")

# Get the data for all countries in the year 2021 and drop
# countries with missing values
df_2021 = pd.pivot_table(df_countries, values="2021", index="Country Name",
                         columns="Series Name").dropna()

# Check the correlation between the indicators
show_correlation(df_2021.corr().round(2))

selected_columns = ["GNI per capita", "Life expectancy at birth"]
df_selected = df_2021[selected_columns]

# Examine the distribution of the data to enable choosing the right scaler
make_boxplot(
    df_selected, "Distribution of GNI and Life Expectancy in 2019", 2)

# Since the GNI column has some outliers, use RobustScaler which is
# robust to outliers
scaler = RobustScaler()
scaled_arr = scaler.fit_transform(df_selected)
df_scaled = pd.DataFrame(scaled_arr, columns=selected_columns,
                         index=df_2021.index)

# Show a scatterplot of the scaled selected indicators
fig, ax = plt.subplots()
df_scaled.plot.scatter(x=0, y=1, s=10, marker="o", ax=ax)
ax.set_title("Relationship Between GNI and Life Expectancy",
             fontsize=16, fontweight="bold")
plt.show()

# View the silhoutte scores to enable picking the appropraite cluster number
plot_silhoute_scores(df_scaled)

# Plot the clustered data on the original scale
cluster_labels = show_clusters(df_selected, df_scaled)

# Add the cluster labels to the original 2021 data
df_2021["Cluster"] = cluster_labels
