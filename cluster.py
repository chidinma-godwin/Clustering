#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 17:56:06 2024

@author: Chidex
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
from matplotlib.lines import Line2D

import sklearn.preprocessing as pp
import sklearn.cluster as cluster
import sklearn.metrics as skmet

from scipy.optimize import curve_fit

import errors as err


def read_and_clean_data(filename):
    """
    Reads a data file into a dataframe, cleans and transposes the dataframe.
    Missing data will not be dropped in this function to reduce data loss.

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

    # Ensure the values are numeric
    df[df.columns] = df[df.columns].apply(pd.to_numeric, errors="coerce")

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
    ax.set_title("Silhoutte Score for Different Number of Clusters",
                 fontweight="bold")

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

    # Change the cluster labels to 1, 2, 3 instead of 0, 1, 2
    adjusted_labels = [label+1 for label in cluster_labels]
    
    # Plot the original data showing the kmeans clusters
    scatter = ax.scatter(original_df.iloc[:, 0], original_df.iloc[:, 1], s=10,
                         c=adjusted_labels, marker="o", cmap=cm["Set1"])

    # Show cluster centres
    ax.scatter(xkmeans, ykmeans, 45, "k", marker="d")

    ax.legend(*scatter.legend_elements(), title='Clusters')

    ax.set_title("Country Clusters Using GNI and Life Expectancy in 2019",
                 fontweight="bold")
    ax.set_xlabel("GNI per Capita, 2019")
    ax.set_ylabel("Life Expectancy at Birth (Years)")

    plt.savefig("Cluster.png")

    plt.show()

    return adjusted_labels


def get_clusters_frequency(df_2021):
    """
    Shows a bar chart displaying the number of countries in each cluster

    Parameters
    ----------
    df_2021 : DataFrame
        The dataframe containing the countries and the cluster they belong to.

    Returns
    -------
    None.

    """

    # Group the data by the clusters and rename the columns
    df_clusters = df_2021.groupby("Cluster").size().reset_index()
    df_clusters.columns = ["Cluster", "Number of Countries"]

    fig, ax = plt.subplots()

    ax.set_ylabel("Number of Countries")

    df_clusters.plot.bar(x="Cluster", y="Number of Countries", legend=False,
                         ax=ax, color=["tab:red", "tab:green", "tab:orange"])

    ax.set_title("Number of Countries in Each Cluster", fontweight="bold")

    # Add text at the middle of the bars showing the number of countries
    # in each cluster
    for p in ax.patches:
        ax.annotate(p.get_height(),
                    xy=(p.get_x() + p.get_width()/2, p.get_height() / 2),
                    ha='center', va='center', fontsize=14, color="white",
                    fontweight="bold")

    plt.savefig("cluster_freq.png", bbox_inches="tight")

    return


def compare_clusters(df_2021):
    """
    Plots a multiple bar chart to compare the clusters across the different
    indicators. The indicators values are scaled to enhance visualising the
    relative differences irrespective of the unit or scale of the indicators

    Parameters
    ----------
    df_2021 : DataFrame
        The dataframe containing countries data for 2021 and the
        cluster they belong to.

    Returns
    -------
    None.

    """

    df_to_scale = df_2021.loc[:, df_2021.columns != 'Cluster']

    # Scale the data uniformly using QuantileTransformer to enable comparison
    # irrespective of the indicators scale and also reduce the impact
    # of outliers when computing the mean
    scaled_arr = pp.QuantileTransformer(
        n_quantiles=df_to_scale.shape[0], random_state=10) \
        .fit_transform(df_to_scale)
    scaled_df = pd.DataFrame(scaled_arr, columns=df_to_scale.columns,
                             index=df_to_scale.index)

    # Add the cluster column back to the scaled dataframe
    scaled_df["Cluster"] = df_2021["Cluster"]

    # Compute the mean of each indicators for each cluster
    df_agg = scaled_df.groupby(by="Cluster").mean().T

    fig, ax = plt.subplots()

    df_agg.plot.bar(ax=ax, color=["tab:red", "tab:green", "tab:orange"])

    ax.set_title("Comparison of Clusters Across Different Indicators",
                 fontweight="bold")
    ax.set_ylabel("Scaled Values")

    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=30,
             ha="right", rotation_mode="anchor")

    plt.savefig("cluster_compare.png", bbox_inches="tight")

    plt.show()

    return


def show_clusters_top_countries(df_2021):
    """
    Displays the top 5 countries with the highest GNI per capita in each
    cluster in an horizontal bar chart

    Parameters
    ----------
    df_2021 : DataFrame
        The dataframe containing countries data for 2021 and the
        cluster they belong to.

    Returns
    -------
    None.

    """

    # Sort the countries in each cluster by their GNI per capita
    df_cluster = df_2021.pivot_table(values="GNI per capita",
                                     index=["Cluster", "Country Name"])\
        .sort_values(by="GNI per capita")

    # Extract the countries in each cluster
    cluster1 = df_cluster.loc[1]
    cluster2 = df_cluster.loc[2]
    cluster3 = df_cluster.loc[3]

    # Create empty dataframes to use for adding space between each cluster
    # on the graph. Two dataframes with different index are created to avoid
    # them overriding each other
    df_empty = pd.DataFrame(
        {"GNI per capita": [0]}, index=[""])
    df_empty2 = pd.DataFrame(
        {"GNI per capita, PPP (current international $)": [0]}, index=["  "])

    # Select 5 countries with the highest GNI per capita in each cluster.
    # Concatenate the selected countries in the order of clusters with lower
    # GNI to clusters with hegher GNI and add the empty dataframes between
    # the clusters.
    selected_countries = pd.concat(
        [cluster1[-5:], df_empty, cluster3[-5:], df_empty2, cluster2[-5:]])

    fig, ax = plt.subplots(layout="constrained")

    # Set different colors for bars of countries in different clusters
    color = (["tab:red"]*6)+(["tab:orange"]*6)+(["tab:green"]*6)
    ax.barh(y=selected_countries.index,
            width=selected_countries["GNI per capita"], color=color)

    ax.set_title("Five Countries with Highest GNI per capita in Each Cluster",
                 fontweight="bold", y=1.03)
    ax.set_xlabel("GNI per capital")

    # Create a matching legend for the clusters
    cluster1L = Line2D([], [], color="tab:red", label="Cluster 1", lw=6)
    cluster2L = Line2D([], [], color="tab:green", label="Cluster 2", lw=6)
    cluster3L = Line2D([], [], color="tab:orange", label="Cluster 3", lw=6)
    ax.legend(handles=[cluster1L, cluster2L, cluster3L], borderpad=0.7,
              loc="lower right", fontsize=12)

    plt.savefig("clusters_sample_barplot.png", bbox_inches="tight")

    plt.show()

    return


def logistic(t, n0, g, t0):
    """Calculates logistic function with scale factor n0 and growth rate g"""

    func = n0 / (1 + np.exp(-g*(t - t0)))

    return func


def show_fitted_model(df_gni_cluster2, title, forecast=False):
    """
    Create a line plot showing the original and fitted data

    Parameters
    ----------
    df_gni_cluster2 : DataFrame
        DataFrame containing the GNI per capita for the selected countries
        in the second cluster.

    title: str
        The plot title

    forecast: bool, default False
        If true, the plot shows the predicted values for 8 extra years.

    Returns
    -------
    None.

    """

    fig, ax = plt.subplots()

    colors = plt.cm.tab20c([0, 4])

    for i, column in enumerate(df_gni_cluster2.columns[1:]):
        xdata = df_gni_cluster2["Year"]
        ydata = df_gni_cluster2[column]

        # Fit a logistic function to the data and pass the initial guess
        # for the parameters
        params, covariance = curve_fit(logistic, xdata, ydata,
                                       p0=(45000, 0.05, 2010))

        # Define the x values to use for prediction so that it can be
        # overwritten when specifying a different x values for forecasting
        pred_x = xdata.copy()
        label = f"{column} (Fitted GNI)"
        if forecast:
            pred_x = np.linspace(xdata.min(), xdata.max()+8, 100)
            label = f"{column} (GNI Forecast)"

        # Pass the obtained optimal parameters values to the logistic function
        # to get the fitted values
        yfit = logistic(pred_x, *params)

        # Plot the original and fitted GNI per capita with the same line
        # color but different line styles
        ax.plot(xdata, ydata, label=f"{column} (Original GNI)",
                color=colors[i], lw=2)
        ax.plot(pred_x, yfit, "--", label=label,
                color=colors[i], lw=2)

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Year")
        ax.set_ylabel("GNI per capita")
        ax.set_xlim(pred_x.min(), pred_x.max())
        ax.legend(frameon=False)

        # Compute the error range caused by the uncertainty of the fit
        # and show it in the plot
        sigma = err.error_prop(pred_x, logistic, params, covariance)
        upper_limit = yfit + sigma
        lower_limit = yfit - sigma
        ax.fill_between(pred_x, lower_limit, upper_limit,
                        color="yellow", alpha=0.6)

    plot_name = "gni_forecast.png" if forecast else "gni_fit.png"

    plt.savefig(plot_name)

    plt.show()

    return


# Read the data into pandas dataframes and clean the dataframe
df_countries, df_transposed = read_and_clean_data("data.csv")

# Get the data for all countries in the year 2021 and drop countries with
# with missing values. Missing values are dropped at this stage to avoid
# loosing too much data
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
scaler = pp.RobustScaler()
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

# Display a bar graph showing the number of countries in each cluster
get_clusters_frequency(df_2021)

# Compare the clusters across the different indicators
compare_clusters(df_2021)

# Show the top 5 countries with the highest GNI per capita in each cluster
show_clusters_top_countries(df_2021)

# Show line plot of the original and fitted data for 2 randomly selected
# countries from the second cluster.
df_gni = df_countries.xs("GNI per capita", level="Series Name")
# Add a new "Cluster" column with matching cluster label for each country
# and drop countries with missing data before selecting the sample
df_gni = df_gni.join(df_2021[["Cluster"]]).dropna()
# Select only the data for the second cluster
df_gni_cluster2 = df_gni.pivot_table(
    index=["Cluster", "Country Name"]).xs(2, level="Cluster")
# Randomly select 2 countries from the cluster
df_gni_cluster2_sample = df_gni_cluster2.sample(n=2, random_state=42).T
df_gni_cluster2_sample.index.name = "Year"
df_gni_cluster2_sample.reset_index(inplace=True)
df_gni_cluster2_sample["Year"] = df_gni_cluster2_sample["Year"].astype(int)
# Show line plot of the original and fitted data for the 2 selected countries
show_fitted_model(df_gni_cluster2_sample,
                  "Comparison of Fitted and Actual Data Trends")

# Show a line plot of the original and forecasted GNI per capital for UK
df_gni_uk = df_gni_cluster2_sample[["Year", "United Kingdom"]]
show_fitted_model(df_gni_uk, "Forecasted GNI per Capita for United Kingdom",
                  forecast=True)
