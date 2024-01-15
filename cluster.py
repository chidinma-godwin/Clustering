#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 17:56:06 2024

@author: Chidex
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


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

    # Unstack the dataframe so that "Country Name" is the only column, drop
    # drop countries with missing data and stack the dataframe back to its
    # original shape
    cleaned_df = df.unstack("Series Name").dropna().stack()

    cleaned_df_transposed = cleaned_df.transpose()

    return cleaned_df, cleaned_df_transposed


def show_correlation(corr):
    """
    Plot heatmap to show the correlation between indicators

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


def make_boxplot(df, title, n_rows, n_cols):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(25, 15))
    
    fig.suptitle(title, fontweight="bold", fontsize=30, y=0.95)

    i = 0
    j = 0
    for series_name in df.columns:
        ax = axes[i, j]

        df[[series_name]].boxplot(ax=ax, grid=True, vert=False)

        plt.setp(ax.get_xticklabels(), fontsize=20)
        plt.setp(ax.get_yticklabels(), rotation=90, ha="center", va="center",
                 fontsize=25)

        # Set the axes to make the next plot on
        if j < (n_cols - 1):
            j += 1
        else:
            i += 1
            j = 0
            
    plt.savefig("boxplot.png", bbox_inches="tight")
    
    plt.show()

    return


def show_clusters():
    return


df_countries, df_transposed = read_and_clean_data("data.csv")

# Get the data for all countries in the most recent year (2021)
df_2021 = pd.pivot_table(df_countries, values="2021", index="Country Name",
                         columns="Series Name")

# Check the correlation between the indicators
show_correlation(df_2021.corr().round(2))

# Examine the distribution of the data to enable the right scaler
make_boxplot(df_2021, "Distribution of Indicators in 2019", 2, 3)

