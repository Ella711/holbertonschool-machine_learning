#!/usr/bin/env python3
"""
0. When to Invest
"""
import pandas as pd
import matplotlib.pyplot as plt


def preprocess():
    """preprocess data"""
    coinbase = 'coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv'
    coinbasedf = pd.read_csv(coinbase)

    # get a feel of the data
    print("Coinbase Describe:")
    print(coinbasedf.describe().transpose())
    print("TOP 5 ROWS AND SHAPE")
    print(coinbasedf.head())
    print(coinbasedf.shape)
    print("********************")
    print('')

    # convert Timestamp to Datetime
    print("UPDATE TIMESTAMP FROM UNIX TO DATETIME")
    coinbasedf["Timestamp"] = pd.to_datetime(coinbasedf["Timestamp"], unit='s')
    print(coinbasedf.head())
    print("")

    # view amount of NAN per column
    print("TOTAL NAN VALUES PER COLUMN:")
    print(coinbasedf.isna().sum())
    print('')

    # Forward fill all missing NAN values
    print("FORWARD FILL NAN VALUES:")
    df = coinbasedf.ffill()
    print(df.head())
    print("********************")

    # convert minutes to hours
    print("CONVERT MINUTES TO HOURS")
    data = df.copy()
    data = data[8::60]
    print(data.head())
    print("********************")

    # show correlation to know what cols to drop
    print("VIEW CORRELATION MATRIX TO REMOVE COLUMNS HIGHLY CORRELATED")
    # View correlation matrix
    correlation = data.corr(method='pearson', numeric_only=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(correlation, cmap='Blues')
    fig.colorbar(im, orientation='vertical', fraction=0.5)
    ax.set_xticklabels(data.columns, rotation=65, fontsize=15)
    ax.set_yticklabels(data.columns, rotation=0, fontsize=15)

    for i in range(len(data.columns) - 1):
        for j in range(len(data.columns) - 1):
            text = ax.text(j, i, round(correlation.to_numpy()[i, j], 2), ha="center", va="center", color="black")
    plt.show()
    print("********************")

    # Droping Highly correlated feature cols
    print("DROPPING HIGHLY CORRELATED FEATURE COLUMNS AND CHANGING INDEX TO TIME")
    # MAKES DATA UNIVARIATE
    data = data[['Timestamp', 'Weighted_Price']].reset_index()
    data.drop(data.columns[[0]], axis=1, inplace=True)
    print(data.head())
    print("********************")

    # Show Weighted Prices by year
    plt.figure(figsize=(12, 8))
    plt.plot(data["Weighted_Price"])
    plt.ylabel("Weighted_Price")
    plt.xlabel("Year")
    plt.title("Weighted_Price")
    plt.show()

    return data


def normalize(data):
    """normalize data"""
    train_mean = data.mean()
    train_std = data.std()
    data = (data - train_mean) / train_std

    return data


if __name__ == "__main__":
    data = preprocess()
    norm_data = normalize(data)

    # Save as new CSV
    data.to_csv('cleancoinbase.csv')
