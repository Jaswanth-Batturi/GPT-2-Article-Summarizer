# Cleaning the EXCEL datasheet

import pandas as pd

reviews = pd.read_excel("./data/datasheet.xlsx")

reviews = reviews.drop(['Source link'], axis=1)

# Removing the blank columns
reviews = reviews.dropna()

reviews =  reviews.reset_index(drop=True)

reviews.columns = ['Article', 'Oneliner']

print(reviews.shape)

print(reviews.head())

# Creating a new EXCEL file
reviews.to_excel("./data/oneliner.xlsx", index=False)

print("Process finished...")