import json
import numpy as np
import pandas as pd
import re, nltk, spacy, string
import en_core_web_sm
# nlp = en_core_web_sm.load()
nlp = spacy.load('en_core_web_sm',  disable=["parser", "ner"])
import seaborn as sns
import matplotlib.pyplot as plt

from plotly.offline import plot
import plotly.graph_objects as go
import plotly.express as px

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from nltk.tokenize import word_tokenize
from collections import Counter
import string
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

# import spellchecker
# from spellchecker import SpellChecker

from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from imblearn.over_sampling import SMOTE

# Import pickle to save and load the model
import pickle

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score
# from sklearn.metrics import plot_roc_curve
from sklearn.metrics import RocCurveDisplay
# Importing LogisticRegression from sklearn
from sklearn.linear_model import LogisticRegression

# Importing Random Forest Classifier from sklearn
from sklearn.ensemble import RandomForestClassifier

# importing libraries for XGBoost classifier
import xgboost as xgb
from xgboost import XGBClassifier

from sklearn.metrics.pairwise import pairwise_distances

from pprint import pprint


# Setting max rows and columns
pd.options.display.max_columns = 50
pd.options.display.max_rows =  50
pd.options.display.max_colwidth= 300
pd.set_option("display.precision", 2)

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Loading the data
df = pd.read_csv("sample30.csv")

# Exploratory Data Analysis
df.shape

# Inspect the dataframe to understand the given data.
df.info()

# Print top 5 rows of the data
df.head()

# Getting total number of NULL values and percentage of the columns
null_count = df[df.columns[df.isna().any()]].isna().sum().sort_values(ascending=False)
null_perc = (df[df.columns[df.isna().any()]].isna().sum() * 100 / df.shape[0]).sort_values(ascending=False)
null_data = pd.concat([null_count, null_perc], axis=1, keys=['Count', 'Percentage'])
null_data

df_clean = df.copy()
df_clean.drop(columns=['reviews_userProvince','reviews_userCity','reviews_didPurchase'], inplace=True)
df_clean.shape

# Checking NULLs again
null_count = df_clean[df_clean.columns[df_clean.isna().any()]].isna().sum().sort_values(ascending=False)
null_perc = (df_clean[df_clean.columns[df_clean.isna().any()]].isna().sum() * 100 / df_clean.shape[0]).sort_values(ascending=False)
null_data = pd.concat([null_count, null_perc], axis=1, keys=['Count', 'Percentage'])
null_data


df_clean.reviews_doRecommend.value_counts(normalize=True)
sns.countplot(x = 'reviews_doRecommend', data = df_clean)
plt.show()

df_clean.drop(columns=['reviews_doRecommend'], inplace=True)

# Checking rowcount before deletion
df_clean.shape

df_clean = df_clean[~ df_clean.reviews_title.isna() ]

# Checking rowcount post deletion
df_clean.shape

df_clean = df_clean[~ df_clean.reviews_title.isna() ]

# Checking rowcount post deletion
df_clean.shape

df_clean = df_clean[~ df_clean.reviews_username.isna() ]

# Checking rowcount post deletion
df_clean.shape

df_clean.user_sentiment.value_counts()

df_clean[ df_clean.user_sentiment.isna() ]

df_clean[ df_clean.user_sentiment.isna() ].user_sentiment

df_clean.user_sentiment.fillna('Positive', inplace=True)

# Checking NULLs again
null_count = df_clean[df_clean.columns[df_clean.isna().any()]].isna().sum().sort_values(ascending=False)
null_perc = (df_clean[df_clean.columns[df_clean.isna().any()]].isna().sum() * 100 / df_clean.shape[0]).sort_values(ascending=False)
null_data = pd.concat([null_count, null_perc], axis=1, keys=['Count', 'Percentage'])
print(null_data)

# Check the data for top 5 rows
print(df_clean.head())

### Checking Distribution of `reviews_rating` column
sns.countplot(x = 'reviews_rating', data = df_clean).set(title="Distribution of reviews rating by count", xlabel="reviews rating", ylabel="reviews count")
plt.show()

### Checking Top 5 Brands with negative reviews
df_clean[ df_clean.user_sentiment == 'Negative' ].brand.value_counts(normalize=True, ascending=False).head(5).plot(kind='bar')
plt.title("Top 5 Brands with negative reviews")
plt.xlabel("Brands")
plt.ylabel("Percentage of negative reviews")
plt.show()

### Checking Top 5 Brands with positive reviews
df_clean[ df_clean.user_sentiment == 'Positive' ].brand.value_counts(normalize=True, ascending=False).head(5).plot(kind='bar')
plt.title("Top 5 Brands with positive reviews")
plt.xlabel("Brands")
plt.ylabel("Percentage of positive reviews")
plt.show()

print(df_clean.brand.value_counts(normalize=True).head(5))

### Checking review counts based on the review year

# Before type conversion
print(df_clean.reviews_date.dtype)

df_clean['reviews_date'] = pd.to_datetime(df_clean['reviews_date'], errors='coerce')

# After type conversion
print(df_clean.reviews_date.dtype)

# Getting year component from date
print(df_clean.reviews_date.dt.year)

# Creating a distribution plot based on reviews year
sns.displot(data=df_clean, x=df_clean.reviews_date.dt.year).set(title="Distribution of reviews by year", xlabel="reviews year", ylabel="reviews count")
plt.show()

### Checking Movie categories by Rating

plt.figure(figsize = [10,6])

sns.boxplot(data=df_clean, x='user_sentiment', y='reviews_rating', color='green')
plt.xticks(rotation = 45)

plt.tight_layout(pad = 4)
plt.show()

df_clean[ (df_clean.user_sentiment == 'Negative') & (df_clean.reviews_rating >= 4) ].groupby(['reviews_rating']).count().user_sentiment

## Type Conversion
# Convert all the text columns to string for performing text operations
df_clean['brand'] = df_clean['brand'].astype(str)
df_clean['categories'] = df_clean['categories'].astype(str)
df_clean['manufacturer'] = df_clean['manufacturer'].astype(str)
df_clean['name'] = df_clean['name'].astype(str)
df_clean['reviews_text'] = df_clean['reviews_text'].astype(str)
df_clean['reviews_title'] = df_clean['reviews_title'].astype(str)
df_clean['reviews_username'] = df_clean['reviews_username'].astype(str)

# Getting a copy of dataframe for pre-processing
df_prep = df_clean.copy()

df_prep['reviews_combined'] = df_prep['reviews_text'] + " " + df_prep['reviews_title']
df_prep.drop(columns=['reviews_text', 'reviews_title'], inplace=True)

print(df_prep.shape)

print(df_prep.head(1))

## Removing columns unneeded for analysis
df_prep.drop(columns=['categories', 'manufacturer', 'reviews_date'], inplace=True)

print(df_prep.head(1))

## Creating dataframe for Sentiment analysis with only the required columns
df_sent = df_prep[['id','name','reviews_combined', 'user_sentiment']]
print(df_sent.shape)
print(df_sent.head(2))