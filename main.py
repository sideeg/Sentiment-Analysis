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

## Handling punctuations

# Function to clean the text and remove all the unnecessary elements.
def clean_punctuation(sent):
    sent = sent.lower() # Text to lowercase
    pattern = '[^\w\s]' # Removing punctuation
    sent = re.sub(pattern, '', sent)
    return sent

df_sent['reviews_cleaned'] = df_sent['reviews_combined'].apply(clean_punctuation)

df_sent.head(2)


nltk.download('stopwords')

from nltk.corpus import stopwords
stopwords_set = set(stopwords.words("english"))
print(stopwords_set)

# Function to remove the stopwords
def clean_stopwords(sent):
    # import nltk
    # nltk.download('punkt_tab')

    sent = sent.lower() # Text to lowercase
    words = word_tokenize(sent) # Split sentences into words
    text_nostopwords = " ".join( [each_word for each_word in words if each_word not in stopwords_set] )
    return sent

df_sent['reviews_cleaned'] = df_sent['reviews_cleaned'].apply(clean_stopwords)

print(df_sent.head(2))

## Handling lemmatization

#Function to lemmatize the text
def clean_lemma(text):
    sent = []
    doc = nlp(text)
    for token in doc:
        sent.append(token.lemma_)
    return " ".join(sent)

df_sent['reviews_lemmatized'] = df_sent['reviews_cleaned'].apply(clean_lemma)

print(df_sent.head(2))

df_sent = df_sent[['id','name','reviews_lemmatized', 'user_sentiment']]

print(df_sent.head(2))
print(df_sent.shape)

# Visualizing 'reviews_lemmatized' character length
character_length = [len(each_sent) for each_sent in df_sent['reviews_lemmatized']]

sns.displot(character_length, kind='hist', bins=60)
plt.xlabel("Reviews character length")
plt.ylabel("Total number of Reviews")
plt.title("Distribution of Reviews character length")
plt.show()

#Using a word cloud visualize the top 30 words in review by frequency
stopwords_wordcloud = set(STOPWORDS)
wordcloud = WordCloud(max_font_size=60, max_words=30,
                      background_color="white", random_state=42,
                      stopwords=stopwords_wordcloud).generate(str(df_sent['reviews_lemmatized']))
plt.figure(figsize=[10,10])
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

#Using a word cloud visualize the top 30 words in review by frequency
stopwords_wordcloud = set(STOPWORDS)
wordcloud = WordCloud(max_font_size=60, max_words=30,
                      background_color="white", random_state=42,
                      stopwords=stopwords_wordcloud).generate(str(df_sent['reviews_lemmatized']))
plt.figure(figsize=[10,10])
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# Feature Extraction

# Initialize the TfidfVectorizer
tfidf = TfidfVectorizer(min_df=5, max_df=0.95, stop_words='english', ngram_range=(1,2))

X = tfidf.fit_transform(df_sent['reviews_lemmatized'])

y= df_sent['user_sentiment']

## Train, test split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.25)

print("Train shapes:", X_train.shape, y_train.shape)
print("Test shapes:", X_test.shape, y_test.shape)

## Class Imbalance

# Check the data to see if there is a class imbalance in the data
df_sent['user_sentiment'].value_counts(normalize=True)

print(df_sent.head(2))

counter = Counter(y_train)
print("Before handling imbalance", counter)

#oversampling using SMOTE
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train,y_train)

counter = Counter(y_train_sm)
print("After handling imbalance", counter)

# Model Building

# Function to display scores
def evaluation_scores(classifier, X_test, y_test):
    # Calculating Predicted value
    y_pred = classifier.predict(X_test)

    # Create confusion matrix
    conf_m = confusion_matrix(y_test, y_pred)

    print("Visualizing the Confusion Matrix with a Heatmap")
    print("\n")
    print("*" * 50)
    # Visualize Confusion Matrix with heatmap
    fig, ax = plt.subplots(figsize=(6, 6))
    ax = sns.heatmap(confusion_matrix(y_test, y_pred),
                     annot=True,
                     cbar=False,
                     cmap="RdYlGn", fmt='0.1f')
    plt.xlabel("Actual label")
    plt.ylabel("Predicted label")
    plt.show()
    print("*" * 50)
    print("\n")

    # Calculating the values of True Positives, True Negatives, False Positivies and False Negatives
    TP = conf_m[1][1]
    TN = conf_m[0][0]
    FP = conf_m[0][1]
    FN = conf_m[1][0]

    print("Values of True Positives, True Negatives, False Positivies and False Negatives")
    print("~" * 50)
    print('True Positives:', TP)
    print('True Negatives:', TN)
    print('False Positives:', FP)
    print('False Negatives:', FN)
    print("~" * 50)
    print("\n")

    # Calculating Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracy = round(accuracy, 2)

    # Calculating Sensitivity/Recall
    sensitivity_recall = (TP / float(TP + FN))
    sensitivity_recall = round(sensitivity_recall, 2)

    # Calculating Specificity
    specificity = (TN / float(TN + FP))
    specificity = round(specificity, 2)

    # Calculating Precision
    precision = (TP / float(TP + FP))
    precision = round(precision, 2)

    # Calculating F_1 score
    F1_score = 2 * ((precision * sensitivity_recall) / (precision + sensitivity_recall))
    F1_score = round(F1_score, 2)

    print("Evaluation Score Summary")
    print('-' * 50)
    print(f'Accuracy Score: {round(accuracy, 2)}')
    print(f'Sensitivity/Recall Score: {round(sensitivity_recall, 2)}')
    print(f'Specificity Score: {round(specificity, 2)}')
    print(f'Precision: {round(precision, 2)}')
    print(f'F1 Score: {round(F1_score, 2)}')
    print('-' * 50)

    # Returning evaluation results for comparison later
    evaluation_metrics = []
    evaluation_metrics.append(accuracy)
    evaluation_metrics.append(sensitivity_recall)
    evaluation_metrics.append(specificity)
    evaluation_metrics.append(precision)
    evaluation_metrics.append(F1_score)

    return evaluation_metrics

## Logistic Regression

### Base Model
logreg = LogisticRegression(random_state=42, solver='liblinear').fit(X_train_sm, y_train_sm)
# Getting the score of the base model
lr_metrics = evaluation_scores(logreg, X_test, y_test)

# Printing the scores of the base model as reference
df_lrb_metrics = pd.DataFrame({'Metrics': ['Accuracy','Sensitivity/Recall','Specificity','Precision','F1 Score'], 'Logistic Regression Base Model': lr_metrics},
                             columns = ['Metrics', 'Logistic Regression Base Model']
                             )
print(df_lrb_metrics)

### HyperParameter Tuning

logreg_grid = {"C": [100, 10, 5, 4, 3, 2, 1, 1.0, 0.1, 0.01],
                "solver": ["liblinear"]}

# Setup grid hyperparameter search for LogisticRegression
logreg_hpt = GridSearchCV(LogisticRegression(random_state=42),
                                param_grid=logreg_grid,
                                cv=5,
                                verbose=True,
                                n_jobs=-1,
                                scoring='f1')

# Fit random hyperparameter search model
logreg_hpt.fit(X_train_sm, y_train_sm);

# Checking the best parameters
print(logreg_hpt.best_params_)

### HyperParameter Tuned Model

# Getting the scores of the tuned model
lr_tuned_metrics = evaluation_scores(logreg_hpt, X_test, y_test)

# Printing the scores of the base and tuned Logistic Regression model for reference
dict_lr_bt_metrics = {'Metrics': ['Accuracy','Sensitivity/Recall','Specificity','Precision','F1 Score'],
                               'LR Base Model': lr_metrics,
                               'LR Tuned Model': lr_tuned_metrics}

df_lr_bt_metrics = pd.DataFrame(dict_lr_bt_metrics, columns = ['Metrics', 'LR Base Model', 'LR Tuned Model'])
print(df_lr_bt_metrics)

## Random Forest Classifier

rf = RandomForestClassifier(random_state=42).fit(X_train_sm, y_train_sm)

# Getting the score of the base model
rf_metrics = evaluation_scores(rf, X_test, y_test)

# Printing the scores of the base model as reference
df_rfb_metrics = pd.DataFrame({'Metrics': ['Accuracy','Sensitivity/Recall','Specificity','Precision','F1 Score'], 'RF Base Model': rf_metrics},
                             columns = ['Metrics', 'RF Base Model']
                             )
print(df_rfb_metrics)

### HyperParameter Tuning

rf_grid = {"n_estimators": np.arange(10, 1000, 50),
           "max_depth": np.arange(10, 50, 5),
           "min_samples_split": np.arange(15, 500, 15),
           "min_samples_leaf": np.arange(5, 50, 5)}

# Setup random hyperparameter search for Random Forest Classifier
rf_hpt = RandomizedSearchCV(RandomForestClassifier(random_state=42),
                                param_distributions=rf_grid,
                                cv=5,
                                verbose=True,
                                n_jobs=-1,
                                scoring='f1')

# Fit random hyperparameter search model
rf_hpt.fit(X_train_sm, y_train_sm);

# Check best parameters
print(rf_hpt.best_params_)

evaluation_scores(rf_hpt, X_test, y_test)

# Fine tuning using Grid Search CV
rf_grid = {"n_estimators": [510],
           "max_depth": [20],
           "min_samples_split": [350, 400],
           "min_samples_leaf": [45, 50]}

# Setup random hyperparameter search for Random Forest Classifier
rf_hpt = GridSearchCV(RandomForestClassifier(random_state=42),
                                param_grid=rf_grid,
                                cv=5,
                                verbose=True,
                                n_jobs=-1,
                                scoring='f1')

# Fit random hyperparameter search model
rf_hpt.fit(X_train_sm, y_train_sm);

# Getting the scores of the tuned model
rf_tuned_metrics = evaluation_scores(rf_hpt, X_test, y_test)

# Printing the scores of the base and tuned Random Forest model as reference
dict_rf_bt_metrics = {'Metrics': ['Accuracy','Sensitivity/Recall','Specificity','Precision','F1 Score'],
                               'RF Base Model': rf_metrics,
                               'RF Tuned Model': rf_tuned_metrics}

df_rf_bt_metrics = pd.DataFrame(dict_rf_bt_metrics, columns = ['Metrics', 'RF Base Model', 'RF Tuned Model'])
print(df_rf_bt_metrics)

## XGBoost Classifier

xg = xgb.XGBClassifier(random_state=42).fit(X_train_sm, y_train_sm)
# Getting the score of the base model
xg_metrics = evaluation_scores(xg, X_test, y_test)

# Printing the scores of the base model as reference
df_xgb_metrics = pd.DataFrame({'Metrics': ['Accuracy','Sensitivity/Recall','Specificity','Precision','F1 Score'], 'XG Base Model': xg_metrics},
                             columns = ['Metrics', 'XG Base Model']
                             )
print(df_xgb_metrics)

xg_grid = {"learning_rate": np.arange(0.05, 1, 0.1),
           "max_depth": np.arange(5, 20, 5)
           }

# Setup random hyperparameter search for Random Forest Classifier
xg_hpt = RandomizedSearchCV(XGBClassifier(random_state=42),
                                param_distributions=xg_grid,
                                cv=4,
                                verbose=True,
                                n_jobs=-1,
                                scoring='f1')

# Fit random hyperparameter search model
xg_hpt.fit(X_train_sm, y_train_sm);

# Check best parameters
print(xg_hpt.best_params_)

evaluation_scores(xg_hpt, X_test, y_test)

# Fine tuning with Grid Search CV
xg_grid = {"learning_rate": [0.45, 0.15],
           "max_depth": [5, 10]
           }

# Setup random hyperparameter search for Random Forest Classifier
xg_hpt = GridSearchCV(XGBClassifier(random_state=42),
                                param_grid=xg_grid,
                                cv=5,
                                verbose=True,
                                n_jobs=-1,
                                scoring='f1')

# Fit random hyperparameter search model
xg_hpt.fit(X_train_sm, y_train_sm);

# Check best parameters
print(xg_hpt.best_params_)

# Getting the scores of the tuned model
xg_tuned_metrics = evaluation_scores(xg_hpt, X_test, y_test)

# Printing the scores of the base and tuned XGBoost model as reference
dict_xg_bt_metrics = {'Metrics': ['Accuracy','Sensitivity/Recall','Specificity','Precision','F1 Score'],
                               'XG Base Model': xg_metrics,
                               'XG Tuned Model': xg_tuned_metrics}

df_xg_bt_metrics = pd.DataFrame(dict_xg_bt_metrics, columns = ['Metrics', 'XG Base Model', 'XG Tuned Model'])
print(df_xg_bt_metrics)