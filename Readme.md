
# Sentiment Analysis of Reviews using Machine Learning

This project aims to perform sentiment analysis on online product reviews using machine learning techniques. The dataset contains product reviews, and the goal is to classify the sentiment into positive or negative based on the review text.

## Requirements

Before running the project, ensure you have the following libraries installed:

- `pandas` for data analysis.
- `numpy` for numerical operations.
- `nltk` and `spacy` for text processing.
- `sklearn` for building and training models.
- `xgboost` for applying the XGBoost model.
- `wordcloud` for visualizing the most frequent words in reviews.
- `imblearn` for handling class imbalance using SMOTE.

You can install the required libraries using `pip`:
```bash
pip install -r requirements.txt
```

### The `requirements.txt` file contains:
```
pandas
numpy
nltk
spacy
scikit-learn
xgboost
wordcloud
imblearn
matplotlib
seaborn
plotly
```

### Download the `spaCy` model:
You will also need to download the `spaCy` model:
```bash
python -m spacy download en_core_web_sm
```

## How It Works

### 1. Loading the Data
The data is loaded from a CSV file containing product reviews. Make sure the file is located in the same directory as the script, or modify the path in the code if the file is elsewhere.

```python
df = pd.read_csv("sample30.csv")
```

### 2. Data Preprocessing
In this step, the data is cleaned and text is processed through several stages:
- **Handling Missing Values** (NaN).
- **Combining Review Text and Title** into a single column called `reviews_combined`.
- **Text Cleaning** by removing punctuation and stopwords.
- **Lemmatization** to reduce words to their base form.

### 3. Feature Extraction
Text data is converted into numerical vectors using the `TfidfVectorizer`, which transforms the words into numerical representations based on their frequency in the text.

```python
tfidf = TfidfVectorizer(min_df=5, max_df=0.95, stop_words='english', ngram_range=(1,2))
X = tfidf.fit_transform(df_sent['reviews_lemmatized'])
```

### 4. Handling Class Imbalance
**SMOTE** (Synthetic Minority Over-sampling Technique) is used to handle class imbalance between the "Positive" and "Negative" classes.

```python
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
```

### 5. Model Building
Three models are built for sentiment classification:
- **Logistic Regression**
- **Random Forest**
- **XGBoost**

The models are trained using the processed data, and **confusion matrix** and **classification report** are used to evaluate their performance.

```python
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_sm, y_train_sm)
evaluate_model(log_reg, X_test, y_test)
```

### 6. Evaluation
Models are evaluated using:
- **Accuracy**.
- **Classification Report**.
- **Confusion Matrix**.

### 7. Visualization
**WordCloud** is used to visualize the most frequent words in the reviews.

```python
wordcloud = WordCloud(max_font_size=60, max_words=30, background_color="white").generate(" ".join(df_sent['reviews_lemmatized']))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
```

## Running the Project

To run the project, follow these steps:

1. Ensure you have installed all the required libraries by running `pip install -r requirements.txt`.
2. Run the script:
   ```bash
   python sentiment_analysis.py
   ```

### Outputs:
- **Data Analysis**: Includes distribution of ratings and reviews.
- **Sentiment Classification**: Sentiments are classified as Positive or Negative.
- **Visualization**: **WordCloud** of the most frequent words in the reviews.
- **Confusion Matrix**: Used to evaluate model performance.

## Notes
- **Handling Missing Values**: Ensure your dataset does not contain null values in crucial columns such as `reviews_title` and `reviews_username` before starting the analysis.
- **Libraries and Environment**: If you're working in a virtual environment (like `virtualenv` or `conda`), make sure you've activated the correct environment before installing the packages.

## Contributions
If you have any suggestions for improvements, feel free to open an **Issue** or submit a **Pull Request**.

## License
This project is licensed under the [MIT License](LICENSE).

## Contact Information

You can reach me at:

- **Phone**: +256766699449
- **LinkedIn**: [Sideeg Mohammed on LinkedIn](https://www.linkedin.com/in/sideeg-mohammed-6ba443185/)


