import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import re
import string

# Load datasets (use a subset for testing)
data_fake = pd.read_csv('Fake.csv').head(1000)  # Use the first 1000 rows for testing
data_true = pd.read_csv('True.csv').head(1000)  # Use the first 1000 rows for testing

# Add labels
data_fake["class"] = 0  # Fake news
data_true["class"] = 1  # Real news

# Create manual testing data (last 10 rows for testing)
data_fake_manual_testing = data_fake.tail(10)
data_true_manual_testing = data_true.tail(10)

# Remove manual testing rows from the main datasets
data_fake.drop(data_fake.index[-10:], inplace=True)
data_true.drop(data_true.index[-10:], inplace=True)

# Add labels to the testing data using .loc[] to avoid SettingWithCopyWarning
data_fake_manual_testing.loc[:, 'class'] = 0
data_true_manual_testing.loc[:, 'class'] = 1

# Merge the datasets
data_merge = pd.concat([data_fake, data_true], axis=0)

# Drop unnecessary columns
data = data_merge.drop(['title', 'subject', 'date'], axis=1)

# Check for null values
data.isnull().sum()

# Shuffle the dataset
data = data.sample(frac=1)

# Reset index and drop the old index column
data.reset_index(inplace=True)
data.drop(['index'], axis=1, inplace=True)

# Optimized wordopt function (combined regex and fewer operations)
def wordopt(text):
    # Use a single regex pattern for cleaning text
    text = re.sub(r'https?://\S+|WWW\.S+|<.?>|\[.?\]|\W|\d+', ' ', text)  # Remove URLs, HTML tags, non-word characters, and digits
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text.strip().lower()  # Convert to lowercase and strip

# Apply word optimization to the 'text' column
data['text'] = data['text'].apply(wordopt)

# Define features and labels
x = data['text']
y = data['class']

# Split the dataset into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Vectorization (TF-IDF with max_features to limit the number of features)
from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))  # Limit features and use unigrams and bigrams
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# Train and evaluate models

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

# Logistic Regression
LR = LogisticRegression(max_iter=1000)
LR.fit(xv_train, y_train)
pred_lr = LR.predict(xv_test)
print("Logistic Regression Accuracy: ", LR.score(xv_test, y_test))
print(classification_report(y_test, pred_lr))

# Decision Tree Classifier
DT = DecisionTreeClassifier(random_state=42)
DT.fit(xv_train, y_train)
pred_dt = DT.predict(xv_test)
print("Decision Tree Accuracy: ", DT.score(xv_test, y_test))
print(classification_report(y_test, pred_dt))

# Gradient Boosting Classifier (with fewer estimators for quicker training)
GB = GradientBoostingClassifier(n_estimators=50, random_state=42)
GB.fit(xv_train, y_train)
pred_gb = GB.predict(xv_test)
print("Gradient Boosting Accuracy: ", GB.score(xv_test, y_test))
print(classification_report(y_test, pred_gb))

# Random Forest Classifier (with fewer estimators for quicker training)
RF = RandomForestClassifier(n_estimators=50, random_state=42)
RF.fit(xv_train, y_train)
pred_rf = RF.predict(xv_test)
print("Random Forest Accuracy: ", RF.score(xv_test, y_test))
print(classification_report(y_test, pred_rf))

# Function for output labels
def output_label(n):
    return "Fake News" if n == 0 else "Not A Fake News"

# Manual testing function
def manual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    
    # Predictions from all models
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GB = GB.predict(new_xv_test)
    pred_RF = RF.predict(new_xv_test)
    
    # Print results
    print("\n\nLR Prediction: {} \nDT Prediction: {} \nGB Prediction: {} \nRF Prediction: {}".format(
        output_label(pred_LR[0]),
        output_label(pred_DT[0]),
        output_label(pred_GB[0]),
        output_label(pred_RF[0])
    ))

# Input for manual testing
news = str(input("Enter the news for manual testing: "))
manual_testing(news)





























































































































































'''
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import re
import string

# Load datasets
data_fake = pd.read_csv('Fake.csv')
data_true = pd.read_csv('True.csv')

# Add labels
data_fake["class"] = 0  # Fake news
data_true["class"] = 1  # Real news

# Create manual testing data (last 10 rows for testing)
data_fake_manual_testing = data_fake.tail(10)
data_true_manual_testing = data_true.tail(10)

# Remove manual testing rows from the main datasets
data_fake.drop(data_fake.index[-10:], inplace=True)
data_true.drop(data_true.index[-10:], inplace=True)

# Add labels to the testing data using .loc[] to avoid SettingWithCopyWarning
data_fake_manual_testing.loc[:, 'class'] = 0
data_true_manual_testing.loc[:, 'class'] = 1

# Merge the datasets
data_merge = pd.concat([data_fake, data_true], axis=0)

# Drop unnecessary columns
data = data_merge.drop(['title', 'subject', 'date'], axis=1)

# Check for null values
data.isnull().sum()

# Shuffle the dataset
data = data.sample(frac=1)

# Reset index and drop the old index column
data.reset_index(inplace=True)
data.drop(['index'], axis=1, inplace=True)

# Word optimization function (with improved regex handling)
def wordopt(text):
    # Optimize regex patterns and avoid redundant operations
    text = re.sub(r'\[.*?\]', '', text)  # Remove anything inside square brackets
    text = re.sub(r'https?://\S+|WWW\.S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'[{}]'.format(re.escape(string.punctuation)), '', text)  # Remove punctuation
    text = re.sub(r'\n', '', text)  # Remove newline characters
    text = re.sub(r'\w*\d\w*', '', text)  # Remove words with digits
    text = text.lower().strip()  # Lowercase and strip
    return text

# Apply word optimization to the 'text' column
data['text'] = data['text'].apply(wordopt)

# Define features and labels
x = data['text']
y = data['class']

# Split the dataset into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Vectorization (TF-IDF)
from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# Train and evaluate models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

# Logistic Regression
LR = LogisticRegression(max_iter=1000)
LR.fit(xv_train, y_train)
pred_lr = LR.predict(xv_test)
'''
print("Logistic Regression Accuracy: ", LR.score(xv_test, y_test))
print(classification_report(y_test, pred_lr)) 
'''

# Decision Tree Classifier
DT = DecisionTreeClassifier(random_state=42)
DT.fit(xv_train, y_train)
pred_dt = DT.predict(xv_test)
'''
print("Decision Tree Accuracy: ", DT.score(xv_test, y_test))
print(classification_report(y_test, pred_dt))
'''

# Gradient Boosting Classifier
GB = GradientBoostingClassifier(random_state=42)
GB.fit(xv_train, y_train)
pred_gb = GB.predict(xv_test)
'''
print("Gradient Boosting Accuracy: ", GB.score(xv_test, y_test))
print(classification_report(y_test, pred_gb))
'''

# Random Forest Classifier
RF = RandomForestClassifier(random_state=42)
RF.fit(xv_train, y_train)
pred_rf = RF.predict(xv_test)
'''
print("Random Forest Accuracy: ", RF.score(xv_test, y_test))
print(classification_report(y_test, pred_rf))
'''

# Function for output labels
def output_label(n):
    return "Fake News" if n == 0 else "Not A Fake News"

# Manual testing function
def manual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    
    # Predictions from all models
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GB = GB.predict(new_xv_test)
    pred_RF = RF.predict(new_xv_test)
    
    # Print results
    print("\n\nLR Prediction: {} \nDT Prediction: {} \nGB Prediction: {} \nRF Prediction: {}".format(
        output_label(pred_LR[0]),
        output_label(pred_DT[0]),
        output_label(pred_GB[0]),
        output_label(pred_RF[0])
    ))

# Input for manual testing
news = str(input("Enter the news for manual testing: "))
manual_testing(news)

'''