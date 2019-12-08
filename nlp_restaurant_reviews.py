# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 15:00:50 2019

@author: Shaun
"""
#Natural Language Processing - Restaurant Reviews

#Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter = "\t", quoting = 3)

#Cleaning the text
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
#------------------------------------------------------------------------------

corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', " ", dataset['Review'][i])
    review = review.lower()
    
    #Split review into different words through a loop, create a ps of porterstemmer class
    review = review.split()
    ps = PorterStemmer()
    
    #Looping through list and taking stems of non-stopwords
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = " ".join(review)
    corpus.append(review)

def process_content(content):
    return " ".join(re.findall("[A-Za-z]+",content.lower()))

dataset["Review"] = dataset["Review"].apply(process_content)
#------------------------------------------------------------------------------

#Creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

#------------------------------------------------------------------------------
#Visualising "negative" vs "positive" vocabulary
from wordcloud import WordCloud

#Negative
neg_reviews = dataset[dataset['Liked'] == 0]
negative_string = []
for t in neg_reviews.Review:
    negative_string.append(t)
negative_string = pd.Series(negative_string).str.cat(sep=' ')    

#Wordcloud for negative reviews
wordcloud = WordCloud(width=1600, height=800,max_font_size=200, background_color="red").generate(negative_string)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

#Positive
pos_reviews = dataset[dataset['Liked'] == 1]
positive_string = []
for t in pos_reviews.Review:
    positive_string.append(t)
positive_string = pd.Series(positive_string).str.cat(sep=' ')    

#Wordcloud for positivs
wordcloud = WordCloud(width=1600, height=800,max_font_size=200, background_color="green").generate(positive_string)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

#------------------------------------------------------------------------------

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Checking the split of positive/negative per training and test set
print("Training set has a total of {0} entries with {1:.2f}% \
      negative reviews, {2:.2f}% positive reviews".format \
      (len(X_train), (len(X_train[y_train == 0]) / (len(X_train)*1.))*100, \
      (len(X_train[y_train == 1]) / (len(X_train)*1.))*100))

#Checking the split of positive/negative per training and test set
print("Test set has a total of {0} entries with {1:.2f}% \
      negative reviews, {2:.2f}% positive reviews".format \
      (len(X_test), (len(X_test[y_test == 0]) / (len(X_test)*1.))*100, 
      (len(X_test[y_test == 1]) / (len(X_test)*1.))*100))
#---------------------------------------------------------------------

#Testing PCA variance explanation
#---------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler = StandardScaler()
X_train_2 = scaler.fit_transform(X_train)
X_train_2 = PCA().fit(X_train_2)

fig, ax = plt.subplots(figsize = (8,6))
x_values = range(1, X_train_2.n_components_+1)
ax.plot(x_values, X_train_2.explained_variance_ratio_, lw = 2, label="explained variance")
ax.plot(x_values, np.cumsum(X_train_2.explained_variance_ratio_), lw=2, label="cumulative explained variance")
ax.set_title("Explained variance of components")
ax.set_xlabel("Principal Component")
ax.set_ylabel("Explained Variance")
plt.show()

'''From PCA results, each of the components contributes 
the same amount to the variance, and the blue line is almost 
straight, so PCA does not help in this instance.'''

#----------------------------------------------------------------------
# Fitting classifier to the Training set
#Test 1 - Naive Bayes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

#Test 2 - Decision Tree
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

#Test 3 - Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

#Test 4 - Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

#Test 5 - Max Entropy Loss
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0, solver = 'newton-cg', multi_class = "multinomial")
classifier.fit(X_train, y_train)

#------------------------------------------------------------------------------

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
Accuracy = (cm[0][0]+cm[1][1])/sum(sum(cm))
Precision = (cm[1][1])/(cm[1][1]+cm[1][0])
Recall = cm[1][1]/(cm[1][1] + cm[0][1])
F1_Score = 2 * Precision * Recall/(Precision + Recall)