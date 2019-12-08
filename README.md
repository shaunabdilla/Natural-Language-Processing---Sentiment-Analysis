#Natural-Language-Processing---Sentiment-Analysis
Application of several classification algorithms to a scraped dataset of restaurant reviews and 
comparing performances, attempting dimensionality reduction in the process.

After importing the data included in tsv, data is cleaned, stemmed and the corpus created.
Wordclouds created to visualize "negative" versus "positive" words, and data split into appropriate sets after vectorization.

The 5 algorithms used are:

#Test 1 - Naive Bayes (Overall best)

Accuracy - What was the proportion of correct answers to incorrect ones?
0.73
F1 Score - Mean performance based on Precision and Recall 
0.77
Precision - What proportion of positive identifications was actually correct?
0.88
Recall - What proportion of actual positives was identified correctly?
0.68

#Test 2 - Decision Tree

Accuracy
0.71
F1 Score
0.70
Precision
0.66
Recall
0.74

#Test 3 - Logistic Regression

Accuracy
0.71
F1 Score
0.69
Precision
0.65
Recall
0.76


#Test 4 - Random Forest Classifier (Best recall)

Accuracy
0.72
F1 Score
0.67
Precision
0.55
Recall
0.85


#Test 5 - Max Entropy Loss

Accuracy
0.72
F1 Score
0.70
Precision
0.66
Recall
0.76


Dimensionality Reduction using PCA was found to be not useful in this case, all variables contributed equally to variance in the result.
