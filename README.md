# Sentiment Detector

**Overview**
This is a machine learning program that predicts the emotional sentiment (positive, neutral, negative) and intensity score of text. It uses either
a Naive Bayes or SVM algorithm to train the program. Instenisty is found using a Linear Regression algorithm.

# Before Running
Make sure you have all the necessary python packages installed
such as numpy, pandas, seaborn, etc.

When you run the file for the first time, write:
* nltk.download('punkt') 
* nltk.download('stopwords')
* nltk.download('wordnet')

# Dataset
The current version of the program expects a CSV file to have three columns, 
Sentiment, Text/Description, Score. Due to this format, the previous CSV files are currently
incompatible with the Detector. 

**Run file and Choose From Menu**

# What to Run:
  1. To see our model's quality: Will show how well the model performs (accuracy, confusion matrix, etc.)
  2. Test your own text: Input text to get the predicted sentiment and score.
  3. Compare to a prebuilt model: Compares the machine learning's accuracy to that of a VADER model.

# Choose the model:
  1. Naive Bayes (Faster, recommended)
  2. SVM (Slower, more accurate)

Note: Previous training/testing sets were left to show the difference in training data.
You can customize the dataset as you please.
