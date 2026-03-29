import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LinearRegression

#Global Variables
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.translate(str.maketrans('','',string.punctuation))
    token = word_tokenize(text.lower())
    filtered = []
    for i in token:
        if i not in stop_words:
            filtered.append(lemmatizer.lemmatize(i))
    
    return ' '.join(filtered)
    
def read_file(filename):
    df = pd.read_csv(filename)
    df.columns = ['Sentiment','Text','Score']
    return df 

def display(confusion, precision, recall):
    plt.figure(figsize=(6,4))
    labels = ['Negative','Neutral','Positive']
    sb.heatmap(confusion, annot=True, fmt='d', cmap='Greens', xticklabels=labels,yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title("Confusion Matrix")
    plt.show()
  
    plt.bar(labels,precision,color='green')
    plt.title("Precision")
    plt.show()
   
    plt.bar(labels,recall,color='green')
    plt.title("Recall")
    plt.show()

def emotion_score(X_train, y_train, X_test):
    vector = CountVectorizer()
    X_train2 = vector.fit_transform(X_train)
    X_test2 = vector.transform(X_test)

    model = LinearRegression()
    model.fit(X_train2,y_train)

    score = model.predict(X_test2)
    return score

def prebuilt_model(X_test):
    analyzer = SentimentIntensityAnalyzer()
    predictions = []
    for i in X_test:
        score = analyzer.polarity_scores(i)
        score1 = score['compound']
        if score1 >= 0.2:
            predictions.append('positive')
        elif score1 <= -0.2:
            predictions.append('negative')
        else:
            predictions.append('neutral')
    return np.array(predictions)


def my_model(X_train, y_train, X_test, model):
    vector = CountVectorizer()
    X_train2 = vector.fit_transform(X_train)
    X_test2 = vector.transform(X_test)

    if model == '1':
        model = MultinomialNB()
    elif model == '2':
        model = SVC()
    model.fit(X_train2,y_train)

    pred = model.predict(X_test2)
    return pred

def main():
    # Load dataset
    filename = "train5.csv"
    df = read_file(filename)
    df['Text'] = df['Text'].astype(str)
    df['Text'] = df['Text'].apply(preprocess)
    X, y=df['Text'].values, df['Sentiment'].values

    print("-----------------------------------------------------")
    print("Hello! This program detects the emotional tone of text." )
    print("-----------------------------------------------------")

    version = None
    while version != 'q':
        print("1- To see our model's quality.")
        print("2- Test your own text.")
        print("3- Comapare to a prebuilt model")
        print("q to exit")
        version = input("Enter: ").strip()

        if(version=='1'):
            print("Choose model")
            print("1- Naive Bayes")
            print("2- SVM")
            model = input("Enter Model: ").strip()

            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,test_size=.4)
            
            pred = my_model(X_train, y_train, X_test, model)

            accuracy = accuracy_score(y_test,pred)
            precision = precision_score(y_test,pred, average=None)
            recall = recall_score(y_test,pred,average=None)
            confusion = confusion_matrix(y_test,pred)
            print("Accuracy: ", accuracy)
            print("Precision: ", precision)
            print("Recall: ", recall)
            print("Confusion Matrix: ", confusion)
            print()

            display(confusion, precision, recall)
        elif(version=='2'):
            text = input("Input text: ")

            print("Choose model")
            print("1- Naive Bayes")
            print("2- SVM")
            model = input("Enter Model: ").strip()
            
            X_train, y_train = X, y
            text1 = preprocess(text)
            X_test = [text1]

            train_scores = df['Score'].values
            score = emotion_score(X_train,train_scores,X_test)
            pred = my_model(X_train, y_train, X_test, model)
            
            rounded_score = round(score[0])
            print(f"The program predicts the sentiment of the text is {pred[0]} with an emotional score of {rounded_score}")
            print()
        elif(version=='3'):
            print("Choose model")
            print("1- Naive Bayes")
            print("2- SVM")
            model = input("Enter Model: ").strip()

            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,test_size=.4)
            
            pred = my_model(X_train, y_train, X_test, model)

            print("MY DETECTOR")
            accuracy = accuracy_score(y_test,pred)
            precision = precision_score(y_test,pred, average=None)
            recall = recall_score(y_test,pred,average=None)
            confusion = confusion_matrix(y_test,pred)
            print("Accuracy: ", accuracy)
            print("Precision: ", precision)
            print("Recall: ", recall)
            print("Confusion Matrix: ", confusion)

            v_pred = prebuilt_model(X_test)

            print("VADER")
            v_accuracy = accuracy_score(y_test,v_pred)
            v_precision = precision_score(y_test,v_pred,average=None)
            v_recall = recall_score(y_test,v_pred,average=None)
            v_confusion = confusion_matrix(y_test,v_pred)
            print("Vader Accuracy: ", v_accuracy)
            print("Vader Precision: ", v_precision)
            print("Vader Recall: ", v_recall)
            print("Confusion Matrix: ", v_confusion)

            print(f"VADER is {v_accuracy-accuracy} more accurate than my model")
            print()
        else:
            print(">:(")

if __name__ == "__main__":
    main()

