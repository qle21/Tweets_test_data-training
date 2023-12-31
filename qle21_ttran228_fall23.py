import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import sklearn
import string
import re # helps you filter urls
from IPython.display import display, Latex, Markdown


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('omw-1.4')
# Verify that the following commands work for you, before moving on.

lemmatizer=nltk.stem.wordnet.WordNetLemmatizer()
stopwords=nltk.corpus.stopwords.words('english')


## Q1 (15 points):
# Convert part of speech tag from nltk.pos_tag to word net compatible format
# Simple mapping based on first letter of return tag to make grading consistent
# Everything else will be considered noun 'n'
posMapping = {
# "First_Letter by nltk.pos_tag":"POS_for_lemmatizer"
    "N":'n',
    "V":'v',
    "J":'a',
    "R":'r'
}
# 14% credits
def process(text, lemmatizer=nltk.stem.wordnet.WordNetLemmatizer()):
    """ Normalizes case and handles punctuation
    Inputs:
        text: str: raw text
        lemmatizer: an instance of a class implementing the lemmatize() method
                    (the default argument is of type nltk.stem.wordnet.WordNetLemmatizer)
    Outputs:
        list(str): tokenized text
    """
    # Text tokenization
    symbols = nltk.word_tokenize(text)

    # Correct punctuation and standardize case.
    symbols = [word.lower() for word in symbols if word.isalnum()]

    # Make a lemmatization of the symbols using part-of-speech tags.
    speechTags = nltk.pos_tag(symbols)

    label_symbols = []
    for word, tag in speechTags:
        # Get the initial letter.
        taggedInitial = tag[0].upper() if tag[0].upper() in posMapping else 'N'
        # Translate the initial letter into a another format.
        translateInitial = posMapping[taggedInitial]

        # Utilizing the mapped tag, lemmatize the word.
        lemmatized_word = lemmatizer.lemmatize(word, translateInitial)
        label_symbols.append(lemmatized_word)

    return label_symbols

## Q2 (10 points):
def process_all(df, lemmatizer=nltk.stem.wordnet.WordNetLemmatizer()):
    """ process all text in the dataframe using process() function.
    Inputs
        df: pd.DataFrame: dataframe containing a column 'text' loaded from the CSV file
        lemmatizer: an instance of a class implementing the lemmatize() method
                    (the default argument is of type nltk.stem.wordnet.WordNetLemmatizer)
    Outputs
        pd.DataFrame: dataframe in which the values of text column have been changed from str to list(str),
                        the output from process() function. Other columns are unaffected.
    """
    df['text'] = df['text'].apply(lambda x: process(x, lemmatizer))
    return df




## Q3 (15 points):

from sklearn.feature_extraction.text import TfidfVectorizer
def create_features(processed_tweets, stop_words):
    """ creates the feature matrix using the processed tweet text
    Inputs:
        processed_tweets: pd.DataFrame: processed tweets read from train/test csv file, containing the column 'text'
        stop_words: list(str): stop_words by nltk stopwords (after processing)
    Outputs:
        sklearn.feature_extraction.text.TfidfVectorizer: the TfidfVectorizer object used
            we need this to tranform test tweets in the same way as train tweets
        scipy.sparse.csr.csr_matrix: sparse bag-of-words TF-IDF feature matrix
    """
    processed_tweets['text'] = processed_tweets['text'].apply(lambda x: ' '.join(x))
    stop_words = list( stop_words)
    tokenizer = lambda x: x.split()  # Define a lambda function to tokenize the tweet text
    vectorize_tweets = TfidfVectorizer(lowercase=False, stop_words=stop_words, min_df=2, tokenizer=tokenizer)
    feature_matrix = vectorize_tweets.fit_transform(processed_tweets['text'])
    return vectorize_tweets, feature_matrix

## Q4 (10%):

from sklearn.preprocessing import LabelEncoder
def create_labels(processed_tweets):
    """ creates the class labels from screen_name
    Inputs:
        processed_tweets: pd.DataFrame: tweets read from train file, containing the column 'screen_name'
    Outputs:
        numpy.ndarray(int): dense binary numpy array of class labels
    """
    labelEncoder = LabelEncoder()
    labelResult = labelEncoder.fit_transform(processed_tweets['screen_name'])

    return labelResult

## Q5 (10 points):

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import mode
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# Skeleton of MajorityLabelClassifier is consistent with other sklearn classifiers
# 8% credits
class MajorityLabelClassifier():
    """
    A classifier that predicts the mode of training labels
    """
    def __init__(self):
        """
        Initialize your parameter here
        """
        self.mode_label = None

    def fit(self, X, y):
        """
        Implement fit by taking training data X and their labels y and finding the mode of y
        i.e. store your learned parameter
        """
        self.mode_label = mode(y).mode.item()

    def predict(self, X):
        """
        Implement to give the mode of training labels as a prediction for each data instance in X
        return labels
        """
        if X is None:
            return np.array([self.mode_label])
# Initialize your classifier
baselineClf = MajorityLabelClassifier()

# Fit the classifier to the training data
baselineClf.fit(None, y)

# Predict the labels for the training data
predicted_labels = baselineClf.predict(None)

# Calculate the accuracy by comparing the predicted labels with the true labels
accuracy = np.mean(predicted_labels == y)

# Print the training accuracy
print("Training Accuracy: ", accuracy)

## Q6 (10 points):
def learn_classifier(X_train, y_train, kernel):
    """ learns a classifier from the input features and labels using the kernel function supplied
    Inputs:
        X_train: scipy.sparse.csr.csr_matrix: sparse matrix of features, output of create_features()
        y_train: numpy.ndarray(int): dense binary vector of class labels, output of create_labels()
        kernel: str: kernel function to be used with classifier. [linear|poly|rbf|sigmoid]
    Outputs:
        sklearn.svm.SVC: classifier learnt from data
    """
    classifier = SVC(kernel=kernel)
    # Fit the classifier to the training data.
    classifier.fit(X_train, y_train)
    return classifier

## Q7 (10 points):
# 9 points credits
def evaluate_classifier(classifier, X_validation, y_validation):
    """ evaluates a classifier based on a supplied validation data
    Inputs:
        classifier: sklearn.svm.classes.SVC: classifer to evaluate
        X_validation: scipy.sparse.csr.csr_matrix: sparse matrix of features
        y_validation: numpy.ndarray(int): dense binary vector of class labels
    Outputs:
        double: accuracy of classifier on the validation data
    """
    aoc = classifier.score(X_validation, y_validation)
    return aoc

## Q8 (10 points):
def best_model_selection(kf, X, y):
    """
    Select the kernel giving best results using k-fold cross-validation.
    Other parameters should be left default.
    Input:
    kf (sklearn.model_selection.KFold): kf object defined above
    X (scipy.sparse.csr.csr_matrix): training data
    y (array(int)): training labels
    Return:
    best_kernel (string)
    """
    kernels = ['linear', 'rbf', 'poly', 'sigmoid']
    avg_accuracies = []
    for kernel in ['linear', 'rbf', 'poly', 'sigmoid']:
        # Use the documentation of KFold cross-validation to split ..
        # training data and test data from create_features() and create_labels()
        # call learn_classifer() using training split of kth fold
        # evaluate on the test split of kth fold
        # record avg accuracies and determine best model (kernel)
    #return best kernel as string
        accuracies = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clf = SVC(kernel=kernel)
            clf.fit(X_train, y_train)
            accuracy = clf.score(X_test, y_test)
            accuracies.append(accuracy)

        avg_accuracy = np.mean(accuracies)
        avg_accuracies.append(avg_accuracy)

    best_kernel_index = np.argmax(avg_accuracies)
    best_kernel = kernels[best_kernel_index]

    return best_kernel
#Test your code
best_kernel = best_model_selection(kf, X, y)
best_kernel

## Q9 (10 points)
def classify_tweets(tfidf, classifier, unlabeled_tweets):
    """ predicts class labels for raw tweet text
    Inputs:
        tfidf: sklearn.feature_extraction.text.TfidfVectorizer: the TfidfVectorizer object used on training data
        classifier: sklearn.svm.SVC: classifier learned
        unlabeled_tweets: pd.DataFrame: tweets read from tweets_test.csv
    Outputs:
        numpy.ndarray(int): dense binary vector of class labels for unlabeled tweets
    """
    # Preprocess the unlabeled tweets
    unlabeled_features = tfidf.transform(unlabeled_tweets['text'])

    # Make predictions using the classifier
    predictions = classifier.predict(unlabeled_features)

    # Return the predicted class labels
    return predictions
