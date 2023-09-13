import sys
import pandas as pd
from sqlalchemy import create_engine
import pickle

import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer

import sklearn
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.model_selection import GridSearchCV

import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.base import BaseEstimator, TransformerMixin

def load_data(database_filepath):
    """
    Loads Database data from SQL engine database_filepath.
 
    Args:
        database_filepath (string): Database directory name.
 
    Returns:
        array: Splited X, Y values and category names list.
    """
    # load data from database
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table("comments",engine)
    X = df["message"].values
    Y = df.iloc[:,4:].values 
    category_names = df.iloc[:,4:].columns 
    return X,Y, category_names

def tokenize(text):
    """
    Tokenizes and cleans original text  / comments.
 
    Args:
        text (string): Text / comments.
 
    Returns:
        array: clean word tokens.
    """
    # Tokenize text into words 
    tokens = word_tokenize(text)
    lemmer = WordNetLemmatizer()

    #Clean tokens by lemmetizing them and removing stopwords
    clean_tokens = [lemmer.lemmatize(x.lower().strip()) for x in tokens if x.lower().strip() not in stopwords.words('english')]
    return clean_tokens

def build_model():
    """
    #Create a Pipeline model that vectorizes text and pass it to a Random Forest with Multioutput labels.
 
    Returns:
        model: pipeline model.
    """
    model = Pipeline([("vect",TfidfVectorizer(tokenizer=tokenize)),
                         ("clf",MultiOutputClassifier(RandomForestClassifier()))
                    ])
    return model

def build_GridSearch_model(model):
    # Test different min sample leafs for the RandomForest estimator. Not used in this script, but 2 is the most optimal as I tested offline.
    parameters = {
              "clf__estimator__min_samples_leaf":[1,2,5]
                 }
    cv = GridSearchCV(pipeline, parameters)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates model's performance using classification_report
 
    Args:
        model (sklearn): ML Pipeline.
        X_test (array): X test values.
        Y_test (array): Y test labels.
        category_names (array): labels names.
 
    Returns:
        print: Each category name classification_report.
    """
    y_pred = model.predict(X_test)
    #Run Classification report for each column / label
    for i,col_name in enumerate(category_names):
        print(col_name)
        print(classification_report(Y_test[:,i],y_pred[:,i]))



def save_model(model, model_filepath):
    """
    Save model into directory
 
    Args:
        model (sklearn): ML Pipeline.
        model_filepath (string): filepath / name of the model to save.
 
    Returns:
        Saves model into directory
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
