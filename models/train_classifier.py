import sys
import pandas as pd
import numpy as np
import re
from sklearn.externals.joblib import dump
from sqlalchemy import create_engine

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download(['punkt','wordnet','stopwords'])

from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

# Load set of stop_words globally once
stop_words = set(stopwords.words('english')) 

def load_data(database_filepath):
    '''
    Connects to SQL database table at path provided. Creates a Pandas dataframe from it.
    Separates out the Dataframe into features (input) and target (output) dataframes.
    
    Args:
        database_filepath: str
            Path to valid, cleaned SQL table containing messages and categories.
    Returns:
        X: Pandas DataFrame
            Features dataframe containing text messages
        y: Output DataFrame containing categories and their values for each message
    '''
    
    # Create SQLAlchemy connector to database at 'database_filepath'
    engine = create_engine('sqlite:///'+database_filepath)
    
    # Parse tablename
    table_name = 'cleaned_messages'
    
    # Create dataframe from connector
    df = pd.read_sql_table(table_name=table_name,con=engine)
    
    # List all columns in the dataframe
    df_cols = df.columns.tolist()
    
    # List columns that need to be excluded from output
    exclude_cols=['id','message','original','genre']
    
    # Assign 'message' column to features dataframe 'X'
    X = df['message'].copy()
    
    # Filter out columns that need to be excluded and assign the result to output dataframe 'y'
    y = df[[col for col in df_cols if col not in exclude_cols]].copy()
    
    return X, y

def tokenize_text(text):
    '''
    Normalizes case, tokenizes, lemmatizes and removes all common words in English.
    
    Args:
        text:str
            The text on which the tokenize operations need to be performed.
    Returns:
        lem_tokens: list
            A list of string tokens obtained by performing the above operations on 'text' string.
    '''
    # Make lowercase and replace punctuations marks with spaces
    text  = re.sub(pattern=r'[^a-zA-Z0-9]',string=text.lower(),repl=' ')
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Initialize Lemmatizer
    lem_tokens=[]
    lemmatizer = WordNetLemmatizer()
    
    # Loop through tokens
    for token in tokens:
        # Lemmatize each token
        lem_token = lemmatizer.lemmatize(token)

        # Append to list of valid words only if not stop word
        if lem_token not in stop_words:
            lem_tokens.append(lem_token)

    return lem_tokens


def build_model():
    '''
    Builds a machine learning pipeline for classifying messages.
    Pipeline consists of a Count Vectorizer, TF-IDF Transformer and a classifier adapted for multiple outputs.
    
    Args:
        X_train: Pandas Dataframe.
            Features Dataframe.
        y_train: Pandas Dataframe.
            Target Dataframe with columns representing message categories.
    Returns:
        pipeline: sklearn Pipeline object.
            Pipeline to be fit and cross validated
    '''
    
    # Create Pipeline consisting of CountVectorizer, TfidfTransformer and MultiOutputClassifier
    pipeline = Pipeline([('cvect',CountVectorizer(tokenizer=tokenize_text)),
                         ('tfidf',TfidfTransformer()),
                         ('multi_rf',MultiOutputClassifier(estimator=RandomForestClassifier()))
                        ])
    
    return pipeline

def fit_model(model,X_train,y_train):
    '''
    Fits model using train data. 
    
    Args:
        model: sklearn Pipeline object
            ML Pipeline created using build_model
        X_train: Python Dataframe
            Training data features
        y_train: Python Dataframe
            Training data targets
    
    Returns:
        model: sklearn Pipeline object
            Model fit on training data.
   
    '''
    
    # Fit model
    model.fit(X_train,y_train)
    
    return model

def optimize_model(model, X_train, y_train):
    '''
    Perform Grid Search CV on model. Return best estimator.
    
    Args:
        model: sklearn Pipeline object
            Model (fit or unfit) created using build_model and/or fit_model
        X_train: Python Dataframe
            Training data features
        y_train: Python Dataframe
            Training data targets            
    Returns:
        best_estimator: sklearn Pipeline object
            Best model obtained upon cross validation
    '''
    # Create parameter grid for Grid  Search CV
    parameters = {'multi_rf__estimator__n_estimators':[10,100]}
    
    # Grid Search CV object
    cv = GridSearchCV(estimator=model,param_grid=parameters,cv=3,n_jobs=8)
    cv.fit(X_train,y_train)
    
    # Return best estimator after fitting to data.
    model = build_model()
    model.set_params(multi_rf__estimator__n_estimators = cv.best_estimator_.get_params()['multi_rf__estimator__n_estimators'])
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test):
    '''
    Evaluate model on test data.
    
    Args:
        model: sklearn Pipeline with Multi Output Classifier as final stage.
            Pipeline created in build_model() above.
        X_test: Pandas Dataframe object.
            Test features dataframe.
        y_test: Pandas Dataframe object.
            Test target dataframe.
            
    Returns:
        None. Prints out results of evaluation.
        
    '''
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Loop through all categories and print a classification report for each
    for i in range(y_pred.shape[1]):
        print('Message Category -- {}'.format(y_test.columns[i]))
        report = classification_report(y_test.iloc[:,i].values,y_pred[:,i])
        print(report)


def save_model(model, model_filepath):
    '''
    Save model for future prediction.
    
    Args:
        model: sklearn Pipeline with Multi Output Classifier as final stage.
        
    Returns:
        None. Pickles and saves model at desired path.
    '''
    
    # Pickle and save
    dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        #print('Fitting model...')
        #model = fit_model(model, X_train, y_train)
        
        print('Optimizing model...')
        model = optimize_model(model,X_train,y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test)

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
