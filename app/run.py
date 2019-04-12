#!/usr/bin/python3
import json
import plotly
import pandas as pd

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
import sys
sys.path.append('../models')
from train_classifier import tokenize_text


app = Flask(__name__)



# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('cleaned_messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    top5_cat_counts, top5_categories = get_top5_categories(df)
    top5_word_counts, top5_words = get_top5_words(df)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }    
        },
        {
            'data': [
                Bar(
                    x=top5_categories,
                    y=top5_cat_counts
                )
            ],

            'layout': {
                'title': 'Top 5 categories of disaster messages',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Disaster Message Categories"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=top5_words,
                    y=top5_word_counts
                )
            ],

            'layout': {
                'title': 'Top 5 words by frequency of occurrence',
                'yaxis': {
                    'title': "Total count in database"
                },
                'xaxis': {
                    'title': "Words in Disaster messages"
                }
            }    
        }     
         
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

def get_top5_categories(df):
    '''
    Returns top 5 message categories in the database
    
    Args:
        df: Pandas dataframe
            Dataframe containing messages and categories
        
    Returns:
        top5_counts, top5_categories: lists of type 'int' and 'str' respectively
            Lists containing top 5 categories and their respective counts in the database
    '''
    
    # Filter out non-category columns and create categories dataframe
    exclude_columns = ['id','message','genre','original']
    category_columns = [col for col in df.columns.tolist() if col not in exclude_columns]
    categories = df[category_columns].copy()
    
    # Sum the columns and sort in descending order
    categories_sum_sorted = categories.sum().sort_values(ascending=False)
    
    # Extract top 5 categories and their counts and return
    top5_categories = categories_sum_sorted.index[:5].tolist()
    top5_counts = categories_sum_sorted[:5].values
    
    return top5_counts, top5_categories

def get_top5_words(df):

    '''
    Returns top 5 words in the database by frequency of occurrence and their respective counts
    
    Args:
        df: Pandas dataframe
            Dataframe containing messages
        
    Returns:
        top5_word_counts, top5_words: lists of type 'int' and 'str' respectively
            Lists containing top 5 words and their respective counts in the database
    '''
    # Iterate over messages and use Hashing to quickly find frequency
    all_words=[]
    freq_dict={}
    
    for message in df['message']:
        
        tokens = tokenize_text(message)
        
        for token in tokens:
            
            if token in freq_dict:
                freq_dict[token]+=1
            else:
                freq_dict[token]=1
    
    # Convert dictionary to Series, sort and return top 5
    freq_series = pd.Series(freq_dict).sort_values(ascending=False)
    top5_words = freq_series.index[:5].tolist()
    top5_word_counts = freq_series[:5].values
    
    return top5_word_counts, top5_words


def main():
    app.run(host='127.0.0.1', port=3001, debug=True)


if __name__ == '__main__':
    main()
