import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def extract_data(messages_filepath, categories_filepath):
    '''
    Extracts data from csv files containing disaster response messages and categories respectively.
    Merges these and returns a Pandas Dataframe.
    
    Args:
        messages_filepath: str
            Valid filepath for file containing disaster messages in .csv format.
        categories_filepath: str
            Valid filepath for file containing categories for each message in file at 'messages_filepath' above.
    Returns:
        df: Pandas dataframe containing messages and categories.
    '''
    
    # Read csv files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # Merge dataframes
    df = pd.merge(left=messages,right=categories,on='id',how='left')
    
    return df
    
def append_child_alone(categories,df):
    '''
    The original dataframe has no rows corresponding to child_alone=1.
    This makes it fail for simple classifiers like LogisticRegression.
    The function appends a few rows containing sample messages and child_alone=1
    
    Args:
        categories: Pandas dataframe
            Dataframe with categories and values
        df: Pandas Dataframe
            Dataframe containing messages
            
    Returns:
        Same dataframes as arguments but with child_alone = 1 and corresponding messages appended.
    '''
    child_alone_messages = ['My child is alone',
                            'Children are by themselves and no one to help',
                            'son alone',
                            'daughter alone',
                            'kids alone',
                            'So many children are fending for themselves',
                            'children have no parents',
                            'the parents of this child are dead',
                            'anybody home? kid alone and no one around',
                            'I am alone. 12 years old and parents are dead']*10
    
    # Append rows corresponding to child_alone=1 since this has no 1's
    # Add 100 rows since we don't know how the training and test datasets will be split
    for i in range(100):
        bottom_row = categories.iloc[-1]
        bottom_row = bottom_row.apply(lambda x:1)
        bottom_row['child_alone']=1
        categories.append(bottom_row,ignore_index=True)
    
    # Append a row to df with a message corresponding to child alone
        df_bottom_row = pd.Series([3456784567+i,child_alone_messages[i],\
                                   child_alone_messages[i],'direct','child_alone:1'],index = df.columns)
        df = df.append(df_bottom_row,ignore_index=True)
        
    return categories, df

def transform_data(df):
    '''
    Transforms dataframe to a form that can be fed into a NLP pipeline.
    Expects a 'categories' column in 'df' in format '<category>-<value>;<category>-<value>;<category>-<value>....'
    
    Args:
        df: Pandas dataframe
            Dataframe containing messages and their categories in un-processed format.
    Returns:
        transformed_df: Pandas dataframe
            Dataframe with messages and cleaned categories columns.
    '''
    transformed_df = df.copy()
    
    # Split category column and assign each category to new column in a dataframe
    categories = df['categories'].str.split(pat=';',expand=True)
    
    # Assign category names by splitting <category>-<value> from first  row of categories dataframe
    categories.columns = [cat.split('-')[0] for cat in categories.iloc[0].values]
    
    # Assign category values by splitting each cell in dataframe
    categories = categories.applymap(lambda x:x.split('-')[1])
    
    # The 'related' column contains 0, 1 and 2, map all 2 values to 1
    categories['related'].map({'0':'0','1':'1','2':'1'})

    # Convert dataframe to integer format
    categories = categories.astype(int)
    
    # Append data corresponding to child_alone=1 since the original dataframe has none
    categories, df = append_child_alone(categories,df)
    
    # Drop 'categories' column in original dataframe
    transformed_df = transformed_df.drop(columns=['categories'],axis=1)
    
    # Concatenate df with categories dataframe to form new dataframe
    transformed_df = pd.concat([transformed_df,categories],axis=1)
    
    # Drop duplicates
    transformed_df = transformed_df.drop_duplicates()
    
   
    return transformed_df


def load_data(df, database_filename):
    '''
    Load data from database into SQL database at specified path.
    
    Args:
        df: Pandas dataframe
            Cleaned dataframe with messages and categories.
        database_filename: str
            Path to SQL database table.
    Returns:
        None. Saves SQL table upon successful execution.
    '''
    
    # Create SQLAlchemy engine
    engine = create_engine('sqlite:///'+database_filename)
    
    # Save dataframe to SQL table named 'cleaned_messages'
    df.to_sql('cleaned_messages', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Extracting data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = extract_data(messages_filepath, categories_filepath)

        print('Transforming data...')
        df = transform_data(df)
        
        print('Loading data into SQL database...\n    DATABASE: {}'.format(database_filepath))
        load_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
