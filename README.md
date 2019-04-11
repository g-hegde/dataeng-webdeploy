# Disaster Response Pipeline Project

This project demonstrates a number of skills relevant to data science workflows  
* Creating Extract Transform and Load (ETL) pipelines for text data.  
* Creating Natural Language Processing (NLP) pipelines and Machine Learning (ML) pipelines for text messages.  
* Deploying trained and validated ML models to webapps.    
### Files and Directories included  
1. app/  
    - templates/  
        - go.html - Udacity provided template for displaying relevant classification labels  
        - master.html - Udacity provided dashboard template. Front page of webapp.  
    - run.py - Udacity provided Python template file containing Flask back-end code for webapp. Modified for use case.
2. data/  
    - disaster_messages.csv - table containing disaster messages translated into english and transcribed in english from their respective original languages.  
    - disaster_categories.csv - table containing uncleaned classification of text messages.  
    - DisasterResponse.db - SQL table containing clean, merged table with text messages and corresponding categories as columns.  
    - process_data.py - ETL pipeline - Extracts data from csv files, cleans and transforms category texts into category columns and loads into SQL database.  
3. models/  
    - train_classifier.py - Creates and trains NLP and ML pipelines for text message classification.  
    - <optional> - classifier.pkl - Trained and cross-validated text classification pipeline model saved in pickle  format.  

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
