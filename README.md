# Disaster Response Pipeline Project

This project demonstrates a number of skills relevant to data science workflows  
* Creating Extract Transform and Load (ETL) pipelines for text data.  
* Creating Natural Language Processing (NLP) pipelines and Machine Learning (ML) pipelines for text messages.  
* Deploying trained and validated ML models to webapps.

The dataset for this project is a set of text messages sent during real disasters. The messages were translated into English and categorized into several different classes by Figure 8 and provided for use by Udacity. The message 'We need food and water over here' for instance could be classified simultaneously as 'Aid related', 'Water', 'Food' and 'Request'. A large corpus of these messages is used for training multi-output classification models which can then be deployed to a back-end (using Flask) that runs on a web-server.   
User input is obtained in the form of typed text messages into web forms. The machine learning application running on the back-end makes a prediction and sends the results to the front-end where a categories corresponding to that message are visually displayed.  

A fairly basic dashboard displaying key statistics of text messages in the database is also displayed on the front page.
    
### Files and Directories included  
1. app/  
    - templates/  
        - go.html - Udacity provided HTML file for displaying relevant classification labels. Modified for use.  
        - master.html - Udacity provided HTML file. Front page of webapp. Dashboard containing overview plots for disaster message database. Modified for use.  
    - run.py - Udacity provided Python template file containing Flask back-end code for webapp. Modified for use case.
2. data/  
    - disaster_messages.csv - Table containing disaster messages translated into English and transcribed in English from their respective original languages. Provided by Udacity.  
    - disaster_categories.csv - Table containing uncleaned classification of text messages. Provided by Udacity.  
    - DisasterResponse.db - SQL table containing clean, merged table with text messages and corresponding categories as columns. Created in this project.    
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

3. Go to http://127.0.0.0:3001/ (if running on your laptop) or an appropriate web-server address.  

4. As it stands the run.py script expects a 'classifier.pkl' model file for prediction.

