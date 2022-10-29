# Disaster Response Pipeline Project

### Generel info:

This project is for the Udacity's Data Scientist Nanodegree, to analyze disaster data from [Appen](https://appen.com/) (formally Figure 8) and build a model for an API that classifies disaster messages. The data set containing real messages that were sent during disaster events, and this project will create a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.
The project will also include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

### Instructions:

1. To install the required libraries you will need to execute the command in the project's root directory:

```
pip install -r requirements.txt
```

2. Run the following commands in the project's root directory to set up your database and model.

   - To run ETL pipeline that cleans data and stores in database
     `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
   - To run ML pipeline that trains classifier and saves
     `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Run the following command in the app's directory to run your web app.
   `python run.py`

4. Go to http://0.0.0.0:3001/

### Files structure:

```bash
├── requirments.txt
├── README.md - This file.
├── app
│   ├── template
│   │   ├── master.html # main page of web app
│   │   └── go.html # classification result page of web app
│   ├── graph.py # helper functions for graphs
│   └── run.py # Flask file that runs app
├── data
│   ├── disaster_categories.csv # data to process
│   ├── disaster_messages.csv # data to process
│   ├── process_data.py
│   └── InsertDatabaseName.db # database to save clean data to
└── models
    ├── train_classifier.py
    └── classifier.pkl # saved model
```
