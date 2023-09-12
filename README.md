# Disaster Response Pipeline Project
### Project Summary:
The project aims to classify into different categories (36) comments from people related to different disasters situations. 

The Projects is composed of tree basic steps:
1. Create an ETL (extract and clean the data)
2. Create a ML Pipeline (Extract date and prepare it for training)
3. Deployed a Web App to consume the model's predictions.
4. 
Data Source: Figure 8 & Udacity


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage
