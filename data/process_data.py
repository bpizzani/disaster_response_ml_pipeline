import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = pd.merge(messages,categories,on="id")
    return df

def clean_data(df):
    #Split the categories into different columns separated by ";"
    categories = df["categories"].str.split(";",expand=True)
    
    #Get the columns names
    category_colnames = [x.split("-")[0] for x in categories.iloc[0]]
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    #Checking for non-binary classes columns
    non_binary_classes = categories.columns[categories.apply(lambda x: sorted(x.unique()) != [0, 1] and sorted(x.unique()) != [0] and sorted(x.unique()) != [1])].values
    
    #Drop rows where label is not binary (0 or 1)
    for col in non_binary_classes:
        categories = categories[categories[col] <= 1]
    
    #drop origina lcategories
    df.drop("categories",axis=1,inplace=True)
    
    #Join all toegther
    df = pd.concat([df,categories],axis=1)
    
    #drop duplicates
    df = df.drop_duplicates("id")
    return df


def save_data(df, database_filename):
    #Save dataframe into an SQL database for further consumption
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('comments', engine, index=False,if_exists='replace')

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
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
