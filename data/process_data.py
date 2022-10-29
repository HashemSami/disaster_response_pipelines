import pandas as pd
from sqlalchemy import create_engine
import sys


def load_data(messages_filepath, categories_filepath):
    """
    INPUT
    messages_filepath - string
    categories_filepath - string

    OUTPUT
    df - pandas dataframe

    This function loads the data from file paths and
    merge them in one dataframe using the following steps:
    1. Load messages data
    2. Load categories data
    3. Merging data into one dataframe
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on="id")

    return df


def clean_data(df):
    """
    INPUT
    df - pandas dataframe

    OUTPUT
    df - pandas dataframe

    This function will clean the dataframe using the following steps:
    1. Create a column for each category with the correct value
    2. Drop the Categories column
    3. Drop duplicates from data
    """
    categories = (
        pd.Series(df.iloc[0]["categories"])
        .str.split(";", expand=True)
        .apply(lambda x: x.str[:-2])
    )
    categories_df = pd.DataFrame(columns=list(categories.values))

    for column in categories_df:
        # set each value to be the last character of the string
        df[column[0]] = (
            df["categories"]
            .apply(
                lambda x: x[
                    x.index(column[0]) : x.index(column[0]) + len(column[0]) + 2
                ]
            )
            .str[-1]
        )

        # convert column from string to numeric
        df[column[0]] = pd.to_numeric(df[column[0]], downcast="integer")

        df.loc[df[column[0]] > 1, column[0]] = 1

    df.drop_duplicates(subset="id", keep="first", inplace=True)

    df.drop(["categories"], axis=1, inplace=True)

    return df


def save_data(df, database_filename):
    """
    INPUT
    df - pandas dataframe
    database_filename - string

    OUTPUT
    void

    This function will save the dataframe into SQL table called 'MESSAGES'
    and save the database inside the current directory.
    """
    engine = create_engine(f"sqlite:///{database_filename}")
    df.to_sql("MESSAGES", engine, if_exists="replace", index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(
            "Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}".format(
                messages_filepath, categories_filepath
            )
        )
        df = load_data(messages_filepath, categories_filepath)

        print("Cleaning data...")
        df = clean_data(df)

        print("Saving data...\n    DATABASE: {}".format(database_filepath))
        save_data(df, database_filepath)

        print("Cleaned data saved to database!")

    else:
        print(
            "Please provide the filepaths of the messages and categories "
            "datasets as the first and second argument respectively, as "
            "well as the filepath of the database to save the cleaned data "
            "to as the third argument. \n\nExample: python process_data.py "
            "disaster_messages.csv disaster_categories.csv "
            "DisasterResponse.db"
        )


if __name__ == "__main__":
    main()
