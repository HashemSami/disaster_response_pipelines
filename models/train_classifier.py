import sys
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords

nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")
nltk.download("stopwords")
import numpy as np
import pandas as pd
import pickle
from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report

import xgboost as xgb


def load_data(database_filepath):
    engine = create_engine(f"sqlite:///{database_filepath}")
    # df = pd.read_sql('SELECT * FROM MESSAGES', con = engine)
    df = pd.read_sql_table("MESSAGES", engine)
    X = df["message"]
    y = df[df.columns[4:]]

    category_names = y.columns

    return X, y, category_names


def tokenize(text_raw):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text_raw.lower())
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words("english")

    def pos_tagger(nltk_tag):
        if nltk_tag.startswith("J"):
            return wordnet.ADJ
        elif nltk_tag.startswith("V"):
            return wordnet.VERB
        elif nltk_tag.startswith("N"):
            return wordnet.NOUN
        elif nltk_tag.startswith("R"):
            return wordnet.ADV
        else:
            return None

    # tokenize the sentence and find the POS tag for each token
    pos_tagged = nltk.pos_tag(nltk.word_tokenize(text))

    # we use our own pos_tagger function to make things simpler to understand.
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))

    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if word not in stop_words:
            if tag is None:
                # if there is no available tag, append the token as is
                lemmatized_sentence.append(word.strip())
            else:
                # else use the tag to lemmatize the token
                lemmatized_sentence.append(lemmatizer.lemmatize(word, tag).strip())

    return lemmatized_sentence


def build_model():
    xgboost = xgb.XGBClassifier(
        tree_method="hist", eval_metric="mlogloss", use_label_encoder=False
    )

    pipeline = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "text_pipeline",
                            Pipeline(
                                [
                                    ("vect", CountVectorizer(tokenizer=tokenize)),
                                    ("tfidf", TfidfTransformer()),
                                ]
                            ),
                        ),
                    ]
                ),
            ),
            ("clf", MultiOutputClassifier(xgboost)),
        ]
    )

    return pipeline


def evaluate_model(model, X_test, y_test, category_names):

    y_pred = model.predict(X_test)

    print(
        classification_report(
            y_test,
            pd.DataFrame(y_pred, columns=y_test.columns),
            target_names=y_test.columns,
        )
    )


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print("Loading data...\n    DATABASE: {}".format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print("Building model...")
        model = build_model()

        print("Training model...")
        model.fit(X_train, Y_train)

        print("Evaluating model...")
        evaluate_model(model, X_test, Y_test, category_names)

        print("Saving model...\n    MODEL: {}".format(model_filepath))
        save_model(model, model_filepath)

        print("Trained model saved!")

    else:
        print(
            "Please provide the filepath of the disaster messages database "
            "as the first argument and the filepath of the pickle file to "
            "save the model to as the second argument. \n\nExample: python "
            "train_classifier.py ../data/DisasterResponse.db classifier.pkl"
        )


if __name__ == "__main__":
    main()
