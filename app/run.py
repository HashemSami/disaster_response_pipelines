import json
import plotly
import pandas as pd

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

nltk.download(
    ["omw-1.4", "punkt", "wordnet", "averaged_perceptron_tagger", "stopwords"]
)

from flask import Flask
from flask import render_template, request, jsonify
import pickle
from sqlalchemy import create_engine

from graphs import draw_bar, draw_hor_bar, draw_stacked_bar


app = Flask(__name__)


def tokenize(text):
    """
    INPUT
    text_raw - string

    OUTPUT
    lemmatized_sentence - string list

    This function will clean and tokenize the provided text
    using the following steps:
    1. Extract characters from string
    2. Tokenize words
    3. Create POS tag each token
    4. Remove stop words
    """
    lemmatizer = WordNetLemmatizer()

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
        if tag is None:
            # if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:
            # else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))

    return lemmatized_sentence


# load data
engine = create_engine("sqlite:///../data/DisasterResponse.db")
df = pd.read_sql_table("MESSAGES", engine)

# load model
model = pickle.load(open("../models/classifier.pkl", "rb"))


# index webpage displays cool visuals and receives user input text for model
@app.route("/")
@app.route("/index")
def index():

    genre_counts = df.groupby("genre").count()["message"]
    genre_names = list(genre_counts.index)

    genre_cat_counts = df.groupby("genre").sum()[df.columns[4:]]

    categories_counts = df[df.columns[4:]].sum()
    categories_names = list(categories_counts.index)

    graphs = [
        draw_bar(genre_names, genre_counts, "Distribution of Message Genres", "Genre"),
        draw_stacked_bar(
            genre_cat_counts,
            categories_names,
            "Distribution of Message Genres by Categories",
            "Genre",
        ),
        draw_hor_bar(
            categories_names,
            categories_counts,
            "Distribution of Message Categories",
            "Category",
        ),
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template("master.html", ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route("/go")
def go():
    # save user input in query
    query = request.args.get("query", "")

    print(query)

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        "go.html", query=query, classification_result=classification_results
    )


def main():
    app.run(host="0.0.0.0", port=3001, debug=True)


if __name__ == "__main__":
    main()
