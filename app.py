from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pickle


book = pickle.load(open('book.pkl', 'rb'))
app = Flask(__name__)


@app.route("/")
def Book_recommendation():
    return render_template('app.html',
                           book_name=list(book['Title'].values)
                           )


@app.route("/recommend")
def recommend():
    return render_template('recommend.html')


if __name__ == "__main__":
    app.run(debug=True)
