from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pickle


book = pickle.load(open('book.pkl', 'rb'))
app = Flask(__name__)

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(book['Genre'])


@app.route("/")
def Book_recommendation():
    return render_template('app.html',
                           book_name=list(book['Title'].values)
                           )


@app.route("/recommend", methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        user_input = request.form['book_title']

        user_tfidf = tfidf_vectorizer.transform([user_input])

        # Computing similarity
        similarity = linear_kernel(tfidf_matrix, tfidf_matrix)

        # Recommendations
        # index = data[data['Title'] == title].index[0]
        # score = list(enumerate(similarity[index]))
        # score = sorted(score, key=lambda x: x[1], reverse=True)[1:4]
        book_indices = similarity.argsort()[1:4]
        recommended_book = book.iloc[book_indices][['Title', 'Genre']]
        return render_template('recommend.html', user_input=user_input, recommended_book=recommended_book.to_dict(orient='records'))

    return render_template('recommend.html')


if __name__ == "__main__":
    app.run(debug=True)
