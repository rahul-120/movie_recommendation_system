from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

# Load your movie dataset
# Replace 'movies.csv' with the actual filename of your dataset
movies_df = pd.read_csv('movies.csv')

# Perform TF-IDF vectorization on the 'genre' and 'storyline' columns
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
movies_df['combined_features'] = movies_df['genre'] + ' ' + movies_df['storyline']
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['combined_features'])

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Map movie titles to their indices
indices = pd.Series(movies_df.index, index=movies_df['title']).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    movie_indices = [i[0] for i in sim_scores[1:11]]  # Get top 10 similar movies
    return movies_df['title'].iloc[movie_indices].tolist()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        movie_title = request.form.get('movie_title')

        if movie_title not in movies_df['title'].values:
            return jsonify({'status': 'error', 'message': 'Movie not found'})

        recommendations = get_recommendations(movie_title)

        return jsonify({'status': 'success', 'recommendations': recommendations})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
