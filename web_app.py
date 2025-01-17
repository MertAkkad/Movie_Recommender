from flask import Flask, render_template, request
from Embedding_Generator import Embedding_Generator
from Faiss_Voronoi_Generator import Faiss_Voronoi_Generator
from Find_Movies import Find_Movies
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

app = Flask(__name__)

sbert_model = SentenceTransformer('all-mpnet-base-v2')
data_file = 'movies_with_keywords.csv'
embeddings_file = 'sbert_embeddings.npz'
N_Cells = 50 # Number of cluster cells
top_k = 5 

# Load dataset
def load_dataset(data_file):
    df = pd.read_csv(data_file)
    titles = df['title'].tolist()
    overviews = df['overview'].tolist()
    keywords = df['keywords'].tolist()
    indices = df['index'].tolist()
    release_dates = df['release_date'].tolist()
    runtimes = df['runtime'].tolist()
    return indices, titles, overviews, keywords, release_dates, runtimes


Faiss_Index = Faiss_Voronoi_Generator(embeddings_file, N_Cells).create_voronois()
indices, titles, overviews, keywords, release_dates, runtimes = load_dataset(data_file)
@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    if request.method == 'POST':
        user_prompt = request.form['user_prompt']
        user_embedding = sbert_model.encode(user_prompt).reshape(1, -1)
        
        Find_The_Movie = Find_Movies(Faiss_Index, user_embedding, top_k)
        top_matches = Find_The_Movie.find_most_similar_sbert()
        
        for idx in top_matches[0]:
            results.append({
                'index': indices[idx],
                'title': titles[idx],
                'overview': overviews[idx],
                'keywords': keywords[idx],
                'release_date': release_dates[idx],
                'runtime': int(runtimes[idx])
            })
    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
