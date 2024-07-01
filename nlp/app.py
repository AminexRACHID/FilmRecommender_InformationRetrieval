from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

app = Flask(__name__)

# Chargement des données et prétraitement
finaldata = pd.read_pickle("processed_data.pkl")
# Création de la matrice tf-idf
tfidf = TfidfVectorizer()
tfidf_movieid = tfidf.fit_transform((finaldata["new_plot"]))

# Recherche de similitude cosinus entre les vecteurs
similarity = cosine_similarity(tfidf_movieid, tfidf_movieid)

# Fonction de recommandation
indices = pd.Series(finaldata.index)

def recommendations(title, cosine_sim = similarity):
  try:
    index = indices[indices == title].index[0]
    similarity_scores = pd.Series(cosine_sim[index]).sort_values(ascending = False)
    top_10_movies = list(similarity_scores.iloc[1:9].index)
    recommended_movies = [list(finaldata.index)[i] for i in top_10_movies]
    return recommended_movies
  except:
    return []

# Route pour la page d'accueil
@app.route("/")
def home():
    return render_template('index.html')

# Route pour la page de résultats
@app.route("/recommandations", methods=['POST'])
def result():
    title = request.form.get('title')
    recommended_movies = recommendations(title)
    return render_template('films.html', title=title, recommended_movies=enumerate(recommended_movies))

if __name__ == '__main__':
    app.run(debug=True)
