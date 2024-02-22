from flask import Flask, render_template, request, jsonify
import joblib
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

app = Flask(__name__)

# Load the trained model and vectorizer
clf = joblib.load('/Users/vallirajasekar/Desktop/NLP_Challenge/Disaster_Tweet/multinomial_nb_classifier.pkl')
vectorizer = joblib.load('/Users/vallirajasekar/Desktop/NLP_Challenge/Disaster_Tweet/tfidf_vectorizer.pkl')

# Download NLTK resources (only needs to be done once)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['tweet_text']
    cleaned_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = clf.predict(vectorized_text)[0]
    sentiment = "Real Disaster" if prediction == 1 else "Not a Real Disaster"
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
