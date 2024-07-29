from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import random
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Suppress NLTK warnings
nltk.download('popular', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Load the corpus and initialize lemmatizer
with open('Anxiety.txt', 'r', encoding='utf8', errors='ignore') as fin:
    raw_corpus = fin.read().lower()

lemmatizer = WordNetLemmatizer()
sent_tokens = nltk.sent_tokenize(raw_corpus)

def LemNormalize(text):
    return [lemmatizer.lemmatize(token) for token in nltk.word_tokenize(text.lower())]

def greeting(sentence):
    GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey")
    GREETING_RESPONSES = [
        "hi", "hey", "nods", "I'm here to listen and assist you in finding the support you need",
        "hello", "I am glad! You are talking to me"
    ]
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

def generate_response(user_response):
    global sent_tokens
    sent_tokens.append(user_response)
    
    # TF-IDF Vectorization
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    
    # Calculating cosine similarity
    cosine_similarities = cosine_similarity(tfidf[-1], tfidf[:-1])
    
    # Get indices of top similar sentences
    similar_sentence_indices = cosine_similarities.argsort()[0][-2:-7:-1]
    
    # Concatenate multiple similar sentences to form a comprehensive response
    mira_response = ''
    for idx in similar_sentence_indices:
        mira_response += sent_tokens[idx] + ' '
    
    # If no suitable response found, return a default message
    if mira_response == '':
        mira_response = "I am sorry! I don't understand you."
    
    # Remove the user response from the sentence tokens if it exists
    if user_response in sent_tokens:
        sent_tokens.remove(user_response)
    
    return mira_response

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def get_bot_response():
    try:
        user_response = request.json.get("msg")
        user_response = user_response.lower().strip()
        
        if user_response == 'bye':
            response = "MIRA: Bye! Take care."
        elif user_response in ('thanks', 'thank you'):
            response = "MIRA: You are welcome."
        else:
            greeting_resp = greeting(user_response)
            if greeting_resp is not None:
                response = "MIRA: " + greeting_resp
            else:
                response = "MIRA: " + generate_response(user_response)
        
        return jsonify({"response": response}), 200

    except Exception as e:
        print("Error in get_bot_response:", e)
        return jsonify({"response": "Internal Server Error"}), 500

if __name__ == '__main__':
    app.run(debug=True)
