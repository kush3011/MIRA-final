# import necessary libraries
import io
import random
import string
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import WordNetLemmatizer

# suppress warnings
warnings.filterwarnings('ignore')

# download necessary NLTK data
nltk.download('popular', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# define lemmatizer
lemmatizer = WordNetLemmatizer()

# define greeting inputs and responses
GREETING_INPUTS = ("hello","hi", "greetings", "sup", "what's up", "hey",)
GREETING_RESPONSES = ["hi", "hey", "nods","I'm here to listen and assist you in finding the support you need", "hello", "I am glad! You are talking to me"]

# read the corpus
with open('Anxiety.txt', 'r', encoding='utf8', errors='ignore') as fin:
    raw = fin.read().lower()

# tokenize the corpus
sent_tokens = nltk.sent_tokenize(raw)

# define lemmatization function
def LemNormalize(text):
    return [lemmatizer.lemmatize(token) for token in nltk.word_tokenize(text.lower())]

# define greeting function
def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# define response function
# define response function
def response(user_response):
    mira_response = ''
    sent_tokens.append(user_response)
    
    # TF-IDF Vectorization
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    
    # Calculating cosine similarity
    cosine_similarities = cosine_similarity(tfidf[-1], tfidf[:-1])
    
    # Get indices of top similar sentences
    similar_sentence_indices = cosine_similarities.argsort()[0][-2:-6:-1]
    
    # Concatenate multiple similar sentences to form a comprehensive response
    for idx in similar_sentence_indices:
        mira_response += sent_tokens[idx] + ' '
    
    # If no suitable response found, return a default message
    if mira_response == '':
        mira_response = "I am sorry! I don't understand you."
    
    # Remove the user response from the sentence tokens if it exists
    if user_response in sent_tokens:
        sent_tokens.remove(user_response)
    
    return mira_response

# Main program loop
flag = True
print("MIRA Hi I am MIRA.I'm here to provide support, guidance,through any challenges you may be facing with your mental health.")
while flag:
    user_response = input()
    user_response = user_response.lower()
    
    if user_response != 'bye':
        if user_response in ('thanks', 'thank you'):
            flag = False
            print("MIRA: You are welcome..")
        else:
            if greeting(user_response) is not None:
                print("MIRA: " + greeting(user_response))
            else:
                print("MIRA: ", end="")
                print(response(user_response))
                # Remove the user response from the sentence tokens if it exists
                if user_response in sent_tokens:
                    sent_tokens.remove(user_response)
    else:
        flag = False
        print("MIRA: Bye! take care..")