import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import nltk

# Initialize NLTK and download necessary resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize stopwords
stop_words = set(stopwords.words('english'))

# Creating the dataset
data = {
    'query': [
        'Where can I find the product ingredients?',
        'How can I contact support?',
        'What is your return policy?',
        'Do you offer international shipping?',
        'Can I change my order?',
        'Where can I find the instructions for each product?',
        'What are the benefits to these products?',
        'Have these products been tested on animals?',
        'Is there scientific research to prove the effectiveness of the products?',
        'Does this product include SPF properties?',
        'Is this safe for sensitive skin?',
        'Is this safe for children?',
        'Are the products edible?'
    ],
    'response': [
        'Each product\'s respective ingredients are listed on the product page under the ingredients section.',
        'You can contact support via email at support@lyricsipcandy.com, or reach out via any social media @lyrics.lipcandy.',
        'As our products include lip care products, for sanitation purposes all products are final sale; no returns will be accepted.',
        'At this time, we do not offer international shipping. Shipping is applicable to all 50 US states.',
        'Yes, order changes are accepted within 24 hours of placing the order. After 24 hours, all orders are final.',
        'Each product\'s respective instructions are listed on the product page under the instructions section.',
        'The benefits of these products include moisturizing, nourishing, and providing a natural form of SPF.',
        'No product from Lyrics Lip Candy LLC has been tested on animals.',
        'Currently, Lyrics Lip Candy products are being tested by the chemical engineering department at Bucknell University. CEO, Lyric Abdul-Rasheed, is conducting her own independent study utilizing LLC products to formulate the most effective natural forms of SPF for lip care products.',
        'Yes, all LLC products include natural forms of SPF. The exact value of SPF is currently being tested and researched.',
        'Yes, LLC products are all formulated using naturally derived ingredients hand-picked for their gentle properties.',
        'Yes, LLC products are formulated for sensitive skin and are safe for children.',
        'LLC products are for external use only and should not be ingested.'
    ]
}

# Create DataFrame from the dataset
df = pd.DataFrame(data)

# Example target column creation (random binary labels for demonstration)
df['Label'] = np.random.randint(0, 2, df.shape[0])

# Data Preprocessing
label_encoder = LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['Label'])

# Splitting the dataset into training and test sets
X = df['query']
y = df['Label']

# Convert text data to numerical format using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Split the transformed data
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train a simple model (Random Forest)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the model (optional)
with open('best_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Preprocess text function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(tokens)

# Load the sentence transformer model for embeddings
embedder = pipeline('feature-extraction', model='sentence-transformers/all-MiniLM-L6-v2')

# Function to get a response based on user input
def get_response(user_input):
    # Preprocess the user input
    user_input_processed = preprocess_text(user_input)
    
    # Get embedding for user input
    user_embedding = embedder(user_input_processed)
    
    # Ensure the embedding extraction returns something useful
    if len(user_embedding) == 0 or len(user_embedding[0]) == 0:
        return "Sorry, I couldn't understand your question."
    
    user_embedding = user_embedding[0][0]  # Extract the embedding for the user input

    # Get embeddings for all processed queries
    query_embeddings = embedder(df['query'].tolist())

    # Flatten the query embeddings
    query_embeddings_flat = [embedding[0][0] for embedding in query_embeddings]

    # Ensure all embeddings have the same length
    query_embeddings_array = np.vstack(query_embeddings_flat)

    # Calculate cosine similarity
    cosine_sim = cosine_similarity([user_embedding], query_embeddings_array)
    
    # Get the index of the most similar query
    response_index = np.argmax(cosine_sim)

    # Debug prints to check embeddings and cosine similarity
    print(f"User Input Processed: {user_input_processed}")
    print(f"Cosine Similarities: {cosine_sim}")
    print(f"Selected Response Index: {response_index}")
    
    # Check if the response index is valid
    if response_index < len(df):
        return df['response'].iloc[response_index]
    else:
        # If the index is invalid or out-of-bounds, return a fallback response
        return "I'm sorry, I couldn't find a suitable answer for your query."


# Streamlit Interface
st.title("Lyric's Lip Candy ChatBot")
st.write("Ask me anything about the products!")

# User input
user_input = st.text_input("Your question:")

if user_input:
    response = get_response(user_input)
    st.write("Response:", response)
