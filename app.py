import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
from gensim.models import KeyedVectors
import numpy as np
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def compute_similarity(sentence1, sentence2, similarity_type):
    # Compute similarity based on the selected type
    if similarity_type == 'tfidf':
        # TF-IDF based similarity
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([sentence1, sentence2])
        similarity = 1 - cosine(tfidf_matrix[0].toarray()[0], tfidf_matrix[1].toarray()[0])
    
    elif similarity_type == "sentencetransformer":
        # SentenceTransformer (BERT) similarity
        model = SentenceTransformer('bert-base-nli-mean-tokens')
        embeddings = model.encode([sentence1, sentence2])
        similarity = 1 - cosine(embeddings[0], embeddings[1])
    
    elif similarity_type == "word2vec":
        # Word2Vec similarity
        model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
        embeddings = [
            np.mean(
                [model[word] for word in sentence.split() if word in model] or [np.zeros(model.vector_size)],
                axis=0
            )
            for sentence in [sentence1, sentence2]
        ]
        similarity = 1 - cosine(embeddings[0], embeddings[1])
    
    else:
        raise ValueError("Invalid similarity_type. Choose either 'tfidf', 'sentencetransformer', or 'word2vec'")
    
    return similarity

# Streamlit application
st.title("Sentence Similarity Checker")

# User input
sentence1 = st.text_input("Enter the first sentence")
sentence2 = st.text_input("Enter the second sentence")

# Option to select similarity type
similarity_types = st.multiselect(
    "Choose similarity type(s):",
    ['tfidf', 'sentencetransformer', 'word2vec']
)

if st.button("Compute Similarity"):
    if sentence1 and sentence2:
        # Iterate through selected similarity types
        for sim_type in similarity_types:
            st.write(f"Computing {sim_type.capitalize()} Similarity...")
            try:
                similarity = compute_similarity(sentence1, sentence2, sim_type)
                st.write(f"{sim_type.capitalize()} Similarity: {similarity:.4f}")
            except ValueError as ve:
                st.error(f"ValueError: {str(ve)}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter both sentences.")
