import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import string
import os

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Define a function to preprocess text
def preprocess_text(text):
    # Tokenize sentences into words
    tokens = word_tokenize(text.lower())

    # Remove punctuation and stopwords
    tokens = [word for word in tokens if word.isalpha()]  # Keep only alphabetic tokens
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords

    return tokens

# Function to load and preprocess corpus
def load_and_preprocess_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    sentences = nltk.sent_tokenize(text)  # Split text into sentences
    processed_sentences = [preprocess_text(sentence) for sentence in sentences]  # Process each sentence
    return processed_sentences

# Function to train and save Word2Vec model
def train_word2vec(corpus, output_file):
    # Train Word2Vec model
    model = Word2Vec(corpus, vector_size=100, window=5, min_count=5, workers=4)
    
    # Save in Word2Vec text format (not binary)
    model.wv.save_word2vec_format(output_file, binary=False)
    print(f"Word2Vec model saved to {output_file}")

# Prepare the text files
def prepare_embeddings():
    # Define file paths
    file_1900s = '1900s_corpus.txt'  # Replace with your own 1900s corpus file
    file_2000s = '2000s_corpus.txt'  # Replace with your own 2000s corpus file

    # Load and preprocess the text files
    print("Processing 1900s corpus...")
    corpus_1900s = load_and_preprocess_corpus(file_1900s)
    
    print("Processing 2000s corpus...")
    corpus_2000s = load_and_preprocess_corpus(file_2000s)

    # Train Word2Vec models for each corpus
    print("Training Word2Vec for 1900s...")
    train_word2vec(corpus_1900s, 'glove_1900s.txt')

    print("Training Word2Vec for 2000s...")
    train_word2vec(corpus_2000s, 'glove_2000s.txt')

# Main function
if __name__ == '__main__':
    prepare_embeddings()
