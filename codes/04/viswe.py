import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from gensim.models import KeyedVectors

# Load pre-trained Word2Vec or GloVe embeddings for different time periods
# These should be in Word2Vec format (you can convert GloVe if needed)
# Ensure you have embeddings like 'glove_1900s.txt' and 'glove_2000s.txt'
def load_embeddings():
    try:
        # Embedding from the 1900s
        model_1900s = KeyedVectors.load_word2vec_format('glove_1900s.txt', binary=False)

        # Embedding from the 2000s
        model_2000s = KeyedVectors.load_word2vec_format('glove_2000s.txt', binary=False)

        return model_1900s, model_2000s

    except FileNotFoundError:
        print("Embedding files not found. Please ensure 'glove_1900s.txt' and 'glove_2000s.txt' are available.")
        exit()

# Retrieve vectors for words from embeddings
def get_word_vectors(model_1900s, model_2000s, words):
    vectors_1900s = np.array([model_1900s[word] for word in words if word in model_1900s])
    vectors_2000s = np.array([model_2000s[word] for word in words if word in model_2000s])

    # Combine the vectors and add labels
    vectors = np.concatenate([vectors_1900s, vectors_2000s], axis=0)
    labels = words * 2  # Words are repeated for each time period
    time_labels = ['1900s'] * len(words) + ['2000s'] * len(words)

    return vectors, labels, time_labels

# Perform t-SNE on word vectors
def perform_tsne(vectors):
    tsne_model = TSNE(n_components=2, random_state=42)
    reduced_vectors = tsne_model.fit_transform(vectors)
    return reduced_vectors

# Plot the t-SNE result
def plot_tsne(reduced_vectors, labels, time_labels):
    plt.figure(figsize=(10, 6))
    
    # Plot each point
    for i, label in enumerate(labels):
        x, y = reduced_vectors[i, :]
        color = 'blue' if time_labels[i] == '1900s' else 'red'
        plt.scatter(x, y, color=color)
        plt.text(x+0.02, y+0.02, f"{label} ({time_labels[i]})", fontsize=12)
    
    plt.title('t-SNE visualization of word meanings over time')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.grid(True)
    plt.show()

# Main function to execute the workflow
def main():
    # Load embeddings from different time periods
    model_1900s, model_2000s = load_embeddings()

    # Words of interest to track semantic changes
    words = ['gay', 'broadcast', 'awful']  # You can modify this list with other words

    # Get word vectors and corresponding labels
    vectors, labels, time_labels = get_word_vectors(model_1900s, model_2000s, words)

    # Perform t-SNE
    reduced_vectors = perform_tsne(vectors)

    # Plot the t-SNE visualization
    plot_tsne(reduced_vectors, labels, time_labels)

# Run the program
if __name__ == '__main__':
    main()
