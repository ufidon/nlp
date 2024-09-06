import math, copy
from collections import defaultdict, Counter

# Corpus and frequencies
corpus = [("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)]

# initial_vocab = ["h", "u", "g", "hu", "ug", "p", "pu", "n", "un", "b", "bu", "s", "hug", "gs", "ugs"]
# subword_freq = {
#     "h": 15, "u": 36, "g": 20, "hu": 15, "ug": 20, "p": 17, "pu": 17, 
#     "n": 16, "un": 16, "b": 4, "bu": 4, "s": 5, "hug": 15, "gs": 5, "ugs": 5
# }

# Function to extract all strict substrings from a word
def get_substrings(word):
    substrings = set()
    n = len(word)
    for i in range(n):
        for j in range(i + 1, n + 1):
            substrings.add(word[i:j])  # Add substring word[i:j]
    substrings.discard(word)
    return substrings

# Generate initial vocabulary and subword frequencies
def generate_vocab_and_freq(corpus):
    initial_vocab = set()  # Use set to avoid duplicates
    subword_freq = defaultdict(int)  # Dictionary to count frequencies

    for word, count in corpus:
        substrings = get_substrings(word)
        for subword in substrings:
            initial_vocab.add(subword)
            subword_freq[subword] += count  # Count the occurrence of each subword
    for word, count in corpus:
        for subword in substrings:
            if subword == word:
                subword_freq[subword] += count
    return list(initial_vocab), dict(subword_freq)

# Function to compute probability of a subword
def subword_prob(subword, freq):
    return freq.get(subword, 0) / sum(freq.values())

# Function to find the best tokenization using the Viterbi algorithm
def best_tokenization(word, freq):
    n = len(word)
    # Scores and best segmentations for each position in the word
    scores = [float('-inf')] * (n + 1)
    segmentations = [[] for _ in range(n + 1)]
    scores[0] = 0  # Starting score

    # Iterate over each position in the word
    for i in range(n):
        for j in range(i + 1, n + 1):
            subword = word[i:j]
            if subword in freq:
                prob = subword_prob(subword, freq)
                score = scores[i] + math.log(prob)  # Add log-probability
                if score > scores[j]:  # Update score if better
                    scores[j] = score
                    segmentations[j] = segmentations[i] + [subword]

    # The best tokenization ends at position n
    return segmentations[-1], math.exp(scores[-1])

# Loss computation for the corpus
def compute_loss(corpus, freq):
    loss = 0
    for word, count in corpus:
        tokenization, prob = best_tokenization(word, freq)
        loss -= count * math.log(prob)
    return loss

# Generate initial_vocab and subword_freq
_, subword_freq = generate_vocab_and_freq(corpus)

# Test tokenization on corpus
for word, count in corpus:
    tokenization, prob = best_tokenization(word, subword_freq)
    print(f"Word: {word}, Best Tokenization: {tokenization}, Probability: {prob:.6f}")

# Compute loss for the corpus
loss = compute_loss(corpus, subword_freq)
print(f"Total Loss: {loss:.2f}")

# Test tokenization on corpus: removing 'pu'
freq1 = copy.deepcopy(subword_freq)
freq1.pop('pu')
for word, count in corpus:
    tokenization, prob = best_tokenization(word, freq1)
    print(f"Word: {word}, Best Tokenization: {tokenization}, Probability: {prob:.6f}")

# Compute loss for the corpus
loss = compute_loss(corpus, freq1)
print(f"Total Loss: {loss:.2f}")

# Test tokenization on corpus: removing 'hug'
freq2 = copy.deepcopy(subword_freq)
freq2.pop('hug')
for word, count in corpus:
    tokenization, prob = best_tokenization(word, freq2)
    print(f"Word: {word}, Best Tokenization: {tokenization}, Probability: {prob:.6f}")

# Compute loss for the corpus
loss = compute_loss(corpus, freq2)
print(f"Total Loss: {loss:.2f}")