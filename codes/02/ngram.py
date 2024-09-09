from collections import defaultdict, Counter

class NGramModel:
    def __init__(self, n):
        self.n = n
        self.ngrams = defaultdict(Counter)
        self.contexts = defaultdict(int)
    
    def train(self, corpus):
        for sentence in corpus:
            tokens = ["<s>"] * (self.n - 1) + sentence.split() + ["</s>"]
            for i in range(len(tokens) - self.n + 1):
                context = tuple(tokens[i:i+self.n-1])
                word = tokens[i+self.n-1]
                self.ngrams[context][word] += 1
                self.contexts[context] += 1
    
    def probability(self, context, word):
        context = tuple(context)
        count_context = self.contexts[context]
        count_ngram = self.ngrams[context][word]
        return count_ngram / count_context if count_context > 0 else 0
    
    def sentence_probability(self, sentence):
        tokens = ["<s>"] * (self.n - 1) + sentence.split() + ["</s>"]
        prob = 1.0
        for i in range(len(tokens) - self.n + 1):
            context = tokens[i:i+self.n-1]
            word = tokens[i+self.n-1]
            prob *= self.probability(context, word)
        return prob

# Example usage
corpus = [
    "I love natural language processing",
    "I love deep learning",
    "I enjoy coding",
    "I love coding",
    "I love learning"
]

# Train a trigram model
trigram_model = NGramModel(n=3)
trigram_model.train(corpus)
print(f"{trigram_model.contexts=}")
print(f"{trigram_model.ngrams=}")
# Calculate the probability of a sentence
sentence = "I love coding"
print(f"Probability of '{sentence}': {trigram_model.sentence_probability(sentence)}")