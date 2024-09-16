import math
from collections import defaultdict

# Tokenize documents into words
def tokenize(doc):
    return doc.lower().split()

# Training the Naive Bayes classifier
def train_naive_bayes(data):
    # Initialize variables
    logprior = {}
    loglikelihood = defaultdict(lambda: defaultdict(float))
    class_word_count = defaultdict(lambda: defaultdict(int))
    class_doc_count = defaultdict(int)
    vocabulary = set()
    Ndoc = len(data)
    
    # Concatenate documents by class
    for label, doc in data:
        class_doc_count[label] += 1
        words = tokenize(doc)
        vocabulary.update(words)
        for word in words:
            class_word_count[label][word] += 1
    
    print(f'\n{class_doc_count=}\n{class_word_count=}\n')
    for k,v in class_word_count.items():
        print(k, sum(v.values()))

    # Calculate log P(c)
    for label in class_doc_count:
        logprior[label] = math.log(class_doc_count[label] / Ndoc)
    
    # Calculate log P(w|c) with Laplace smoothing
    for label in class_doc_count:
        total_word_count = sum(class_word_count[label].values())
        for word in vocabulary:
            word_count = class_word_count[label][word] + 1  # Add-one smoothing
            loglikelihood[word][label] = math.log(word_count / (total_word_count + len(vocabulary)))
            print(f'{word}|{label}: {word_count}/{(total_word_count + len(vocabulary))} = {word_count/(total_word_count + len(vocabulary))}#{math.log(word_count/(total_word_count + len(vocabulary)))}')
    
    return logprior, loglikelihood, vocabulary

# Classify a new document
def classify_naive_bayes(doc, logprior, loglikelihood, classes, vocabulary):
    words = tokenize(doc)
    scores = {label: logprior[label] for label in classes}
    
    for word in words:
        if word in vocabulary:
            for label in classes:
                scores[label] += loglikelihood[word][label]
    
    print(f'{doc}|{scores}')
    keyofmax = max(scores, key=scores.get)
    return keyofmax, scores[keyofmax]

# Sample training data
training_data = [
    ("spam", "Buy cheap products now"),
    ("ham", "Meeting is scheduled at 9"),
    ("ham", "Your appointment is confirmed"),
    ("spam", "Cheap deals available today"),
    ("ham", "Please confirm your attendance"),
    ("spam", "Get cheap tickets now")
]

# Train the Naive Bayes classifier
logprior, loglikelihood, vocabulary = train_naive_bayes(training_data)
print(f'\n{logprior=}\n{loglikelihood=}\n{vocabulary=}\n{len(vocabulary)=}\n')

# Sample classes
classes = ["spam", "ham"]

# Test the classifier with a new email
test_email_1 = "Get cheap products"
test_email_2 = "Your meeting is confirmed"

predicted_class_1 = classify_naive_bayes(test_email_1, logprior, loglikelihood, classes, vocabulary)
predicted_class_2 = classify_naive_bayes(test_email_2, logprior, loglikelihood, classes, vocabulary)

# Output the predictions
print(f"The email '{test_email_1}' is classified as: {predicted_class_1}")
print(f"The email '{test_email_2}' is classified as: {predicted_class_2}")
