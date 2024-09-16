import numpy as np

# Confusion matrix for classes A, B, C
confusion_matrix = np.array([
    [50, 10, 5],  # True Class A
    [8, 45, 12],  # True Class B
    [5, 15, 50]   # True Class C
])

# Summing TP, FP, and FN for each class
true_positives = np.diag(confusion_matrix)
false_positives = np.sum(confusion_matrix, axis=0) - true_positives
false_negatives = np.sum(confusion_matrix, axis=1) - true_positives

print(f'{true_positives=}\n{false_negatives=}\n{false_positives=}\n')
# Precision, Recall, and F1-score for each class
precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f1_score = 2 * (precision * recall) / (precision + recall)

# Macro-Averaged Metrics
macro_precision = np.mean(precision)
macro_recall = np.mean(recall)
macro_f1_score = np.mean(f1_score)

# Micro-Averaged Metrics
total_true_positives = np.sum(true_positives)
total_false_positives = np.sum(false_positives)
total_false_negatives = np.sum(false_negatives)

micro_precision = total_true_positives / (total_true_positives + total_false_positives)
micro_recall = total_true_positives / (total_true_positives + total_false_negatives)
micro_f1_score = micro_precision  # since micro precision and recall are equal

# Printing results
classes = ['A', 'B', 'C']
print("Class-wise Precision, Recall, and F1-Score:")
for i, cls in enumerate(classes):
    print(f"Class {cls}:")
    print(f" Precision: {precision[i]:.4f}")
    print(f" Recall: {recall[i]:.4f}")
    print(f" F1-Score: {f1_score[i]:.4f}")
    print()

print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall: {macro_recall:.4f}")
print(f"Macro F1-Score: {macro_f1_score:.4f}")
print()
print(f"Micro Precision: {micro_precision:.4f}")
print(f"Micro Recall: {micro_recall:.4f}")
print(f"Micro F1-Score: {micro_f1_score:.4f}")