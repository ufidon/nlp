import numpy as np
from sklearn.utils import resample

# Original results for Classifier A and B (1: correct, 0: incorrect)
classifier_A = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 1])
classifier_B = np.array([1, 0, 1, 1, 1, 0, 1, 0, 1, 0])

# Observed accuracy difference
observed_diff = np.mean(classifier_A) - np.mean(classifier_B)

# Bootstrap sampling to calculate p-value
n_iterations = 10000
n_size = len(classifier_A)
bootstrap_diffs = []

# Perform bootstrap sampling
for i in range(n_iterations):
    # Resample the test set with replacement
    A_sample = resample(classifier_A, n_samples=n_size)
    B_sample = resample(classifier_B, n_samples=n_size)
    
    # Calculate the accuracy difference for this bootstrap sample
    bootstrap_diff = np.mean(A_sample) - np.mean(B_sample)
    bootstrap_diffs.append(bootstrap_diff)

# Convert to numpy array for further analysis
bootstrap_diffs = np.array(bootstrap_diffs)

# Calculate p-value: Proportion of bootstrap differences >= observed difference
p_value = np.mean(bootstrap_diffs >= observed_diff)

# Calculate the 95% confidence interval for the difference in accuracy
confidence_interval = np.percentile(bootstrap_diffs, [2.5, 97.5])

# Display results
print(f"Observed difference in accuracy: {observed_diff:.3f}")
print(f"p-value: {p_value:.3f}")
print(f"95% Confidence Interval: {confidence_interval[0]:.3f}, {confidence_interval[1]:.3f}")
