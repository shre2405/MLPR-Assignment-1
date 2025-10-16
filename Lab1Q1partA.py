import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
N = 10000
p0 = 0.65
p1 = 0.35

# Class 0 parameters
m0 = np.array([-0.5, -0.5, -0.5])
C0 = np.array([[1, -0.5, 0.3],
               [-0.5, 1, -0.5],
               [0.3, -0.5, 1]])

# Class 1 parameters
m1 = np.array([1, 1, 1])
C1 = np.array([[1, 0.3, -0.2],
               [0.3, 1, 0.3],
               [-0.2, 0.3, 1]])

# Generate data
labels = np.random.rand(N) >= p0
N0 = np.sum(labels == 0)
N1 = np.sum(labels == 1)

print(f"Generated {N0} samples from class 0 and {N1} samples from class 1")

# Generate samples from each class
X0 = np.random.multivariate_normal(m0, C0, N0)
X1 = np.random.multivariate_normal(m1, C1, N1)

# Combine data
X = np.zeros((N, 3))
X[labels == 0] = X0
X[labels == 1] = X1

# Visualize the data
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X0[:, 0], X0[:, 1], X0[:, 2], c='blue', marker='o', alpha=0.5, label='Class 0')
ax.scatter(X1[:, 0], X1[:, 1], X1[:, 2], c='red', marker='^', alpha=0.5, label='Class 1')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')
ax.set_title('Generated 3D Data')
ax.legend()
plt.tight_layout()
plt.savefig('data_visualization.png', dpi=150, bbox_inches='tight')
plt.show()

# Compute likelihood ratios for all samples
pdf_x_given_L0 = multivariate_normal.pdf(X, mean=m0, cov=C0)
pdf_x_given_L1 = multivariate_normal.pdf(X, mean=m1, cov=C1)
likelihood_ratios = pdf_x_given_L1 / pdf_x_given_L0

# Generate range of gamma values (log scale for better coverage)
gamma_values = np.concatenate([
    [0],
    np.logspace(-3, 3, 1000),
    [np.inf]
])

# Compute TPR and FPR for each gamma
TPR = []  # P(D=1|L=1)
FPR = []  # P(D=1|L=0)
P_error = []  # Probability of error

for gamma in gamma_values:
    # Apply decision rule: D=1 if likelihood_ratio > gamma
    decisions = likelihood_ratios > gamma
    
    # Compute TPR: P(D=1|L=1)
    tpr = np.sum(decisions[labels == 1]) / N1 if N1 > 0 else 0
    TPR.append(tpr)
    
    # Compute FPR: P(D=1|L=0)
    fpr = np.sum(decisions[labels == 0]) / N0 if N0 > 0 else 0
    FPR.append(fpr)
    
    # Compute P(error)
    # P(error) = P(D=1|L=0)*P(L=0) + P(D=0|L=1)*P(L=1)
    p_err = fpr * p0 + (1 - tpr) * p1
    P_error.append(p_err)

TPR = np.array(TPR)
FPR = np.array(FPR)
P_error = np.array(P_error)

# Find minimum probability of error
min_error_idx = np.argmin(P_error)
min_error = P_error[min_error_idx]
optimal_gamma = gamma_values[min_error_idx]
optimal_TPR = TPR[min_error_idx]
optimal_FPR = FPR[min_error_idx]

# Theoretical optimal gamma for 0-1 loss
theoretical_gamma = p0 / p1

print(f"\n=== Results ===")
print(f"Theoretical optimal gamma (0-1 loss): {theoretical_gamma:.4f}")
print(f"Empirically optimal gamma: {optimal_gamma:.4f}")
print(f"Minimum P(error): {min_error:.4f}")
print(f"At optimal operating point:")
print(f"  TPR = {optimal_TPR:.4f}")
print(f"  FPR = {optimal_FPR:.4f}")

# Plot ROC curve
plt.figure(figsize=(10, 8))
plt.plot(FPR, TPR, 'b-', linewidth=2, label='ROC Curve')
plt.plot(optimal_FPR, optimal_TPR, 'ro', markersize=12, 
         label=f'Min P(error) = {min_error:.4f}')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random Classifier')
plt.xlabel('False Positive Rate (FPR)', fontsize=12)
plt.ylabel('True Positive Rate (TPR)', fontsize=12)
plt.title('ROC Curve for Minimum Expected Risk Classifier', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150, bbox_inches='tight')
plt.show()

# Plot P(error) vs gamma
plt.figure(figsize=(10, 6))
valid_gamma_idx = np.isfinite(gamma_values)
plt.semilogx(gamma_values[valid_gamma_idx], P_error[valid_gamma_idx], 
             'b-', linewidth=2)
plt.axvline(optimal_gamma, color='r', linestyle='--', linewidth=2,
            label=f'Optimal γ = {optimal_gamma:.4f}')
plt.axvline(theoretical_gamma, color='g', linestyle='--', linewidth=2,
            label=f'Theoretical γ = {theoretical_gamma:.4f}')
plt.xlabel('Threshold γ', fontsize=12)
plt.ylabel('P(error)', fontsize=12)
plt.title('Probability of Error vs Threshold', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('error_vs_gamma.png', dpi=150, bbox_inches='tight')
plt.show()

# Apply optimal classifier and show confusion
decisions_optimal = likelihood_ratios > optimal_gamma
TP = np.sum((decisions_optimal == 1) & (labels == 1))
FP = np.sum((decisions_optimal == 1) & (labels == 0))
TN = np.sum((decisions_optimal == 0) & (labels == 0))
FN = np.sum((decisions_optimal == 0) & (labels == 1))

print(f"\n=== Confusion Matrix (at optimal threshold) ===")
print(f"                Predicted L=0    Predicted L=1")
print(f"Actual L=0      {TN:6d}          {FP:6d}")
print(f"Actual L=1      {FN:6d}          {TP:6d}")
print(f"\nError breakdown:")
print(f"  False Positives (D=1|L=0): {FP} ({FP/N0*100:.2f}% of class 0)")
print(f"  False Negatives (D=0|L=1): {FN} ({FN/N1*100:.2f}% of class 1)")

# Save data for use in other parts
np.savez('q1_data.npz', X=X, labels=labels, 
         likelihood_ratios=likelihood_ratios,
         m0=m0, C0=C0, m1=m1, C1=C1, p0=p0, p1=p1)