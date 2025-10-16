import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Set random seed for reproducibility
np.random.seed(42)

print("="*80)
print("Naive Bayes Classifier (Model Mismatch Analysis)")
print("="*80)

# ============================================================================
# STEP 1: Load the SAME data from Part A
# ============================================================================
print("\n" + "="*80)
print("STEP 1: Using Same Data as Part A")
print("="*80)

N = 10000
p0 = 0.65
p1 = 0.35

# Generate labels (same seed as Part A)
u = np.random.rand(N) >= p0
N0 = np.sum(u == 0)
N1 = np.sum(u == 1)

print(f"\nData Statistics:")
print(f"  Total samples: {N}")
print(f"  Class 0 samples: {N0} ({N0/N*100:.2f}%)")
print(f"  Class 1 samples: {N1} ({N1/N*100:.2f}%)")

# TRUE class parameters
m0_true = np.array([-0.5, -0.5, -0.5])
C0_true = np.array([[1, -0.5, 0.3],
                    [-0.5, 1, -0.5],
                    [0.3, -0.5, 1]])

m1_true = np.array([1, 1, 1])
C1_true = np.array([[1, 0.3, -0.2],
                    [0.3, 1, 0.3],
                    [-0.2, 0.3, 1]])

# Generate samples
r0 = np.random.multivariate_normal(m0_true, C0_true, N0)
r1 = np.random.multivariate_normal(m1_true, C1_true, N1)

# Combine into single dataset
X = np.zeros((N, 3))
X[u == 0] = r0
X[u == 1] = r1
labels = u.astype(int)

# ============================================================================
# STEP 2: Define NAIVE BAYES model (incorrect assumptions)
# ============================================================================
print("\n" + "="*80)
print("STEP 2: Naive Bayes Model Definition (Model Mismatch)")
print("="*80)

m0_naive = m0_true.copy()  # Use TRUE means
m1_naive = m1_true.copy()  # Use TRUE means

# INCORRECT covariance: Identity matrix (assumes independence)
C0_naive = np.eye(3)
C1_naive = np.eye(3)

print("\nModel Assumptions:")
print("\nTRUE Model (Part A):")
print("  Class 0 mean:", m0_true)
print("  Class 0 covariance:\n", C0_true)
print("\n  Class 1 mean:", m1_true)
print("  Class 1 covariance:\n", C1_true)

print("\nNAIVE BAYES Model (Part B - INCORRECT):")
print("  Class 0 mean:", m0_naive, "(SAME - CORRECT)")
print("  Class 0 covariance:\n", C0_naive, "(IDENTITY - INCORRECT!)")
print("\n  Class 1 mean:", m1_naive, "(SAME - CORRECT)")
print("  Class 1 covariance:\n", C1_naive, "(IDENTITY - INCORRECT!)")

print("\n*** MODEL MISMATCH ***")
print("The Naive Bayes model assumes features are INDEPENDENT,")
print("but the true data has CORRELATED features!")

# ============================================================================
# STEP 3: Compute likelihood ratios using NAIVE model
# ============================================================================
print("\n" + "="*80)
print("STEP 3: Computing Likelihood Ratios with Naive Bayes Model")
print("="*80)

# Compute PDFs using NAIVE Bayes model
pdf_x_given_L0_naive = multivariate_normal.pdf(X, mean=m0_naive, cov=C0_naive)
pdf_x_given_L1_naive = multivariate_normal.pdf(X, mean=m1_naive, cov=C1_naive)

# Compute likelihood ratios
likelihood_ratios_naive = pdf_x_given_L1_naive / pdf_x_given_L0_naive

print(f"\nLikelihood ratio statistics (Naive Bayes):")
print(f"  Min LR = {np.min(likelihood_ratios_naive):.6f}")
print(f"  Max LR = {np.max(likelihood_ratios_naive):.6f}")
print(f"  Mean LR = {np.mean(likelihood_ratios_naive):.6f}")
print(f"  Median LR = {np.median(likelihood_ratios_naive):.6f}")

# ============================================================================
# STEP 4: Vary gamma and compute ROC curve for Naive Bayes
# ============================================================================
print("\n" + "="*80)
print("STEP 4: Computing ROC Curve for Naive Bayes Classifier")
print("="*80)

# Create gamma values
gamma_values = np.concatenate([
    [0.0],
    np.logspace(-4, 4, 2000),
    [np.inf]
])

TPR_naive = []
FPR_naive = []
P_error_naive = []

for gamma in gamma_values:
    # Decision rule: D=1 if likelihood_ratio > gamma
    decisions = likelihood_ratios_naive > gamma
    
    # TPR and FPR
    tpr = np.sum(decisions[labels == 1]) / N1
    fpr = np.sum(decisions[labels == 0]) / N0
    
    TPR_naive.append(tpr)
    FPR_naive.append(fpr)
    
    # P(error)
    p_err = fpr * p0 + (1 - tpr) * p1
    P_error_naive.append(p_err)

TPR_naive = np.array(TPR_naive)
FPR_naive = np.array(FPR_naive)
P_error_naive = np.array(P_error_naive)

# ============================================================================
# STEP 5: Find minimum P(error) for Naive Bayes
# ============================================================================
min_error_idx_naive = np.argmin(P_error_naive)
min_P_error_naive = P_error_naive[min_error_idx_naive]
optimal_gamma_naive = gamma_values[min_error_idx_naive]
optimal_TPR_naive = TPR_naive[min_error_idx_naive]
optimal_FPR_naive = FPR_naive[min_error_idx_naive]

theoretical_gamma = p0 / p1

print(f"\nNaive Bayes Results:")
print(f"  Theoretical optimal γ = {theoretical_gamma:.6f}")
print(f"  Empirical optimal γ = {optimal_gamma_naive:.6f}")
print(f"  Minimum P(error) = {min_P_error_naive:.6f}")
print(f"  At optimal operating point:")
print(f"    TPR = {optimal_TPR_naive:.6f}")
print(f"    FPR = {optimal_FPR_naive:.6f}")

# ============================================================================
# STEP 6: Compute Part A results for comparison
# ============================================================================
print("\n" + "="*80)
print("STEP 6: Comparing with Part A (True Model)")
print("="*80)

# Compute PDFs using TRUE model
pdf_x_given_L0_true = multivariate_normal.pdf(X, mean=m0_true, cov=C0_true)
pdf_x_given_L1_true = multivariate_normal.pdf(X, mean=m1_true, cov=C1_true)

# Compute likelihood ratios for true model
likelihood_ratios_true = pdf_x_given_L1_true / pdf_x_given_L0_true

TPR_true = []
FPR_true = []
P_error_true = []

for gamma in gamma_values:
    decisions = likelihood_ratios_true > gamma
    
    tpr = np.sum(decisions[labels == 1]) / N1
    fpr = np.sum(decisions[labels == 0]) / N0
    
    TPR_true.append(tpr)
    FPR_true.append(fpr)
    
    p_err = fpr * p0 + (1 - tpr) * p1
    P_error_true.append(p_err)

TPR_true = np.array(TPR_true)
FPR_true = np.array(FPR_true)
P_error_true = np.array(P_error_true)

min_P_error_true = np.min(P_error_true)
min_error_idx_true = np.argmin(P_error_true)
optimal_TPR_true = TPR_true[min_error_idx_true]
optimal_FPR_true = FPR_true[min_error_idx_true]

print(f"\nComparison:")
print(f"{'Metric':<30} {'Part A (True)':<20} {'Part B (Naive)':<20} {'Difference':<15}")
print(f"{'-'*85}")
print(f"{'Min P(error)':<30} {min_P_error_true:<20.6f} {min_P_error_naive:<20.6f} {min_P_error_naive - min_P_error_true:<15.6f}")
print(f"{'Optimal TPR':<30} {optimal_TPR_true:<20.6f} {optimal_TPR_naive:<20.6f} {optimal_TPR_naive - optimal_TPR_true:<15.6f}")
print(f"{'Optimal FPR':<30} {optimal_FPR_true:<20.6f} {optimal_FPR_naive:<20.6f} {optimal_FPR_naive - optimal_FPR_true:<15.6f}")

# Calculate performance degradation
degradation_percent = ((min_P_error_naive - min_P_error_true) / min_P_error_true) * 100
print(f"\n*** Performance Impact of Model Mismatch ***")
print(f"  Error rate increased by {degradation_percent:.2f}%")
print(f"  Absolute increase: {(min_P_error_naive - min_P_error_true)*100:.2f} percentage points")

if min_P_error_naive > min_P_error_true + 0.001:  # Small tolerance
    print(f"  ⚠️  Model mismatch NEGATIVELY impacted performance!")
else:
    print(f"  ✓ Surprisingly, model mismatch did not hurt performance significantly.")

# ============================================================================
# STEP 7: Plot comparison of ROC curves
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# ROC Curves
ax1.plot(FPR_true, TPR_true, 'b-', linewidth=2.5, label='Part A: True Model')
ax1.plot(FPR_naive, TPR_naive, 'r--', linewidth=2.5, label='Part B: Naive Bayes')
ax1.plot(optimal_FPR_true, optimal_TPR_true, 'bo', markersize=12, 
         markerfacecolor='blue', label=f'Part A Min P(error)={min_P_error_true:.4f}')
ax1.plot(optimal_FPR_naive, optimal_TPR_naive, 'ro', markersize=12, 
         markerfacecolor='red', label=f'Part B Min P(error)={min_P_error_naive:.4f}')
ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random Classifier')
ax1.set_xlabel('False Positive Rate: P(D=1|L=0)', fontsize=12, fontweight='bold')
ax1.set_ylabel('True Positive Rate: P(D=1|L=1)', fontsize=12, fontweight='bold')
ax1.set_title('ROC Curve Comparison:\nTrue Model vs Naive Bayes', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)
ax1.set_xlim([-0.02, 1.02])
ax1.set_ylim([-0.02, 1.02])
ax1.set_aspect('equal')

# P(error) vs gamma
valid_idx = np.isfinite(gamma_values) & (gamma_values > 0)
ax2.semilogx(gamma_values[valid_idx], P_error_true[valid_idx], 
             'b-', linewidth=2.5, label='Part A: True Model')
ax2.semilogx(gamma_values[valid_idx], P_error_naive[valid_idx], 
             'r--', linewidth=2.5, label='Part B: Naive Bayes')
ax2.axhline(min_P_error_true, color='blue', linestyle=':', alpha=0.5)
ax2.axhline(min_P_error_naive, color='red', linestyle=':', alpha=0.5)
ax2.set_xlabel('Threshold γ', fontsize=12, fontweight='bold')
ax2.set_ylabel('Probability of Error', fontsize=12, fontweight='bold')
ax2.set_title('P(error) vs Threshold γ:\nTrue Model vs Naive Bayes', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)

plt.tight_layout()
plt.savefig('Q1_PartB_Comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# STEP 8: Confusion matrices comparison
# ============================================================================
print("\n" + "="*80)
print("STEP 8: Confusion Matrices at Optimal Thresholds")
print("="*80)

# Part A confusion matrix
decisions_true = likelihood_ratios_true > gamma_values[min_error_idx_true]
TP_true = np.sum((decisions_true == 1) & (labels == 1))
FP_true = np.sum((decisions_true == 1) & (labels == 0))
TN_true = np.sum((decisions_true == 0) & (labels == 0))
FN_true = np.sum((decisions_true == 0) & (labels == 1))

# Part B confusion matrix
decisions_naive = likelihood_ratios_naive > optimal_gamma_naive
TP_naive = np.sum((decisions_naive == 1) & (labels == 1))
FP_naive = np.sum((decisions_naive == 1) & (labels == 0))
TN_naive = np.sum((decisions_naive == 0) & (labels == 0))
FN_naive = np.sum((decisions_naive == 0) & (labels == 1))

print("\nPart A (True Model) Confusion Matrix:")
print(f"                    Predicted")
print(f"                    D=0      D=1")
print(f"           L=0   {TN_true:6d}   {FP_true:6d}")
print(f"  Actual   L=1   {FN_true:6d}   {TP_true:6d}")
print(f"  Accuracy: {(TP_true + TN_true)/N*100:.2f}%")

print("\nPart B (Naive Bayes) Confusion Matrix:")
print(f"                    Predicted")
print(f"                    D=0      D=1")
print(f"           L=0   {TN_naive:6d}   {FP_naive:6d}")
print(f"  Actual   L=1   {FN_naive:6d}   {TP_naive:6d}")
print(f"  Accuracy: {(TP_naive + TN_naive)/N*100:.2f}%")

print("\n" + "="*80)
print("FINAL ANALYSIS: Impact of Model Mismatch")
print("="*80)
print(f"\n1. ROC Curve Impact:")
if min_P_error_naive > min_P_error_true + 0.001:
    print(f"   - Naive Bayes ROC curve is WORSE than the true model")
    print(f"   - The model mismatch degraded classifier performance")
else:
    print(f"   - Naive Bayes ROC curve is comparable to the true model")

print(f"\n2. Minimum P(error) Impact:")
print(f"   - True Model: {min_P_error_true:.4f}")
print(f"   - Naive Bayes: {min_P_error_naive:.4f}")
print(f"   - Increase: {(min_P_error_naive - min_P_error_true)*100:.2f} percentage points")

print(f"\n3. Why Model Mismatch Matters:")
print(f"   - True data has CORRELATED features (off-diagonal covariance terms)")
print(f"   - Naive Bayes IGNORES these correlations (assumes independence)")
print(f"   - This incorrect assumption leads to suboptimal decision boundaries")
print(f"   - The classifier cannot fully exploit the correlation structure")

print(f"\n4. Conclusion:")
if degradation_percent > 5:
    print(f"   ⚠️  SIGNIFICANT negative impact ({degradation_percent:.1f}% worse)")
    print(f"   The independence assumption is strongly violated by the data.")
elif degradation_percent > 1:
    print(f"   ⚠️  MODERATE negative impact ({degradation_percent:.1f}% worse)")
    print(f"   The model mismatch noticeably degraded performance.")
else:
    print(f"   ✓ MINIMAL impact ({degradation_percent:.1f}% worse)")
    print(f"   The independence assumption is approximately valid for this data.")

print("\n" + "="*80)