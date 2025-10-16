import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Set random seed for reproducibility
np.random.seed(42)

print("="*80)
print("Question 1 Part C: Fisher Linear Discriminant Analysis (LDA) Classifier")
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

# TRUE class parameters (for data generation)
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
labels = u.astype(int)  # Convert to integer array

# ============================================================================
# STEP 2: Estimate parameters from data using sample averages
# ============================================================================
print("\n" + "="*80)
print("STEP 2: Estimating Parameters from Data (Sample Averages)")
print("="*80)

# Separate data by class
X0 = X[labels == 0]  # Class 0 samples
X1 = X[labels == 1]  # Class 1 samples

# Estimate means using sample average (column vectors)
m0_est = np.mean(X0, axis=0)
m1_est = np.mean(X1, axis=0)

# Estimate covariances using sample covariance
C0_est = np.cov(X0.T)
C1_est = np.cov(X1.T)

print("\nEstimated Parameters:")
print("\nClass 0:")
print(f"  True mean:      {m0_true}")
print(f"  Estimated mean: {m0_est}")
print(f"  Estimation error: {np.linalg.norm(m0_est - m0_true):.6f}")
print("\n  True covariance:")
print(C0_true)
print("  Estimated covariance:")
print(C0_est)

print("\nClass 1:")
print(f"  True mean:      {m1_true}")
print(f"  Estimated mean: {m1_est}")
print(f"  Estimation error: {np.linalg.norm(m1_est - m1_true):.6f}")
print("\n  True covariance:")
print(C1_true)
print("  Estimated covariance:")
print(C1_est)

# ============================================================================
# STEP 3: Compute Fisher LDA projection vector
# ============================================================================
print("\n" + "="*80)
print("STEP 3: Computing Fisher LDA Projection Vector")
print("="*80)

# Within-class scatter matrix: S_W = S_0 + S_1
# Note: Using EQUAL weights as specified (not weighted by priors)
S_W = C0_est + C1_est

print("\nWithin-class scatter matrix S_W:")
print(S_W)

# Between-class scatter matrix: S_B = (m1 - m0)(m1 - m0)^T
# Note: Using EQUAL weights as specified
mean_diff = m1_est - m0_est
S_B = np.outer(mean_diff, mean_diff)

print("\nBetween-class scatter matrix S_B:")
print(S_B)
print(f"\nRank of S_B: {np.linalg.matrix_rank(S_B)} (should be 1)")

# Fisher LDA solution: w_LDA = S_W^(-1) * (m1 - m0)
# This is equivalent to solving the generalized eigenvalue problem:
# S_B * w = λ * S_W * w

# Method 1: Direct solution (simpler and more stable for 2-class problem)
w_LDA = np.linalg.solve(S_W, mean_diff)

# Normalize the projection vector
w_LDA = w_LDA / np.linalg.norm(w_LDA)

print("\nFisher LDA projection vector w_LDA:")
print(w_LDA)
print(f"Norm of w_LDA: {np.linalg.norm(w_LDA):.6f}")

# ============================================================================
# STEP 4: Project data onto Fisher LDA direction
# ============================================================================
print("\n" + "="*80)
print("STEP 4: Projecting Data onto Fisher LDA Direction")
print("="*80)

# Project all samples onto w_LDA
y = X @ w_LDA  # y = w_LDA^T * x for all x

# Separate projected values by class
y0 = y[labels == 0]
y1 = y[labels == 1]

print(f"\nProjected data statistics:")
print(f"  Class 0 - Mean: {np.mean(y0):.4f}, Std: {np.std(y0):.4f}")
print(f"  Class 1 - Mean: {np.mean(y1):.4f}, Std: {np.std(y1):.4f}")
print(f"  Separation (difference in means): {np.mean(y1) - np.mean(y0):.4f}")

# ============================================================================
# STEP 5: Vary threshold τ and compute ROC curve
# ============================================================================
print("\n" + "="*80)
print("STEP 5: Computing ROC Curve for Fisher LDA Classifier")
print("="*80)

# Decision rule: Classify as D=1 if w_LDA^T * x > τ
# Create threshold values from -∞ to +∞
tau_values = np.concatenate([
    [-np.inf],
    np.linspace(np.min(y) - 1, np.max(y) + 1, 2000),
    [np.inf]
])

TPR_lda = []
FPR_lda = []
P_error_lda = []

for tau in tau_values:
    # Decision: D=1 if y > tau
    decisions = y > tau
    
    # TPR and FPR
    tpr = np.sum(decisions[labels == 1]) / N1
    fpr = np.sum(decisions[labels == 0]) / N0
    
    TPR_lda.append(tpr)
    FPR_lda.append(fpr)
    
    # P(error)
    p_err = fpr * p0 + (1 - tpr) * p1
    P_error_lda.append(p_err)

TPR_lda = np.array(TPR_lda)
FPR_lda = np.array(FPR_lda)
P_error_lda = np.array(P_error_lda)

# ============================================================================
# STEP 6: Find minimum P(error) for Fisher LDA
# ============================================================================
min_error_idx_lda = np.argmin(P_error_lda)
min_P_error_lda = P_error_lda[min_error_idx_lda]
optimal_tau = tau_values[min_error_idx_lda]
optimal_TPR_lda = TPR_lda[min_error_idx_lda]
optimal_FPR_lda = FPR_lda[min_error_idx_lda]

print(f"\nFisher LDA Results:")
print(f"  Optimal threshold τ = {optimal_tau:.6f}")
print(f"  Minimum P(error) = {min_P_error_lda:.6f}")
print(f"  At optimal operating point:")
print(f"    TPR = {optimal_TPR_lda:.6f}")
print(f"    FPR = {optimal_FPR_lda:.6f}")

# ============================================================================
# STEP 7: Compare with Part A and Part B
# ============================================================================
print("\n" + "="*80)
print("STEP 7: Comparing with Parts A and B")
print("="*80)

# Recompute Part A results (True Model)
pdf_x_given_L0_true = multivariate_normal.pdf(X, mean=m0_true, cov=C0_true)
pdf_x_given_L1_true = multivariate_normal.pdf(X, mean=m1_true, cov=C1_true)
likelihood_ratios_true = pdf_x_given_L1_true / pdf_x_given_L0_true

gamma_values = np.concatenate([[0.0], np.logspace(-4, 4, 2000), [np.inf]])
P_error_true = []
TPR_true = []
FPR_true = []

for gamma in gamma_values:
    decisions = likelihood_ratios_true > gamma
    tpr = np.sum(decisions[labels == 1]) / N1
    fpr = np.sum(decisions[labels == 0]) / N0
    TPR_true.append(tpr)
    FPR_true.append(fpr)
    P_error_true.append(fpr * p0 + (1 - tpr) * p1)

TPR_true = np.array(TPR_true)
FPR_true = np.array(FPR_true)
P_error_true = np.array(P_error_true)
min_P_error_true = np.min(P_error_true)
min_idx_true = np.argmin(P_error_true)
optimal_TPR_true = TPR_true[min_idx_true]
optimal_FPR_true = FPR_true[min_idx_true]

# Recompute Part B results (Naive Bayes)
C0_naive = np.eye(3)
C1_naive = np.eye(3)
pdf_x_given_L0_naive = multivariate_normal.pdf(X, mean=m0_true, cov=C0_naive)
pdf_x_given_L1_naive = multivariate_normal.pdf(X, mean=m1_true, cov=C1_naive)
likelihood_ratios_naive = pdf_x_given_L1_naive / pdf_x_given_L0_naive

P_error_naive = []
TPR_naive = []
FPR_naive = []

for gamma in gamma_values:
    decisions = likelihood_ratios_naive > gamma
    tpr = np.sum(decisions[labels == 1]) / N1
    fpr = np.sum(decisions[labels == 0]) / N0
    TPR_naive.append(tpr)
    FPR_naive.append(fpr)
    P_error_naive.append(fpr * p0 + (1 - tpr) * p1)

TPR_naive = np.array(TPR_naive)
FPR_naive = np.array(FPR_naive)
P_error_naive = np.array(P_error_naive)
min_P_error_naive = np.min(P_error_naive)
min_idx_naive = np.argmin(P_error_naive)
optimal_TPR_naive = TPR_naive[min_idx_naive]
optimal_FPR_naive = FPR_naive[min_idx_naive]

# Comparison table
print(f"\nPerformance Comparison:")
print(f"{'Metric':<25} {'Part A (True)':<18} {'Part B (Naive)':<18} {'Part C (LDA)':<18}")
print(f"{'-'*79}")
print(f"{'Min P(error)':<25} {min_P_error_true:<18.6f} {min_P_error_naive:<18.6f} {min_P_error_lda:<18.6f}")
print(f"{'Optimal TPR':<25} {optimal_TPR_true:<18.6f} {optimal_TPR_naive:<18.6f} {optimal_TPR_lda:<18.6f}")
print(f"{'Optimal FPR':<25} {optimal_FPR_true:<18.6f} {optimal_FPR_naive:<18.6f} {optimal_FPR_lda:<18.6f}")
print(f"{'Accuracy':<25} {(1-min_P_error_true)*100:<18.2f} {(1-min_P_error_naive)*100:<18.2f} {(1-min_P_error_lda)*100:<18.2f}")

print(f"\nPerformance relative to Part A (True Model):")
print(f"  Part A: 0.00% worse (baseline - optimal)")
print(f"  Part B: {((min_P_error_naive/min_P_error_true - 1)*100):.2f}% worse")
print(f"  Part C: {((min_P_error_lda/min_P_error_true - 1)*100):.2f}% worse")

# ============================================================================
# STEP 8: Visualizations - Clean Separate Figures
# ============================================================================
print("\n" + "="*80)
print("STEP 8: Creating Clean Visualizations")
print("="*80)

# FIGURE 1: ROC Curves Comparison
fig1 = plt.figure(figsize=(12, 10))

ax1 = plt.subplot(1, 1, 1)
ax1.plot(FPR_true, TPR_true, 'b-', linewidth=3, label='Part A: True Model (QDA)')
ax1.plot(FPR_naive, TPR_naive, 'r--', linewidth=3, label='Part B: Naive Bayes')
ax1.plot(FPR_lda, TPR_lda, 'g-.', linewidth=3, label='Part C: Fisher LDA')

# Mark optimal points
ax1.plot(optimal_FPR_true, optimal_TPR_true, 'bo', markersize=12, 
         markerfacecolor='blue', markeredgewidth=2, 
         label=f'Part A Optimal (P(error)={min_P_error_true:.4f})')
ax1.plot(optimal_FPR_naive, optimal_TPR_naive, 'ro', markersize=12, 
         markerfacecolor='red', markeredgewidth=2,
         label=f'Part B Optimal (P(error)={min_P_error_naive:.4f})')
ax1.plot(optimal_FPR_lda, optimal_TPR_lda, 'go', markersize=12, 
         markerfacecolor='green', markeredgewidth=2,
         label=f'Part C Optimal (P(error)={min_P_error_lda:.4f})')

ax1.plot([0, 1], [0, 1], 'k--', alpha=0.4, linewidth=2, label='Random Classifier')

ax1.set_xlabel('False Positive Rate: P(D=1|L=0)', fontsize=14, fontweight='bold')
ax1.set_ylabel('True Positive Rate: P(D=1|L=1)', fontsize=14, fontweight='bold')
ax1.set_title('ROC Curves Comparison\nPart A (True) vs Part B (Naive) vs Part C (LDA)', 
              fontsize=16, fontweight='bold', pad=20)
ax1.legend(fontsize=11, loc='lower right', framealpha=0.95)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xlim([-0.02, 1.02])
ax1.set_ylim([-0.02, 1.02])
ax1.set_aspect('equal')

plt.tight_layout()
plt.savefig('Q1_PartC_ROC_Comparison.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: Q1_PartC_ROC_Comparison.png")

# FIGURE 2: Fisher LDA Projection Histogram
fig2 = plt.figure(figsize=(12, 7))

ax2 = plt.subplot(1, 1, 1)
ax2.hist(y0, bins=50, alpha=0.6, color='blue', edgecolor='black', 
         label=f'Class 0 Projected (N={N0})', density=True, linewidth=1.5)
ax2.hist(y1, bins=50, alpha=0.6, color='red', edgecolor='black',
         label=f'Class 1 Projected (N={N1})', density=True, linewidth=1.5)
ax2.axvline(optimal_tau, color='green', linestyle='--', linewidth=3, 
            label=f'Optimal Threshold τ = {optimal_tau:.3f}')
ax2.axvline(np.mean(y0), color='darkblue', linestyle=':', linewidth=2, 
            label=f'Class 0 Mean = {np.mean(y0):.3f}')
ax2.axvline(np.mean(y1), color='darkred', linestyle=':', linewidth=2,
            label=f'Class 1 Mean = {np.mean(y1):.3f}')

ax2.set_xlabel('Projected Value: y = w_LDA^T × x', fontsize=14, fontweight='bold')
ax2.set_ylabel('Probability Density', fontsize=14, fontweight='bold')
ax2.set_title('Fisher LDA: 1D Projection of 3D Data\nClass Separation After Projection', 
              fontsize=16, fontweight='bold', pad=20)
ax2.legend(fontsize=11, loc='best', framealpha=0.95)
ax2.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('Q1_PartC_LDA_Projection.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: Q1_PartC_LDA_Projection.png")

# FIGURE 3: P(error) Comparison
fig3 = plt.figure(figsize=(12, 7))

ax3 = plt.subplot(1, 1, 1)
valid_idx = np.isfinite(gamma_values) & (gamma_values > 0)

ax3.semilogx(gamma_values[valid_idx], P_error_true[valid_idx], 
             'b-', linewidth=3, label='Part A: True Model')
ax3.semilogx(gamma_values[valid_idx], P_error_naive[valid_idx], 
             'r--', linewidth=3, label='Part B: Naive Bayes')
ax3.axhline(min_P_error_lda, color='green', linestyle='-.', linewidth=3, 
            label=f'Part C: Fisher LDA (P(error) = {min_P_error_lda:.4f})')
ax3.axhline(min_P_error_true, color='blue', linestyle=':', linewidth=2, alpha=0.5)
ax3.axhline(min_P_error_naive, color='red', linestyle=':', linewidth=2, alpha=0.5)

ax3.set_xlabel('Threshold γ', fontsize=14, fontweight='bold')
ax3.set_ylabel('Probability of Error', fontsize=14, fontweight='bold')
ax3.set_title('Probability of Error vs Threshold γ\nComparison Across All Three Methods', 
              fontsize=16, fontweight='bold', pad=20)
ax3.legend(fontsize=12, loc='best', framealpha=0.95)
ax3.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('Q1_PartC_Error_Comparison.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: Q1_PartC_Error_Comparison.png")

# FIGURE 4: Performance Summary Bar Chart
fig4 = plt.figure(figsize=(14, 8))

ax4 = plt.subplot(1, 1, 1)
methods = ['Part A\nTrue Model\n(QDA)', 'Part B\nNaive Bayes\n(Independence)', 'Part C\nFisher LDA\n(Linear)']
errors = [min_P_error_true * 100, min_P_error_naive * 100, min_P_error_lda * 100]
accuracies = [(1-min_P_error_true)*100, (1-min_P_error_naive)*100, (1-min_P_error_lda)*100]
colors_bars = ['#3498db', '#e74c3c', '#2ecc71']

bars = ax4.bar(methods, errors, color=colors_bars, alpha=0.8, 
               edgecolor='black', linewidth=2.5, width=0.6)

ax4.set_ylabel('Minimum P(error) (%)', fontsize=14, fontweight='bold')
ax4.set_title('Classification Error Rate Comparison\nQuestion 1: Parts A, B, C', 
              fontsize=16, fontweight='bold', pad=20)
ax4.set_ylim([0, max(errors) * 1.3])
ax4.grid(True, alpha=0.3, axis='y', linestyle='--')

# Add value labels on bars
for i, (bar, error, acc) in enumerate(zip(bars, errors, accuracies)):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2, height + 0.2, 
             f'Error: {error:.2f}%\nAcc: {acc:.2f}%', 
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('Q1_PartC_Performance_Summary.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: Q1_PartC_Performance_Summary.png")

print("\n" + "="*80)
print("All visualizations created successfully!")
print("="*80)
print("\nGenerated 4 separate figures:")
print("  1. Q1_PartC_ROC_Comparison.png - ROC curves for all three methods")
print("  2. Q1_PartC_LDA_Projection.png - 1D projection histogram")
print("  3. Q1_PartC_Error_Comparison.png - P(error) vs threshold")
print("  4. Q1_PartC_Performance_Summary.png - Bar chart summary")

# ============================================================================
# STEP 9: Confusion Matrix for Fisher LDA
# ============================================================================
print("\n" + "="*80)
print("STEP 9: Confusion Matrix for Fisher LDA")
print("="*80)

decisions_lda = y > optimal_tau
TP_lda = np.sum((decisions_lda == 1) & (labels == 1))
FP_lda = np.sum((decisions_lda == 1) & (labels == 0))
TN_lda = np.sum((decisions_lda == 0) & (labels == 0))
FN_lda = np.sum((decisions_lda == 0) & (labels == 1))

print("\nFisher LDA Confusion Matrix:")
print(f"                    Predicted")
print(f"                    D=0      D=1")
print(f"           L=0   {TN_lda:6d}   {FP_lda:6d}")
print(f"  Actual   L=1   {FN_lda:6d}   {TP_lda:6d}")
print(f"  Accuracy: {(TP_lda + TN_lda)/N*100:.2f}%")
print(f"  Total errors: {FP_lda + FN_lda} out of {N} samples")

# ============================================================================
# STEP 10: Final Discussion
# ============================================================================
print("\n" + "="*80)
print("FINAL DISCUSSION: Fisher LDA Performance Analysis")
print("="*80)

print("\n1. How Fisher LDA Works:")
print("   - Projects 3D data onto 1D line (w_LDA direction)")
print("   - Maximizes ratio: (between-class variance) / (within-class variance)")
print("   - Finds optimal linear direction for class separation")
print("   - Uses estimated means and covariances from training data")

print("\n2. Comparison with Part A (True Model):")
if min_P_error_lda < min_P_error_true * 1.01:
    print("   ✓ Fisher LDA performs VERY CLOSE to optimal Bayes classifier!")
    print("   - This suggests data is approximately linearly separable")
elif min_P_error_lda < min_P_error_true * 1.10:
    print("   ✓ Fisher LDA performs WELL, close to optimal")
    print("   - Small performance gap due to linear decision boundary")
else:
    print("   ⚠️ Fisher LDA has noticeable performance gap vs optimal")
    print("   - Linear boundary may not capture optimal quadratic boundary")

print("\n3. Comparison with Part B (Naive Bayes):")
if min_P_error_lda < min_P_error_naive:
    print("   ✓ Fisher LDA OUTPERFORMS Naive Bayes")
    print("   - LDA exploits correlations through covariance estimation")
    print("   - Better than independence assumption")
else:
    print("   ⚠️ Fisher LDA performs worse than Naive Bayes")

print("\n4. Key Advantages of Fisher LDA:")
print("   - Simple: Only requires mean and covariance estimation")
print("   - Efficient: Projects to 1D, fast classification")
print("   - Interpretable: w_LDA shows discriminative direction")
print("   - No distribution assumption: Works beyond Gaussians")

print("\n5. Key Limitations of Fisher LDA:")
print("   - LINEAR decision boundary only")
print("   - Optimal for equal covariances (LDA assumption)")
print("   - When covariances differ (like this problem), suboptimal vs QDA")
print("   - Single projection dimension may lose information")

print("\n6. When True Covariances Are Different:")
print("   - Part A uses TRUE different covariances → Quadratic boundary")
print("   - Fisher LDA assumes COMMON covariance → Linear boundary")
print("   - Performance gap shows value of quadratic boundary")

degradation_lda = ((min_P_error_lda / min_P_error_true - 1) * 100)
print(f"\n7. Overall Assessment:")
print(f"   - Error increase vs optimal: {degradation_lda:.2f}%")
if degradation_lda < 5:
    print("   - Verdict: EXCELLENT performance for a linear classifier")
elif degradation_lda < 15:
    print("   - Verdict: GOOD performance, acceptable trade-off")
else:
    print("   - Verdict: SIGNIFICANT gap, linear boundary insufficient")

print("\n" + "="*80)
print("Analysis Complete!")
print("="*80)