import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

"""
Multi-Class Gaussian Mixture Classification
Best Hybrid Implementation with Step-by-Step Visualization
"""

# Set random seed for reproducibility
np.random.seed(42)

print("="*80)
print("QUESTION 2: 4-CLASS GAUSSIAN MIXTURE CLASSIFICATION")
print("="*80)

# ============================================================================
# CONFIGURATION AND DATA GENERATION
# ============================================================================

N = 10000  # Number of samples
num_classes = 4
P_L = np.array([0.25, 0.25, 0.25, 0.25])  # Equal class priors

# Define 4 distinct Gaussian class-conditional pdfs
# Class 4 is positioned centrally to maximize overlap with other classes
m = {
    1: np.array([-3.0, -3.0]),  # Top-left
    2: np.array([3.0, -3.0]),   # Top-right
    3: np.array([0.0, 4.0]),    # Bottom
    4: np.array([0.0, 0.0])     # Center - overlapping with others
}

C = {
    1: np.array([[1.5, 0.5], [0.5, 1.0]]),
    2: np.array([[1.0, -0.3], [-0.3, 1.5]]),
    3: np.array([[2.0, 0.0], [0.0, 0.5]]),
    4: np.array([[0.5, 0.1], [0.1, 0.5]])  # Narrow covariance -> tight cluster
}

# Create multivariate Gaussian objects for efficient PDF evaluation
gaussians = {j: multivariate_normal(mean=m[j], cov=C[j]) 
             for j in range(1, num_classes + 1)}

print("\nClass Parameters:")
for j in range(1, num_classes + 1):
    print(f"  Class {j}: mean = {m[j]}, prior = {P_L[j-1]}")

# Generate samples (VECTORIZED - much faster than loop-based approach)
print(f"\nGenerating {N} samples...")
L_true = np.random.choice(np.arange(1, num_classes + 1), size=N, p=P_L)
X = np.zeros((N, 2))

for j in range(1, num_classes + 1):
    idx = np.where(L_true == j)[0]
    if idx.size > 0:
        X[idx, :] = gaussians[j].rvs(size=idx.size)

class_counts = np.array([np.sum(L_true == j) for j in range(1, num_classes + 1)])
print(f"Sample generation complete!")
print(f"  Class distribution: {class_counts}")

# ============================================================================
# PART A: MAP CLASSIFICATION (Minimum Probability of Error)
# ============================================================================

print("\n" + "="*80)
print("PART A: MAP CLASSIFICATION (0-1 LOSS)")
print("="*80)

D_map = np.zeros(N, dtype=int)
is_correct_map = np.zeros(N, dtype=bool)

print("\nClassifying samples using MAP rule...")
for i in range(N):
    x = X[i, :]
    # Compute p(x|L=j) * P(L=j) for all classes
    post_prop = np.array([gaussians[j].pdf(x) * P_L[j-1] 
                          for j in range(1, num_classes + 1)])
    # MAP decision: argmax of posterior
    D_map[i] = np.argmax(post_prop) + 1
    is_correct_map[i] = (D_map[i] == L_true[i])

# Confusion matrix
conf_counts_map = confusion_matrix(L_true, D_map, labels=[1, 2, 3, 4])
# Normalize to get P(D=i|L=j): divide each row by class count
conf_prob_map = conf_counts_map.astype(float) / class_counts[:, None]

# Calculate probability of error
num_correct_map = np.trace(conf_counts_map)
P_error_map = 1.0 - (num_correct_map / N)

print("\nMAP Classification Results:")
print(f"  Correct classifications: {num_correct_map}/{N}")
print(f"  Error probability: {P_error_map:.6f} ({P_error_map*100:.4f}%)")

print("\nConfusion Matrix (Counts):")
print("Rows: True Label L, Columns: Decision D")
print(conf_counts_map)

print("\nConfusion Matrix P(D=i|L=j) (Normalized):")
print("Rows: True Label L, Columns: Decision D")
print(conf_prob_map)

# ============================================================================
# PART B: ERM CLASSIFICATION WITH CUSTOM LOSS MATRIX
# ============================================================================

print("\n" + "="*80)
print("PART B: ERM CLASSIFICATION WITH CUSTOM LOSS MATRIX")
print("="*80)

# Loss matrix Lambda (rows=decision D, columns=true label L)
Lambda = np.array([
    [0,   10,  10,  100],
    [1,   0,   10,  100],
    [1,   1,   0,   100],
    [1,   1,   1,   0]
], dtype=float)

print("\nLoss Matrix Λ (rows=Decision D, cols=True Label L):")
print(Lambda)

D_erm = np.zeros(N, dtype=int)
is_correct_erm = np.zeros(N, dtype=bool)

print("\nClassifying samples using ERM rule...")
# For each sample, compute conditional risk R(x|D=i) = Σ_j λ_ij * p(x|L=j) * P(L=j)
for i in range(N):
    x = X[i, :]
    # Compute posterior proportional values
    post_prop = np.array([gaussians[j].pdf(x) * P_L[j-1] 
                          for j in range(1, num_classes + 1)])
    # Compute risk for each decision using matrix multiplication
    risks = Lambda @ post_prop  # Efficient vectorized computation
    # ERM decision: argmin of risk
    D_erm[i] = np.argmin(risks) + 1
    is_correct_erm[i] = (D_erm[i] == L_true[i])

# Calculate empirical expected risk
total_risk = 0.0
for i in range(N):
    total_risk += Lambda[D_erm[i]-1, L_true[i]-1]
emp_expected_risk = total_risk / N

# Confusion matrix for ERM
conf_counts_erm = confusion_matrix(L_true, D_erm, labels=[1, 2, 3, 4])
conf_prob_erm = conf_counts_erm.astype(float) / class_counts[:, None]

print("\nERM Classification Results:")
print(f"  Empirical expected risk: {emp_expected_risk:.6f}")

print("\nConfusion Matrix (Counts):")
print("Rows: True Label L, Columns: Decision D")
print(conf_counts_erm)

print("\nConfusion Matrix P(D=i|L=j) (Normalized):")
print("Rows: True Label L, Columns: Decision D")
print(conf_prob_erm)

# ============================================================================
# COMPARATIVE ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("COMPARATIVE ANALYSIS")
print("="*80)

print("\nDecision Distribution:")
print(f"  MAP: {np.bincount(D_map)[1:]}")
print(f"  ERM: {np.bincount(D_erm)[1:]}")

print("\nKey Observations:")
print(f"  1. MAP P(error) = {P_error_map:.4f}")
print(f"  2. ERM Expected Risk = {emp_expected_risk:.4f}")
print(f"  3. Class 4 detection rate:")
print(f"     MAP: {conf_prob_map[3, 3]:.4f}")
print(f"     ERM: {conf_prob_erm[3, 3]:.4f}")
print(f"  4. ERM heavily favors Class 4 to avoid 100-unit penalty")

# ============================================================================
# VISUALIZATION - STEP BY STEP
# ============================================================================

print("\n" + "="*80)
print("GENERATING VISUALIZATIONS (Step-by-Step)")
print("="*80)

# Define marker shapes for each class
markers = {1: 'o', 2: 's', 3: '^', 4: 'D'}
from matplotlib.lines import Line2D

# ============================================================================
# FIGURE 1: DATA WITH TRUE LABELS
# ============================================================================

fig1 = plt.figure(figsize=(12, 10))
ax1 = fig1.add_subplot(111)

for j in range(1, num_classes + 1):
    idx = (L_true == j)
    ax1.scatter(X[idx, 0], X[idx, 1], marker=markers[j], s=35, 
               alpha=0.6, label=f'Class {j}', edgecolors='black', linewidths=0.5)

ax1.set_xlabel('$x_1$', fontsize=14, fontweight='bold')
ax1.set_ylabel('$x_2$', fontsize=14, fontweight='bold')
ax1.set_title('Figure 1: Data with True Class Labels', fontsize=16, fontweight='bold', pad=20)
ax1.legend(loc='best', fontsize=12, framealpha=0.9)
ax1.grid(True, alpha=0.3)
ax1.axis('equal')

plt.tight_layout()
plt.savefig('q2_fig1_data_true_labels.png', dpi=200, bbox_inches='tight')
print("\nSaved: q2_fig1_data_true_labels.png")
plt.show()

# ============================================================================
# FIGURE 2: CLASS MEAN POSITIONS
# ============================================================================

fig2 = plt.figure(figsize=(12, 10))
ax2 = fig2.add_subplot(111)

for j in range(1, num_classes + 1):
    ax2.scatter(m[j][0], m[j][1], marker=markers[j], s=500, 
               label=f'Class {j}: μ={m[j]}', alpha=0.7, edgecolors='black', linewidths=3)
    ax2.text(m[j][0], m[j][1], f'{j}', ha='center', va='center', 
            fontsize=18, fontweight='bold', color='white')

ax2.set_xlabel('$x_1$', fontsize=14, fontweight='bold')
ax2.set_ylabel('$x_2$', fontsize=14, fontweight='bold')
ax2.set_title('Figure 2: Class Mean Positions\n(Class 4 is central for maximum overlap)', 
             fontsize=16, fontweight='bold', pad=20)
ax2.legend(fontsize=11, loc='best', framealpha=0.9)
ax2.grid(True, alpha=0.3)
ax2.axis('equal')
ax2.set_xlim(-5, 5)
ax2.set_ylim(-5, 6)

plt.tight_layout()
plt.savefig('q2_fig2_class_means.png', dpi=200, bbox_inches='tight')
print("Saved: q2_fig2_class_means.png")
plt.show()

# ============================================================================
# FIGURE 3: MAP CLASSIFICATION RESULTS
# ============================================================================

fig3 = plt.figure(figsize=(12, 10))
ax3 = fig3.add_subplot(111)

for j in range(1, num_classes + 1):
    # Correctly classified
    idx_correct = (L_true == j) & is_correct_map
    ax3.scatter(X[idx_correct, 0], X[idx_correct, 1], marker=markers[j], 
               s=35, c='green', alpha=0.6, edgecolors='darkgreen', linewidths=0.5)
    # Incorrectly classified
    idx_incorrect = (L_true == j) & ~is_correct_map
    ax3.scatter(X[idx_incorrect, 0], X[idx_incorrect, 1], marker=markers[j], 
               s=35, c='red', alpha=0.8, edgecolors='darkred', linewidths=0.5)

legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
           markersize=10, label='Correct', markeredgecolor='darkgreen', markeredgewidth=2),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
           markersize=10, label='Incorrect', markeredgecolor='darkred', markeredgewidth=2)
]

ax3.set_xlabel('$x_1$', fontsize=14, fontweight='bold')
ax3.set_ylabel('$x_2$', fontsize=14, fontweight='bold')
ax3.set_title(f'Figure 3: Part A - MAP Classification Results\nP(error) = {P_error_map:.4f} ({P_error_map*100:.2f}%)', 
             fontsize=16, fontweight='bold', pad=20)
ax3.legend(handles=legend_elements, loc='best', fontsize=12, framealpha=0.9)
ax3.grid(True, alpha=0.3)
ax3.axis('equal')

plt.tight_layout()
plt.savefig('q2_fig3_map_results.png', dpi=200, bbox_inches='tight')
print("Saved: q2_fig3_map_results.png")
plt.show()

# ============================================================================
# FIGURE 4: MAP CONFUSION MATRIX
# ============================================================================

fig4 = plt.figure(figsize=(10, 8))
ax4 = fig4.add_subplot(111)

sns.heatmap(conf_prob_map, annot=True, fmt='.4f', cmap='Blues', 
            xticklabels=[1, 2, 3, 4], yticklabels=[1, 2, 3, 4],
            ax=ax4, cbar_kws={'label': 'Probability P(D|L)'}, 
            annot_kws={'fontsize': 11}, linewidths=1, linecolor='gray')

ax4.set_xlabel('Decision D', fontsize=13, fontweight='bold', labelpad=12)
ax4.set_ylabel('True Label L', fontsize=13, fontweight='bold', labelpad=12)
ax4.set_title('Figure 4: MAP Confusion Matrix P(D=i|L=j)', 
             fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('q2_fig4_map_confusion.png', dpi=200, bbox_inches='tight')
print("Saved: q2_fig4_map_confusion.png")
plt.show()

# ============================================================================
# FIGURE 5: LOSS MATRIX
# ============================================================================

fig5 = plt.figure(figsize=(10, 8))
ax5 = fig5.add_subplot(111)

sns.heatmap(Lambda, annot=True, fmt='.0f', cmap='YlOrRd', 
            xticklabels=[1, 2, 3, 4], yticklabels=[1, 2, 3, 4],
            ax=ax5, cbar_kws={'label': 'Loss Value'}, 
            annot_kws={'fontsize': 12, 'fontweight': 'bold'}, 
            linewidths=1, linecolor='black')

ax5.set_xlabel('True Label L', fontsize=13, fontweight='bold', labelpad=12)
ax5.set_ylabel('Decision D', fontsize=13, fontweight='bold', labelpad=12)
ax5.set_title('Figure 5: Loss Matrix Λ\n(λ_i4 = 100: High penalty for missing Class 4)', 
             fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('q2_fig5_loss_matrix.png', dpi=200, bbox_inches='tight')
print("Saved: q2_fig5_loss_matrix.png")
plt.show()

# ============================================================================
# FIGURE 6: ERM CLASSIFICATION RESULTS
# ============================================================================

fig6 = plt.figure(figsize=(12, 10))
ax6 = fig6.add_subplot(111)

for j in range(1, num_classes + 1):
    # Correctly classified
    idx_correct = (L_true == j) & is_correct_erm
    ax6.scatter(X[idx_correct, 0], X[idx_correct, 1], marker=markers[j], 
               s=35, c='green', alpha=0.6, edgecolors='darkgreen', linewidths=0.5)
    # Incorrectly classified
    idx_incorrect = (L_true == j) & ~is_correct_erm
    ax6.scatter(X[idx_incorrect, 0], X[idx_incorrect, 1], marker=markers[j], 
               s=35, c='red', alpha=0.8, edgecolors='darkred', linewidths=0.5)

ax6.set_xlabel('$x_1$', fontsize=14, fontweight='bold')
ax6.set_ylabel('$x_2$', fontsize=14, fontweight='bold')
ax6.set_title(f'Figure 6: Part B - ERM Classification Results\nExpected Risk = {emp_expected_risk:.4f}', 
             fontsize=16, fontweight='bold', pad=20)
ax6.legend(handles=legend_elements, loc='best', fontsize=12, framealpha=0.9)
ax6.grid(True, alpha=0.3)
ax6.axis('equal')

plt.tight_layout()
plt.savefig('q2_fig6_erm_results.png', dpi=200, bbox_inches='tight')
print("Saved: q2_fig6_erm_results.png")
plt.show()

# ============================================================================
# FIGURE 7: ERM CONFUSION MATRIX
# ============================================================================

fig7 = plt.figure(figsize=(10, 8))
ax7 = fig7.add_subplot(111)

sns.heatmap(conf_prob_erm, annot=True, fmt='.4f', cmap='Reds', 
            xticklabels=[1, 2, 3, 4], yticklabels=[1, 2, 3, 4],
            ax=ax7, cbar_kws={'label': 'Probability P(D|L)'}, 
            annot_kws={'fontsize': 11}, linewidths=1, linecolor='gray')

ax7.set_xlabel('Decision D', fontsize=13, fontweight='bold', labelpad=12)
ax7.set_ylabel('True Label L', fontsize=13, fontweight='bold', labelpad=12)
ax7.set_title('Figure 7: ERM Confusion Matrix P(D=i|L=j)', 
             fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('q2_fig7_erm_confusion.png', dpi=200, bbox_inches='tight')
print("Saved: q2_fig7_erm_confusion.png")
plt.show()

# ============================================================================
# FIGURE 8: DECISION DISTRIBUTION COMPARISON
# ============================================================================

fig8 = plt.figure(figsize=(12, 8))
ax8 = fig8.add_subplot(111)

x_pos = np.arange(1, num_classes + 1)
width = 0.35
map_counts = np.bincount(D_map)[1:]
erm_counts = np.bincount(D_erm)[1:]

bars1 = ax8.bar(x_pos - width/2, map_counts, width, label='MAP', 
               alpha=0.85, color='steelblue', edgecolor='navy', linewidth=2)
bars2 = ax8.bar(x_pos + width/2, erm_counts, width, label='ERM', 
               alpha=0.85, color='coral', edgecolor='darkred', linewidth=2)

ax8.set_xlabel('Decision Class', fontsize=13, fontweight='bold', labelpad=12)
ax8.set_ylabel('Number of Samples', fontsize=13, fontweight='bold', labelpad=12)
ax8.set_title('Figure 8: Decision Distribution Comparison (MAP vs ERM)', 
             fontsize=16, fontweight='bold', pad=20)
ax8.set_xticks(x_pos)
ax8.legend(fontsize=12, loc='best', framealpha=0.9)
ax8.grid(True, axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('q2_fig8_decision_distribution.png', dpi=200, bbox_inches='tight')
print("Saved: q2_fig8_decision_distribution.png")
plt.show()

# ============================================================================
# FIGURE 9: PER-CLASS ACCURACY COMPARISON
# ============================================================================

fig9 = plt.figure(figsize=(12, 8))
ax9 = fig9.add_subplot(111)

class_labels = [1, 2, 3, 4]
map_accuracy = np.diag(conf_prob_map)
erm_accuracy = np.diag(conf_prob_erm)

x_pos = np.arange(len(class_labels))
width = 0.35

bars1 = ax9.bar(x_pos - width/2, map_accuracy, width, label='MAP', 
               alpha=0.85, color='steelblue', edgecolor='navy', linewidth=2)
bars2 = ax9.bar(x_pos + width/2, erm_accuracy, width, label='ERM', 
               alpha=0.85, color='coral', edgecolor='darkred', linewidth=2)

ax9.set_xlabel('Class', fontsize=13, fontweight='bold', labelpad=12)
ax9.set_ylabel('Accuracy P(D=j|L=j)', fontsize=13, fontweight='bold', labelpad=12)
ax9.set_title('Figure 9: Per-Class Classification Accuracy (MAP vs ERM)', 
             fontsize=16, fontweight='bold', pad=20)
ax9.set_xticks(x_pos)
ax9.set_xticklabels(class_labels)
ax9.set_ylim([0, 1.1])
ax9.legend(fontsize=12, loc='best', framealpha=0.9)
ax9.grid(True, axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax9.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('q2_fig9_per_class_accuracy.png', dpi=200, bbox_inches='tight')
print("Saved: q2_fig9_per_class_accuracy.png")
plt.show()

# ============================================================================
# FIGURE 10: PERFORMANCE METRICS COMPARISON
# ============================================================================

fig10 = plt.figure(figsize=(10, 8))
ax10 = fig10.add_subplot(111)

metrics = ['P(error)\n(MAP)', 'Expected Risk\n(ERM)']
values = [P_error_map, emp_expected_risk]
colors_bar = ['steelblue', 'coral']

bars = ax10.bar(metrics, values, alpha=0.85, color=colors_bar, 
               edgecolor='black', linewidth=2, width=0.5)

ax10.set_ylabel('Risk/Error Value', fontsize=13, fontweight='bold', labelpad=12)
ax10.set_title('Figure 10: Performance Metrics Comparison\n(Different objectives, different metrics)', 
              fontsize=16, fontweight='bold', pad=20)
ax10.grid(True, axis='y', alpha=0.3)
ax10.set_ylim([0, max(values) * 1.3])

for bar, val in zip(bars, values):
    ax10.text(bar.get_x() + bar.get_width()/2., val + max(values)*0.05,
            f'{val:.4f}', ha='center', va='bottom', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('q2_fig10_metrics_comparison.png', dpi=200, bbox_inches='tight')
print("Saved: q2_fig10_metrics_comparison.png")
plt.show()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print("\n1. DATA GENERATION:")
print(f"   • Generated {N} samples from 4-class Gaussian mixture")
print(f"   • Equal priors (0.25 each)")
print(f"   • Class 4 positioned centrally for maximum overlap")

print("\n2. PART A - MAP CLASSIFICATION:")
print(f"   • Decision rule: D(x) = arg max_j [p(x|L=j) P(L=j)]")
print(f"   • Minimizes probability of error (0-1 loss)")
print(f"   • P(error) = {P_error_map:.4f} ({P_error_map*100:.2f}%)")
print(f"   • Correct: {num_correct_map}/{N}")

print("\n3. PART B - ERM CLASSIFICATION:")
print(f"   • Decision rule: D(x) = arg min_i Σ_j λ_ij p(x|L=j)P(L=j)")
print(f"   • Minimizes expected risk with custom loss matrix")
print(f"   • Expected Risk = {emp_expected_risk:.4f}")
print(f"   • High penalty (100) for missing Class 4")

print("\n4. KEY INSIGHT:")
print("   • ERM classifier sacrifices overall accuracy to avoid costly errors")
print("   • Class 4 detection improves from {:.2%} (MAP) to {:.2%} (ERM)".format(
          conf_prob_map[3,3], conf_prob_erm[3,3]))
print("   • This demonstrates asymmetric loss optimization")

print("\n" + "="*80)
print("ANALYSIS COMPLETE - Files saved:")
print("  Figure 1: q2_fig1_data_true_labels.png")
print("  Figure 2: q2_fig2_class_means.png")
print("  Figure 3: q2_fig3_map_results.png")
print("  Figure 4: q2_fig4_map_confusion.png")
print("  Figure 5: q2_fig5_loss_matrix.png")
print("  Figure 6: q2_fig6_erm_results.png")
print("  Figure 7: q2_fig7_erm_confusion.png")
print("  Figure 8: q2_fig8_decision_distribution.png")
print("  Figure 9: q2_fig9_per_class_accuracy.png")
print("  Figure 10: q2_fig10_metrics_comparison.png")
print("="*80)