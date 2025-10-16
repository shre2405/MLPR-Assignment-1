import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# FILE PATHS
WINE_WHITE_PATH = r"D:\MS Sem-3 (fall 2025)\EECE 5644 - Machine Learning and Pattern Recognition\Assignments\Assignment-1 Q3 datasets\wine+quality\winequality-white.csv"
WINE_RED_PATH = r"D:\MS Sem-3 (fall 2025)\EECE 5644 - Machine Learning and Pattern Recognition\Assignments\Assignment-1 Q3 datasets\wine+quality\winequality-red.csv"

HAR_X_TRAIN_PATH = r"D:\MS Sem-3 (fall 2025)\EECE 5644 - Machine Learning and Pattern Recognition\Assignments\Assignment-1 Q3 datasets\human+activity+recognition+using+smartphones\UCI HAR Dataset\UCI HAR Dataset\train\X_train.txt"
HAR_Y_TRAIN_PATH = r"D:\MS Sem-3 (fall 2025)\EECE 5644 - Machine Learning and Pattern Recognition\Assignments\Assignment-1 Q3 datasets\human+activity+recognition+using+smartphones\UCI HAR Dataset\UCI HAR Dataset\train\y_train.txt"
HAR_X_TEST_PATH = r"D:\MS Sem-3 (fall 2025)\EECE 5644 - Machine Learning and Pattern Recognition\Assignments\Assignment-1 Q3 datasets\human+activity+recognition+using+smartphones\UCI HAR Dataset\UCI HAR Dataset\test\X_test.txt"
HAR_Y_TEST_PATH = r"D:\MS Sem-3 (fall 2025)\EECE 5644 - Machine Learning and Pattern Recognition\Assignments\Assignment-1 Q3 datasets\human+activity+recognition+using+smartphones\UCI HAR Dataset\UCI HAR Dataset\test\y_test.txt"

class GaussianClassifier:
    """Minimum Probability of Error Classifier with Gaussian Class-Conditional PDFs"""
    
    def __init__(self, regularization_alpha=0.01):
        self.alpha = regularization_alpha
        self.class_params = {}
        self.class_priors = {}
        self.classes = None
        
    def fit(self, X, y):
        """
        Estimate Gaussian parameters for each class
        X: n_samples x n_features
        y: n_samples (class labels)
        """
        self.classes = np.unique(y)
        n_samples = len(y)
        
        print(f"\nTraining classifier on {n_samples} samples with {X.shape[1]} features...")
        print(f"Classes found: {self.classes}")
        
        for c in self.classes:
            # Get samples for this class
            X_c = X[y == c]
            n_c = len(X_c)
            
            # Estimate class prior
            self.class_priors[c] = n_c / n_samples
            
            # Estimate mean vector
            mean = np.mean(X_c, axis=0)
            
            # Estimate covariance matrix
            cov = np.cov(X_c, rowvar=False)
            
            # Regularization to handle ill-conditioned covariance
            eigenvalues = np.linalg.eigvalsh(cov)
            nonzero_eigs = eigenvalues[eigenvalues > 1e-10]
            if len(nonzero_eigs) > 0:
                lambda_reg = self.alpha * np.mean(nonzero_eigs)
            else:
                lambda_reg = self.alpha
            
            cov_regularized = cov + lambda_reg * np.eye(cov.shape[0])
            
            # Check condition number
            cond_before = np.linalg.cond(cov) if cov.shape[0] == cov.shape[1] else np.inf
            cond_after = np.linalg.cond(cov_regularized)
            
            self.class_params[c] = {
                'mean': mean,
                'cov': cov_regularized,
                'prior': self.class_priors[c],
                'n_samples': n_c
            }
            
            print(f"  Class {c}: {n_c} samples ({100*n_c/n_samples:.1f}%), "
                  f"prior={self.class_priors[c]:.4f}, "
                  f"cond# before={cond_before:.2e}, after={cond_after:.2e}")
    
    def predict(self, X):
        """
        Classify samples using MAP rule (minimum P(error))
        """
        n_samples = X.shape[0]
        posteriors = np.zeros((n_samples, len(self.classes)))
        
        for idx, c in enumerate(self.classes):
            params = self.class_params[c]
            # Compute log posterior: log p(x|L=c) + log P(L=c)
            try:
                mvn = multivariate_normal(mean=params['mean'], 
                                         cov=params['cov'], 
                                         allow_singular=True)
                log_likelihood = mvn.logpdf(X)
                posteriors[:, idx] = log_likelihood + np.log(params['prior'])
            except Exception as e:
                print(f"Warning: Error computing likelihood for class {c}: {e}")
                posteriors[:, idx] = -np.inf
        
        # Return class with maximum posterior
        predictions = self.classes[np.argmax(posteriors, axis=1)]
        return predictions
    
    def compute_confusion_matrix(self, y_true, y_pred):
        """Compute confusion matrix P(D=i|L=j)"""
        classes = np.sort(np.unique(y_true))
        n_classes = len(classes)
        conf_matrix = np.zeros((n_classes, n_classes))
        
        for j, true_class in enumerate(classes):
            mask = y_true == true_class
            n_true = np.sum(mask)
            if n_true > 0:
                for i, pred_class in enumerate(classes):
                    conf_matrix[i, j] = np.sum((y_pred == pred_class) & mask) / n_true
        
        return conf_matrix, classes

def load_wine_quality():
    """Load Wine Quality datasets (red and white separately)"""
    print("Loading Wine Quality datasets...")
    
    try:
        # Load red wine
        df_red = pd.read_csv(WINE_RED_PATH, sep=';')
        X_red = df_red.iloc[:, :-1].values
        y_red = df_red.iloc[:, -1].values
        
        print(f"  Red wine: {len(df_red)} samples")
        print(f"    Classes: {np.unique(y_red)}")
        
        # Load white wine
        df_white = pd.read_csv(WINE_WHITE_PATH, sep=';')
        X_white = df_white.iloc[:, :-1].values
        y_white = df_white.iloc[:, -1].values
        
        print(f"  White wine: {len(df_white)} samples")
        print(f"    Classes: {np.unique(y_white)}")
        
        feature_names = df_red.columns[:-1].tolist()
        
        return X_red, y_red, X_white, y_white, feature_names
    
    except FileNotFoundError as e:
        print(f"Error: Could not find wine quality files. Please check paths.")
        print(f"Error details: {e}")
        return None, None, None, None, None

def load_har():
    """Load Human Activity Recognition dataset from local files"""
    print("\nLoading Human Activity Recognition dataset...")
    
    try:
        # Load training data
        X_train = np.loadtxt(HAR_X_TRAIN_PATH)
        y_train = np.loadtxt(HAR_Y_TRAIN_PATH).astype(int)
        
        # Load test data
        X_test = np.loadtxt(HAR_X_TEST_PATH)
        y_test = np.loadtxt(HAR_Y_TEST_PATH).astype(int)
        
        print(f"  Training set: {X_train.shape[0]} samples")
        print(f"  Test set: {X_test.shape[0]} samples")
        
        # Combine train and test for this assignment (using all available samples)
        X = np.vstack([X_train, X_test])
        y = np.hstack([y_train, y_test])
        
        print(f"\nCombined HAR: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Classes present: {np.unique(y)}")
        print(f"Class distribution:")
        activity_names = {1: 'Walking', 2: 'Walking Upstairs', 3: 'Walking Downstairs',
                         4: 'Sitting', 5: 'Standing', 6: 'Laying'}
        for cls in np.unique(y):
            count = np.sum(y == cls)
            print(f"  Class {cls} ({activity_names.get(cls, 'Unknown')}): {count} samples ({100*count/len(y):.1f}%)")
        
        return X, y, [f'feature_{i}' for i in range(X.shape[1])]
    
    except FileNotFoundError as e:
        print(f"Error: Could not find HAR files. Please check paths.")
        print(f"Error details: {e}")
        return None, None, None

def create_wine_comparison_plot(X_red, y_red, X_white, y_white, 
                                conf_red, classes_red, 
                                conf_white, classes_white,
                                pca_red, pca_white):
    """Create comprehensive comparison plot for red and white wine"""
    
    fig = plt.figure(figsize=(20, 12))
    
    # Red Wine PCA (2D)
    ax1 = plt.subplot(3, 4, 1)
    X_pca_red = pca_red.transform(X_red)
    colors = plt.cm.Reds(np.linspace(0.3, 1, len(classes_red)))
    for idx, c in enumerate(classes_red):
        mask = y_red == c
        ax1.scatter(X_pca_red[mask, 0], X_pca_red[mask, 1], 
                   label=f'Q{int(c)}', alpha=0.6, s=15, c=[colors[idx]])
    ax1.set_xlabel(f'PC1 ({pca_red.explained_variance_ratio_[0]:.2%})')
    ax1.set_ylabel(f'PC2 ({pca_red.explained_variance_ratio_[1]:.2%})')
    ax1.set_title('Red Wine: PC1 vs PC2', fontweight='bold')
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # White Wine PCA (2D)
    ax2 = plt.subplot(3, 4, 2)
    X_pca_white = pca_white.transform(X_white)
    colors = plt.cm.Blues(np.linspace(0.3, 1, len(classes_white)))
    for idx, c in enumerate(classes_white):
        mask = y_white == c
        ax2.scatter(X_pca_white[mask, 0], X_pca_white[mask, 1], 
                   label=f'Q{int(c)}', alpha=0.6, s=15, c=[colors[idx]])
    ax2.set_xlabel(f'PC1 ({pca_white.explained_variance_ratio_[0]:.2%})')
    ax2.set_ylabel(f'PC2 ({pca_white.explained_variance_ratio_[1]:.2%})')
    ax2.set_title('White Wine: PC1 vs PC2', fontweight='bold')
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)
    
    # Red Wine PCA (3D)
    ax3 = plt.subplot(3, 4, 3, projection='3d')
    colors = plt.cm.Reds(np.linspace(0.3, 1, len(classes_red)))
    for idx, c in enumerate(classes_red):
        mask = y_red == c
        ax3.scatter(X_pca_red[mask, 0], X_pca_red[mask, 1], X_pca_red[mask, 2],
                   label=f'Q{int(c)}', alpha=0.5, s=10, c=[colors[idx]])
    ax3.set_xlabel('PC1', fontsize=8)
    ax3.set_ylabel('PC2', fontsize=8)
    ax3.set_zlabel('PC3', fontsize=8)
    ax3.set_title('Red Wine: 3D PCA', fontweight='bold')
    
    # White Wine PCA (3D)
    ax4 = plt.subplot(3, 4, 4, projection='3d')
    colors = plt.cm.Blues(np.linspace(0.3, 1, len(classes_white)))
    for idx, c in enumerate(classes_white):
        mask = y_white == c
        ax4.scatter(X_pca_white[mask, 0], X_pca_white[mask, 1], X_pca_white[mask, 2],
                   label=f'Q{int(c)}', alpha=0.5, s=10, c=[colors[idx]])
    ax4.set_xlabel('PC1', fontsize=8)
    ax4.set_ylabel('PC2', fontsize=8)
    ax4.set_zlabel('PC3', fontsize=8)
    ax4.set_title('White Wine: 3D PCA', fontweight='bold')
    
    # Red Wine Confusion Matrix
    ax5 = plt.subplot(3, 4, 5)
    im1 = ax5.imshow(conf_red, interpolation='nearest', cmap='Reds', vmin=0, vmax=1)
    ax5.set_xticks(np.arange(len(classes_red)))
    ax5.set_yticks(np.arange(len(classes_red)))
    ax5.set_xticklabels([int(c) for c in classes_red])
    ax5.set_yticklabels([int(c) for c in classes_red])
    ax5.set_xlabel('True Label (L)')
    ax5.set_ylabel('Decision (D)')
    ax5.set_title('Red Wine: Confusion Matrix', fontweight='bold')
    plt.setp(ax5.get_xticklabels(), rotation=45, ha="right")
    
    # Add text annotations
    thresh = conf_red.max() / 2.
    for i in range(conf_red.shape[0]):
        for j in range(conf_red.shape[1]):
            ax5.text(j, i, f'{conf_red[i, j]:.2f}',
                   ha="center", va="center",
                   color="white" if conf_red[i, j] > thresh else "black",
                   fontsize=7)
    
    # White Wine Confusion Matrix
    ax6 = plt.subplot(3, 4, 6)
    im2 = ax6.imshow(conf_white, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    ax6.set_xticks(np.arange(len(classes_white)))
    ax6.set_yticks(np.arange(len(classes_white)))
    ax6.set_xticklabels([int(c) for c in classes_white])
    ax6.set_yticklabels([int(c) for c in classes_white])
    ax6.set_xlabel('True Label (L)')
    ax6.set_ylabel('Decision (D)')
    ax6.set_title('White Wine: Confusion Matrix', fontweight='bold')
    plt.setp(ax6.get_xticklabels(), rotation=45, ha="right")
    
    # Add text annotations
    thresh = conf_white.max() / 2.
    for i in range(conf_white.shape[0]):
        for j in range(conf_white.shape[1]):
            ax6.text(j, i, f'{conf_white[i, j]:.2f}',
                   ha="center", va="center",
                   color="white" if conf_white[i, j] > thresh else "black",
                   fontsize=7)
    
    # Red Wine Variance Explained
    ax7 = plt.subplot(3, 4, 7)
    n_comp = min(11, len(pca_red.explained_variance_ratio_))
    cumvar_red = np.cumsum(pca_red.explained_variance_ratio_[:n_comp])
    ax7.plot(range(1, n_comp+1), cumvar_red, 'ro-', markersize=8, linewidth=2)
    ax7.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5)
    ax7.set_xlabel('Number of Components')
    ax7.set_ylabel('Cumulative Variance')
    ax7.set_title('Red Wine: Variance Explained', fontweight='bold')
    ax7.grid(True, alpha=0.3)
    ax7.set_ylim([0, 1.05])
    
    # White Wine Variance Explained
    ax8 = plt.subplot(3, 4, 8)
    n_comp = min(11, len(pca_white.explained_variance_ratio_))
    cumvar_white = np.cumsum(pca_white.explained_variance_ratio_[:n_comp])
    ax8.plot(range(1, n_comp+1), cumvar_white, 'bo-', markersize=8, linewidth=2)
    ax8.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5)
    ax8.set_xlabel('Number of Components')
    ax8.set_ylabel('Cumulative Variance')
    ax8.set_title('White Wine: Variance Explained', fontweight='bold')
    ax8.grid(True, alpha=0.3)
    ax8.set_ylim([0, 1.05])
    
    # Red Wine Class Distribution
    ax9 = plt.subplot(3, 4, 9)
    class_counts_red = [np.sum(y_red == c) for c in classes_red]
    bars1 = ax9.bar(range(len(classes_red)), class_counts_red, color='darkred', alpha=0.7)
    ax9.set_xticks(range(len(classes_red)))
    ax9.set_xticklabels([int(c) for c in classes_red])
    ax9.set_xlabel('Quality Score')
    ax9.set_ylabel('Number of Samples')
    ax9.set_title('Red Wine: Class Distribution', fontweight='bold')
    ax9.grid(True, alpha=0.3, axis='y')
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax9.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    # White Wine Class Distribution
    ax10 = plt.subplot(3, 4, 10)
    class_counts_white = [np.sum(y_white == c) for c in classes_white]
    bars2 = ax10.bar(range(len(classes_white)), class_counts_white, color='darkblue', alpha=0.7)
    ax10.set_xticks(range(len(classes_white)))
    ax10.set_xticklabels([int(c) for c in classes_white])
    ax10.set_xlabel('Quality Score')
    ax10.set_ylabel('Number of Samples')
    ax10.set_title('White Wine: Class Distribution', fontweight='bold')
    ax10.grid(True, alpha=0.3, axis='y')
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax10.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    # Colorbar for confusion matrices
    cbar_ax1 = plt.subplot(3, 4, 11)
    cbar1 = plt.colorbar(im1, cax=cbar_ax1)
    cbar1.set_label('P(D=i|L=j)', rotation=270, labelpad=15)
    
    cbar_ax2 = plt.subplot(3, 4, 12)
    cbar2 = plt.colorbar(im2, cax=cbar_ax2)
    cbar2.set_label('P(D=i|L=j)', rotation=270, labelpad=15)
    
    plt.suptitle('Wine Quality Classification: Red vs White Comparison', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    return fig

def print_detailed_results(y_true, y_pred, classes, dataset_name):
    """Print detailed classification results"""
    print(f"\n{'='*80}")
    print(f"DETAILED RESULTS FOR {dataset_name.upper()}")
    print(f"{'='*80}")
    
    # Overall accuracy
    accuracy = np.mean(y_pred == y_true)
    error_rate = 1 - accuracy
    print(f"\nOverall Results:")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Error Rate (P(error)): {error_rate:.4f} ({error_rate*100:.2f}%)")
    print(f"  Total samples: {len(y_true)}")
    print(f"  Correct: {np.sum(y_pred == y_true)}")
    print(f"  Incorrect: {np.sum(y_pred != y_true)}")
    
    # Per-class results
    print(f"\nPer-Class Results:")
    print(f"{'Class':<8} {'N_true':<10} {'N_correct':<12} {'Accuracy':<12} {'Recall':<12}")
    print(f"{'-'*60}")
    
    for c in classes:
        mask_true = y_true == c
        n_true = np.sum(mask_true)
        n_correct = np.sum((y_pred == c) & mask_true)
        class_acc = n_correct / n_true if n_true > 0 else 0
        
        print(f"{int(c):<8} {n_true:<10} {n_correct:<12} {class_acc:<12.4f} {class_acc:<12.4f}")

def visualize_pca(X, y, dataset_name, n_components=3):
    """Visualize data using PCA"""
    print(f"\nPerforming PCA for {dataset_name}...")
    pca = PCA(n_components=min(n_components, X.shape[1]))
    X_pca = pca.fit_transform(X)
    
    print(f"  Variance explained by first {n_components} components: "
          f"{100*np.sum(pca.explained_variance_ratio_):.2f}%")
    
    return pca

def plot_confusion_matrix(conf_matrix, classes, dataset_name):
    """Plot confusion matrix"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(conf_matrix, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label('P(D=i|L=j)', rotation=270, labelpad=20)
    
    ax.set(xticks=np.arange(conf_matrix.shape[1]),
           yticks=np.arange(conf_matrix.shape[0]),
           xticklabels=[int(c) for c in classes], 
           yticklabels=[int(c) for c in classes],
           xlabel='True Label (L)',
           ylabel='Decision (D)',
           title=f'{dataset_name}: Confusion Matrix P(D=i|L=j)')
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            color = "white" if conf_matrix[i, j] > thresh else "black"
            fontsize = 10 if conf_matrix.shape[0] <= 7 else 8
            ax.text(j, i, f'{conf_matrix[i, j]:.3f}',
                   ha="center", va="center",
                   color=color, fontsize=fontsize)
    
    plt.tight_layout()
    return fig

# Main Analysis
print("="*80)
print("QUESTION 3: REAL DATASET CLASSIFICATION")
print("="*80)

# Dataset 1: Wine Quality (RED AND WHITE SEPARATELY)
print("\n" + "="*80)
print("DATASET 1: WINE QUALITY (SEPARATE ANALYSIS)")
print("="*80)

X_red, y_red, X_white, y_white, features_wine = load_wine_quality()

if X_red is not None and X_white is not None:
    
    # ========== RED WINE ANALYSIS ==========
    print("\n" + "-"*80)
    print("RED WINE ANALYSIS")
    print("-"*80)
    
    # PCA for red wine
    pca_red = visualize_pca(X_red, y_red, "Red Wine", n_components=11)
    
    # Train classifier for red wine
    clf_red = GaussianClassifier(regularization_alpha=0.01)
    clf_red.fit(X_red, y_red)
    
    # Predict
    print("\nClassifying red wine samples...")
    y_pred_red = clf_red.predict(X_red)
    
    # Confusion matrix
    conf_matrix_red, classes_red = clf_red.compute_confusion_matrix(y_red, y_pred_red)
    
    # Print detailed results
    print_detailed_results(y_red, y_pred_red, classes_red, "Red Wine")
    
    # ========== WHITE WINE ANALYSIS ==========
    print("\n" + "-"*80)
    print("WHITE WINE ANALYSIS")
    print("-"*80)
    
    # PCA for white wine
    pca_white = visualize_pca(X_white, y_white, "White Wine", n_components=11)
    
    # Train classifier for white wine
    clf_white = GaussianClassifier(regularization_alpha=0.01)
    clf_white.fit(X_white, y_white)
    
    # Predict
    print("\nClassifying white wine samples...")
    y_pred_white = clf_white.predict(X_white)
    
    # Confusion matrix
    conf_matrix_white, classes_white = clf_white.compute_confusion_matrix(y_white, y_pred_white)
    
    # Print detailed results
    print_detailed_results(y_white, y_pred_white, classes_white, "White Wine")
    
    # ========== CREATE SEPARATE PLOTS FOR RED AND WHITE WINE ==========
    print("\n" + "-"*80)
    print("Creating separate plots for red and white wine...")
    print("-"*80)
    
    # RED WINE PLOT
    fig_red = plt.figure(figsize=(18, 6))
    
    # Red Wine PCA 2D
    ax1 = fig_red.add_subplot(141)
    X_pca_red = pca_red.transform(X_red)
    colors = plt.cm.Reds(np.linspace(0.3, 1, len(classes_red)))
    for idx, c in enumerate(classes_red):
        mask = y_red == c
        ax1.scatter(X_pca_red[mask, 0], X_pca_red[mask, 1], 
                   label=f'Quality {int(c)}', alpha=0.6, s=20, c=[colors[idx]])
    ax1.set_xlabel(f'PC1 ({pca_red.explained_variance_ratio_[0]:.2%})')
    ax1.set_ylabel(f'PC2 ({pca_red.explained_variance_ratio_[1]:.2%})')
    ax1.set_title('Red Wine: PC1 vs PC2', fontweight='bold', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Red Wine PCA 3D
    ax2 = fig_red.add_subplot(142, projection='3d')
    for idx, c in enumerate(classes_red):
        mask = y_red == c
        ax2.scatter(X_pca_red[mask, 0], X_pca_red[mask, 1], X_pca_red[mask, 2],
                   label=f'Q{int(c)}', alpha=0.6, s=15, c=[colors[idx]])
    ax2.set_xlabel(f'PC1 ({pca_red.explained_variance_ratio_[0]:.1%})')
    ax2.set_ylabel(f'PC2 ({pca_red.explained_variance_ratio_[1]:.1%})')
    ax2.set_zlabel(f'PC3 ({pca_red.explained_variance_ratio_[2]:.1%})')
    ax2.set_title('Red Wine: 3D PCA', fontweight='bold', fontsize=12)
    ax2.legend(fontsize=8)
    
    # Red Wine Confusion Matrix
    ax3 = fig_red.add_subplot(143)
    im_red = ax3.imshow(conf_matrix_red, interpolation='nearest', cmap='Reds', vmin=0, vmax=1)
    ax3.set_xticks(np.arange(len(classes_red)))
    ax3.set_yticks(np.arange(len(classes_red)))
    ax3.set_xticklabels([int(c) for c in classes_red])
    ax3.set_yticklabels([int(c) for c in classes_red])
    ax3.set_xlabel('True Label (L)', fontsize=10)
    ax3.set_ylabel('Decision (D)', fontsize=10)
    ax3.set_title('Red Wine: Confusion Matrix P(D=i|L=j)', fontweight='bold', fontsize=12)
    plt.setp(ax3.get_xticklabels(), rotation=45, ha="right")
    
    # Add annotations to confusion matrix
    thresh = conf_matrix_red.max() / 2.
    for i in range(conf_matrix_red.shape[0]):
        for j in range(conf_matrix_red.shape[1]):
            ax3.text(j, i, f'{conf_matrix_red[i, j]:.2f}',
                   ha="center", va="center",
                   color="white" if conf_matrix_red[i, j] > thresh else "black",
                   fontsize=9)
    
    # Red Wine Variance Explained
    ax4 = fig_red.add_subplot(144)
    n_comp = min(11, len(pca_red.explained_variance_ratio_))
    cumvar_red = np.cumsum(pca_red.explained_variance_ratio_[:n_comp])
    ax4.plot(range(1, n_comp+1), cumvar_red, 'ro-', markersize=8, linewidth=2, label='Red Wine')
    ax4.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='90% variance')
    ax4.set_xlabel('Number of Components', fontsize=10)
    ax4.set_ylabel('Cumulative Variance Explained', fontsize=10)
    ax4.set_title('Red Wine: Variance Explained', fontweight='bold', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1.05])
    ax4.legend(fontsize=9)
    
    plt.suptitle('Red Wine Quality Classification Analysis', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('wine_red_analysis.png', dpi=200, bbox_inches='tight')
    print("Red wine analysis saved as 'wine_red_analysis.png'")
    
    # WHITE WINE PLOT
    fig_white = plt.figure(figsize=(18, 6))
    
    # White Wine PCA 2D
    ax1 = fig_white.add_subplot(141)
    X_pca_white = pca_white.transform(X_white)
    colors = plt.cm.Blues(np.linspace(0.3, 1, len(classes_white)))
    for idx, c in enumerate(classes_white):
        mask = y_white == c
        ax1.scatter(X_pca_white[mask, 0], X_pca_white[mask, 1], 
                   label=f'Quality {int(c)}', alpha=0.6, s=20, c=[colors[idx]])
    ax1.set_xlabel(f'PC1 ({pca_white.explained_variance_ratio_[0]:.2%})')
    ax1.set_ylabel(f'PC2 ({pca_white.explained_variance_ratio_[1]:.2%})')
    ax1.set_title('White Wine: PC1 vs PC2', fontweight='bold', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # White Wine PCA 3D
    ax2 = fig_white.add_subplot(142, projection='3d')
    for idx, c in enumerate(classes_white):
        mask = y_white == c
        ax2.scatter(X_pca_white[mask, 0], X_pca_white[mask, 1], X_pca_white[mask, 2],
                   label=f'Q{int(c)}', alpha=0.6, s=15, c=[colors[idx]])
    ax2.set_xlabel(f'PC1 ({pca_white.explained_variance_ratio_[0]:.1%})')
    ax2.set_ylabel(f'PC2 ({pca_white.explained_variance_ratio_[1]:.1%})')
    ax2.set_zlabel(f'PC3 ({pca_white.explained_variance_ratio_[2]:.1%})')
    ax2.set_title('White Wine: 3D PCA', fontweight='bold', fontsize=12)
    ax2.legend(fontsize=8)
    
    # White Wine Confusion Matrix
    ax3 = fig_white.add_subplot(143)
    im_white = ax3.imshow(conf_matrix_white, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    ax3.set_xticks(np.arange(len(classes_white)))
    ax3.set_yticks(np.arange(len(classes_white)))
    ax3.set_xticklabels([int(c) for c in classes_white])
    ax3.set_yticklabels([int(c) for c in classes_white])
    ax3.set_xlabel('True Label (L)', fontsize=10)
    ax3.set_ylabel('Decision (D)', fontsize=10)
    ax3.set_title('White Wine: Confusion Matrix P(D=i|L=j)', fontweight='bold', fontsize=12)
    plt.setp(ax3.get_xticklabels(), rotation=45, ha="right")
    
    # Add annotations to confusion matrix
    thresh = conf_matrix_white.max() / 2.
    for i in range(conf_matrix_white.shape[0]):
        for j in range(conf_matrix_white.shape[1]):
            ax3.text(j, i, f'{conf_matrix_white[i, j]:.2f}',
                   ha="center", va="center",
                   color="white" if conf_matrix_white[i, j] > thresh else "black",
                   fontsize=8)
    
    # White Wine Variance Explained
    ax4 = fig_white.add_subplot(144)
    n_comp = min(11, len(pca_white.explained_variance_ratio_))
    cumvar_white = np.cumsum(pca_white.explained_variance_ratio_[:n_comp])
    ax4.plot(range(1, n_comp+1), cumvar_white, 'bo-', markersize=8, linewidth=2, label='White Wine')
    ax4.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='90% variance')
    ax4.set_xlabel('Number of Components', fontsize=10)
    ax4.set_ylabel('Cumulative Variance Explained', fontsize=10)
    ax4.set_title('White Wine: Variance Explained', fontweight='bold', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1.05])
    ax4.legend(fontsize=9)
    
    plt.suptitle('White Wine Quality Classification Analysis', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('wine_white_analysis.png', dpi=200, bbox_inches='tight')
    print("White wine analysis saved as 'wine_white_analysis.png'")

# Dataset 2: Human Activity Recognition
print("\n" + "="*80)
print("DATASET 2: HUMAN ACTIVITY RECOGNITION")
print("="*80)

X_har, y_har, features_har = load_har()

if X_har is not None:
    # PCA
    pca_har = visualize_pca(X_har, y_har, "Human Activity Recognition", n_components=3)
    
    # Create separate HAR visualization
    fig_har = plt.figure(figsize=(18, 5))
    
    # 2D PCA
    ax1 = fig_har.add_subplot(131)
    X_pca_har = pca_har.transform(X_har)
    classes_har_plot = np.unique(y_har)
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes_har_plot)))
    activity_names = {1: 'Walking', 2: 'Walk Up', 3: 'Walk Down',
                     4: 'Sitting', 5: 'Standing', 6: 'Laying'}
    
    for idx, c in enumerate(classes_har_plot):
        mask = y_har == c
        ax1.scatter(X_pca_har[mask, 0], X_pca_har[mask, 1], 
                   label=f'{activity_names[c]}', alpha=0.6, s=20, c=[colors[idx]])
    ax1.set_xlabel(f'PC1 ({pca_har.explained_variance_ratio_[0]:.2%})')
    ax1.set_ylabel(f'PC2 ({pca_har.explained_variance_ratio_[1]:.2%})')
    ax1.set_title('HAR: PC1 vs PC2', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 3D PCA
    ax2 = fig_har.add_subplot(132, projection='3d')
    for idx, c in enumerate(classes_har_plot):
        mask = y_har == c
        ax2.scatter(X_pca_har[mask, 0], X_pca_har[mask, 1], X_pca_har[mask, 2],
                   label=f'{activity_names[c]}', alpha=0.5, s=15, c=[colors[idx]])
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_zlabel('PC3')
    ax2.set_title('HAR: 3D PCA', fontweight='bold')
    ax2.legend()
    
    # Variance explained
    ax3 = fig_har.add_subplot(133)
    n_comp = min(20, len(pca_har.explained_variance_ratio_))
    cumvar = np.cumsum(pca_har.explained_variance_ratio_[:n_comp])
    ax3.plot(range(1, n_comp+1), cumvar, 'go-', markersize=8, linewidth=2)
    ax3.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='90%')
    ax3.set_xlabel('Number of Components')
    ax3.set_ylabel('Cumulative Variance')
    ax3.set_title('HAR: Variance Explained', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig('har_pca_visualization.png', dpi=150, bbox_inches='tight')
    print("HAR PCA visualization saved as 'har_pca_visualization.png'")
    
    # Train classifier
    clf_har = GaussianClassifier(regularization_alpha=0.01)
    clf_har.fit(X_har, y_har)
    
    # Predict
    print("\nClassifying HAR samples...")
    y_pred_har = clf_har.predict(X_har)
    
    # Confusion matrix
    conf_matrix_har, classes_har = clf_har.compute_confusion_matrix(y_har, y_pred_har)
    
    # Print detailed results
    print_detailed_results(y_har, y_pred_har, classes_har, "Human Activity Recognition")
    
    # Plot confusion matrix
    fig_conf_har = plot_confusion_matrix(conf_matrix_har, classes_har, "Human Activity Recognition")
    plt.savefig('har_confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("\nHAR confusion matrix saved as 'har_confusion_matrix.png'")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("\nGenerated files:")
print("  - wine_red_analysis.png (Red wine complete analysis)")
print("  - wine_white_analysis.png (White wine complete analysis)")
print("  - har_pca_visualization.png")
print("  - har_confusion_matrix.png")
print("\nRun plt.show() to display all figures.")

# Show all figures
plt.show()