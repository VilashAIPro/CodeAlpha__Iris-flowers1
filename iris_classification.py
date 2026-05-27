# =============================================================================
#  Iris Flower Classification -- CodeAlpha Data Science Internship Task 1
#  Author : Vilash Kumar
#  Date   : May 2026
#  Run    : python iris_classification.py
# =============================================================================

import sys
import io
# Force UTF-8 output so Unicode characters display correctly on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')   # Non-interactive backend — no display window needed

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from itertools import combinations

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ── COLOUR PALETTE ────────────────────────────────────────────────────────────
BG_DARK   = '#0A0A0F'   # Figure background
AX_DARK   = '#12121A'   # Axes background
GRID_CLR  = '#2A2A3E'   # Subtle grid lines
TEXT_CLR  = '#E8E8F0'   # Primary text
ACC_CLR   = '#7C6FF7'   # Accent (purple)

SPECIES_COLORS = {
    'Iris-setosa'    : '#FF6B9D',   # Pink
    'Iris-versicolor': '#00D4FF',   # Cyan
    'Iris-virginica' : '#FFB347'    # Orange
}
COLOR_LIST = list(SPECIES_COLORS.values())   # [pink, cyan, orange]
SPECIES_NAMES = list(SPECIES_COLORS.keys())

# ── GLOBAL PLOT STYLE ─────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor' : BG_DARK,
    'axes.facecolor'   : AX_DARK,
    'axes.edgecolor'   : GRID_CLR,
    'axes.labelcolor'  : TEXT_CLR,
    'axes.titlecolor'  : TEXT_CLR,
    'xtick.color'      : TEXT_CLR,
    'ytick.color'      : TEXT_CLR,
    'text.color'       : TEXT_CLR,
    'grid.color'       : GRID_CLR,
    'grid.alpha'       : 0.5,
    'font.family'      : 'DejaVu Sans',
    'font.size'        : 10,
})

SEPARATOR = '=' * 62

def section(title):
    """Pretty section header for console output."""
    print(f'\n{SEPARATOR}')
    print(f'  {title}')
    print(SEPARATOR)

# =============================================================================
# ── 1. LOAD DATASET ──────────────────────────────────────────────────────────
# =============================================================================
section('1. LOADING DATASET')

df = pd.read_csv('Iris.csv')                  # Read the raw CSV

print(f'[OK] Loaded  : Iris.csv')
print(f'[OK] Shape (before drop): {df.shape}')

df.drop(columns=['Id'], inplace=True)          # Drop the Id column (not a feature)
print(f'[OK] Shape (after drop Id): {df.shape}')

# -- Basic exploration --------------------------------------------------------
print('\n[>] First 5 rows:')
print(df.head())

print('\n[>] Data types:')
print(df.dtypes)

print('\n[>] Statistical summary:')
print(df.describe())

print('\n[>] Species distribution:')
print(df['Species'].value_counts())

print('\n[>] Null values per column:')
print(df.isnull().sum())
print('[OK] No missing values found!' if df.isnull().sum().sum() == 0 else '[!!] Missing values detected!')

FEATURES = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

# =============================================================================
# ── 2. PREPROCESSING ─────────────────────────────────────────────────────────
# =============================================================================
section('2. PREPROCESSING')

# Encode the target label: setosa=0, versicolor=1, virginica=2
le = LabelEncoder()
df['Species_Encoded'] = le.fit_transform(df['Species'])
print('[OK] LabelEncoder mapping:')
for idx, cls in enumerate(le.classes_):
    print(f'     {idx} -> {cls}')

X = df[FEATURES].values          # Feature matrix  (150 x 4)
y = df['Species_Encoded'].values  # Target vector   (150,)

# 80 / 20 stratified split (stratify keeps class balance in both sets)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f'\n[OK] Train size : {X_train.shape[0]} samples')
print(f'[OK] Test  size : {X_test.shape[0]}  samples')

# Standardise features (zero mean, unit variance)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)   # Fit on train, transform train
X_test_sc  = scaler.transform(X_test)        # Transform test with SAME scaler

print('[OK] StandardScaler applied -- features normalised.')

# =============================================================================
# ── 3. TRAIN MODELS ──────────────────────────────────────────────────────────
# =============================================================================
section('3. TRAINING MODELS')

models = {
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Support Vector Machine': SVC(kernel='rbf', random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
}

for name, model in models.items():
    model.fit(X_train_sc, y_train)
    print(f'[OK] Trained : {name}')

# =============================================================================
# ── 4. EVALUATE MODELS ───────────────────────────────────────────────────────
# =============================================================================
section('4. MODEL EVALUATION')

accuracies   = {}
predictions  = {}
target_names = list(le.classes_)

for name, model in models.items():
    y_pred = model.predict(X_test_sc)
    acc    = accuracy_score(y_test, y_pred)
    accuracies[name]  = acc
    predictions[name] = y_pred

    print(f'\n+-- {name} ----------------------------------------')
    print(f'|   Accuracy : {acc * 100:.2f}%')
    print(f'|\n|   Classification Report:')
    report = classification_report(y_test, y_pred, target_names=target_names)
    for line in report.split('\n'):
        print(f'|   {line}')
    print('+' + '-' * 50)

best_model = max(accuracies, key=accuracies.get)
best_acc   = accuracies[best_model]
print(f'\n[BEST] MODEL : {best_model}  ({best_acc * 100:.2f}%)')

# =============================================================================
# ── 5. VISUALISATIONS ────────────────────────────────────────────────────────
# =============================================================================
section('5. GENERATING VISUALIZATIONS')

# Helper: apply dark axis styling
def style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(AX_DARK)
    ax.set_title(title, color=TEXT_CLR, fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel(xlabel, color=TEXT_CLR, fontsize=9)
    ax.set_ylabel(ylabel, color=TEXT_CLR, fontsize=9)
    ax.grid(True, color=GRID_CLR, alpha=0.4, linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_CLR)

# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — EDA Scatter: all 6 feature-pair combinations (2×3 grid)
# ─────────────────────────────────────────────────────────────────────────────
print('  [1/5] Plotting eda_scatter.png ...')

pairs = list(combinations(FEATURES, 2))   # 6 unique pairs
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.patch.set_facecolor(BG_DARK)
fig.suptitle('Iris -- Feature Pair Scatter Plots', color=TEXT_CLR,
             fontsize=16, fontweight='bold', y=1.01)

for ax, (fx, fy) in zip(axes.flatten(), pairs):
    for species, color in SPECIES_COLORS.items():
        mask = df['Species'] == species
        ax.scatter(df.loc[mask, fx], df.loc[mask, fy],
                   c=color, alpha=0.85, s=55, edgecolors='none', label=species)
    style_ax(ax, title=f'{fx}  vs  {fy}',
             xlabel=fx.replace('Cm', ' (cm)'),
             ylabel=fy.replace('Cm', ' (cm)'))

# Shared legend below the plots
patches = [mpatches.Patch(color=c, label=s) for s, c in SPECIES_COLORS.items()]
fig.legend(handles=patches, loc='lower center', ncol=3,
           facecolor=AX_DARK, edgecolor=GRID_CLR, labelcolor=TEXT_CLR,
           fontsize=11, framealpha=0.9, bbox_to_anchor=(0.5, -0.04))

plt.tight_layout()
plt.savefig('eda_scatter.png', dpi=150, bbox_inches='tight',
            facecolor=BG_DARK)
plt.close()
print('  [OK] eda_scatter.png saved.')

# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Feature Boxplots: 1×4 subplots, grouped by species
# ─────────────────────────────────────────────────────────────────────────────
print('  [2/5] Plotting feature_boxplot.png ...')

fig, axes = plt.subplots(1, 4, figsize=(18, 6))
fig.patch.set_facecolor(BG_DARK)
fig.suptitle('Iris -- Feature Distributions by Species',
             color=TEXT_CLR, fontsize=15, fontweight='bold')

for ax, feature in zip(axes, FEATURES):
    data_by_species = [df.loc[df['Species'] == sp, feature].values
                       for sp in SPECIES_NAMES]

    bp = ax.boxplot(data_by_species, patch_artist=True,
                    medianprops=dict(color='white', linewidth=2),
                    whiskerprops=dict(color=TEXT_CLR),
                    capprops=dict(color=TEXT_CLR),
                    flierprops=dict(markerfacecolor=TEXT_CLR,
                                   marker='o', markersize=4, alpha=0.6))

    for patch, color in zip(bp['boxes'], COLOR_LIST):
        patch.set_facecolor(color)
        patch.set_alpha(0.80)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Setosa', 'Versicolor', 'Virginica'],
                       color=TEXT_CLR, fontsize=8, rotation=15)
    style_ax(ax, title=feature.replace('Cm', ' (cm)'),
             ylabel='Value (cm)')

patches = [mpatches.Patch(color=c, label=s)
           for s, c in SPECIES_COLORS.items()]
fig.legend(handles=patches, loc='lower center', ncol=3,
           facecolor=AX_DARK, edgecolor=GRID_CLR, labelcolor=TEXT_CLR,
           fontsize=10, bbox_to_anchor=(0.5, -0.06))

plt.tight_layout()
plt.savefig('feature_boxplot.png', dpi=150, bbox_inches='tight',
            facecolor=BG_DARK)
plt.close()
print('  [OK] feature_boxplot.png saved.')

# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — Correlation Heatmap
# ─────────────────────────────────────────────────────────────────────────────
print('  [3/5] Plotting correlation_heatmap.png ...')

fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor(BG_DARK)
ax.set_facecolor(AX_DARK)

corr_matrix = df[FEATURES].corr()

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    linewidths=1,
    linecolor=BG_DARK,
    ax=ax,
    annot_kws={'size': 12, 'weight': 'bold', 'color': 'white'},
    cbar_kws={'shrink': 0.8},
    vmin=-1, vmax=1
)

ax.set_title('Iris -- Feature Correlation Heatmap',
             color=TEXT_CLR, fontsize=14, fontweight='bold', pad=12)
ax.tick_params(colors=TEXT_CLR, labelsize=10)
ax.set_xticklabels([f.replace('Cm', '') for f in FEATURES],
                   rotation=30, ha='right', color=TEXT_CLR)
ax.set_yticklabels([f.replace('Cm', '') for f in FEATURES],
                   rotation=0, color=TEXT_CLR)

cbar = ax.collections[0].colorbar
cbar.ax.yaxis.set_tick_params(color=TEXT_CLR, labelcolor=TEXT_CLR)

plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight',
            facecolor=BG_DARK)
plt.close()
print('  [OK] correlation_heatmap.png saved.')

# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 — Confusion Matrices: 1×3 subplots
# ─────────────────────────────────────────────────────────────────────────────
print('  [4/5] Plotting confusion_matrices.png ...')

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor(BG_DARK)
fig.suptitle('Iris -- Confusion Matrices', color=TEXT_CLR,
             fontsize=15, fontweight='bold')

SHORT = ['Setosa', 'Versicolor', 'Virginica']
CMAP_CM = sns.color_palette('rocket_r', as_cmap=True)   # Dark-theme friendly

for ax, (name, y_pred) in zip(axes, predictions.items()):
    cm  = confusion_matrix(y_test, y_pred)
    acc = accuracies[name] * 100

    sns.heatmap(cm, annot=True, fmt='d', cmap=CMAP_CM,
                xticklabels=SHORT, yticklabels=SHORT,
                linewidths=0.5, linecolor=BG_DARK,
                ax=ax, cbar=False,
                annot_kws={'size': 14, 'weight': 'bold'})

    title_str = f'{name}\n Accuracy: {acc:.1f}%'
    if name == best_model:
        title_str += '  [BEST]'
    ax.set_title(title_str, color=TEXT_CLR, fontsize=10, fontweight='bold')
    ax.set_xlabel('Predicted', color=TEXT_CLR, fontsize=9)
    ax.set_ylabel('Actual',    color=TEXT_CLR, fontsize=9)
    ax.tick_params(colors=TEXT_CLR, labelsize=8)
    ax.set_facecolor(AX_DARK)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_CLR)

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight',
            facecolor=BG_DARK)
plt.close()
print('  [OK] confusion_matrices.png saved.')

# ─────────────────────────────────────────────────────────────────────────────
# Plot 5 — Model Comparison Bar Chart
# ─────────────────────────────────────────────────────────────────────────────
print('  [5/5] Plotting model_comparison.png ...')

SHORT_NAMES = {
    'K-Nearest Neighbors'   : 'KNN\n(k=5)',
    'Support Vector Machine': 'SVM\n(RBF)',
    'Decision Tree'         : 'Decision\nTree'
}

fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor(BG_DARK)
ax.set_facecolor(AX_DARK)

names    = [SHORT_NAMES[n] for n in accuracies]
acc_vals = [v * 100 for v in accuracies.values()]
model_keys = list(accuracies.keys())

# Color bars (best model gets accent purple)
bar_colors = [ACC_CLR if m == best_model else '#4A4A6A'
              for m in model_keys]

bars = ax.bar(names, acc_vals, color=bar_colors, width=0.45,
              edgecolor=GRID_CLR, linewidth=0.8, zorder=3)

# Percentage label on top of each bar
for bar, val, mname in zip(bars, acc_vals, model_keys):
    label = f'{val:.2f}%'
    if mname == best_model:
        label += '  [BEST]'
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            label,
            ha='center', va='bottom',
            color=TEXT_CLR, fontsize=12, fontweight='bold')

ax.set_ylim(0, 115)
ax.set_ylabel('Accuracy (%)', color=TEXT_CLR, fontsize=11)
ax.set_title('Iris -- Model Accuracy Comparison',
             color=TEXT_CLR, fontsize=14, fontweight='bold', pad=12)
ax.yaxis.grid(True, color=GRID_CLR, alpha=0.5, linewidth=0.6, zorder=0)
ax.set_axisbelow(True)

for spine in ax.spines.values():
    spine.set_edgecolor(GRID_CLR)

# Annotation: best model note
ax.annotate(f'Best: {best_model}  ({best_acc*100:.2f}%)',
            xy=(0.5, 0.96), xycoords='axes fraction',
            ha='center', fontsize=10,
            color='#7C6FF7', style='italic')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight',
            facecolor=BG_DARK)
plt.close()
print('  [OK] model_comparison.png saved.')

# =============================================================================
# ── 6. FINAL SUMMARY ─────────────────────────────────────────────────────────
# =============================================================================
section('6. FINAL SUMMARY')

print(f'{"Model":<28} {"Accuracy":>10}')
print('-' * 40)
for name, acc in accuracies.items():
    star = ' [BEST]' if name == best_model else ''
    print(f'{name:<28} {acc * 100:>9.2f}%{star}')

print(f'\n[WINNER] Best Performing Model : {best_model}')
print(f'[SCORE]  Best Accuracy         : {best_acc * 100:.2f}%')
print('\n[FILES] Saved:')
print('   - iris_classification.py')
print('   - eda_scatter.png')
print('   - feature_boxplot.png')
print('   - correlation_heatmap.png')
print('   - confusion_matrices.png')
print('   - model_comparison.png')
print(f'\n{SEPARATOR}')
print('  [DONE] All done! -- CodeAlpha Internship Task 1 Complete')
print(SEPARATOR)
