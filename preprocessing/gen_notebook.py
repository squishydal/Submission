import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# Title
cells.append(nbf.v4.new_markdown_cell("""# Eksperimen Preprocessing - Dreaddit Stress Detection
**Nama**: squishydal  
**Dataset**: Dreaddit - Stress Analysis in Social Media  
**Task**: Binary Classification (label 0 = tidak stress, label 1 = stress)
"""))

# ── 1. DATA LOADING ──────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 1. Data Loading"))

cells.append(nbf.v4.new_code_cell("""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load raw data
train_df = pd.read_csv('../dreaddit_raw/dreaddit-train.csv')
test_df  = pd.read_csv('../dreaddit_raw/dreaddit-test.csv')

print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)
train_df.head()
"""))

cells.append(nbf.v4.new_code_cell("""\
print("=== Train Info ===")
train_df.info(verbose=False, show_counts=True)
"""))

# ── 2. EDA ───────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 2. Exploratory Data Analysis (EDA)"))

cells.append(nbf.v4.new_markdown_cell("### 2.1 Statistik Deskriptif"))
cells.append(nbf.v4.new_code_cell("""\
train_df.describe().T.head(20)
"""))

cells.append(nbf.v4.new_markdown_cell("### 2.2 Missing Values"))
cells.append(nbf.v4.new_code_cell("""\
missing = train_df.isnull().sum()
missing = missing[missing > 0]
if len(missing) == 0:
    print("Tidak ada missing values pada dataset train.")
else:
    print(missing)
"""))

cells.append(nbf.v4.new_markdown_cell("### 2.3 Distribusi Label (Target)"))
cells.append(nbf.v4.new_code_cell("""\
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Train
train_df['label'].value_counts().plot(kind='bar', ax=axes[0], color=['#5bc0eb','#e55934'])
axes[0].set_title('Distribusi Label - Train')
axes[0].set_xlabel('Label (0=Tidak Stress, 1=Stress)')
axes[0].set_ylabel('Jumlah')
axes[0].tick_params(axis='x', rotation=0)

# Test
test_df['label'].value_counts().plot(kind='bar', ax=axes[1], color=['#5bc0eb','#e55934'])
axes[1].set_title('Distribusi Label - Test')
axes[1].set_xlabel('Label (0=Tidak Stress, 1=Stress)')
axes[1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig('distribusi_label.png', dpi=100, bbox_inches='tight')
plt.show()

print("Train label counts:")
print(train_df['label'].value_counts())
print("\\nTest label counts:")
print(test_df['label'].value_counts())
"""))

cells.append(nbf.v4.new_markdown_cell("### 2.4 Distribusi per Subreddit"))
cells.append(nbf.v4.new_code_cell("""\
fig, ax = plt.subplots(figsize=(12, 5))
subreddit_label = train_df.groupby(['subreddit', 'label']).size().unstack(fill_value=0)
subreddit_label.plot(kind='bar', ax=ax, color=['#5bc0eb','#e55934'])
ax.set_title('Distribusi Label per Subreddit - Train')
ax.set_xlabel('Subreddit')
ax.set_ylabel('Jumlah')
ax.tick_params(axis='x', rotation=45)
ax.legend(['Tidak Stress', 'Stress'])
plt.tight_layout()
plt.savefig('distribusi_subreddit.png', dpi=100, bbox_inches='tight')
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("### 2.5 Korelasi Fitur Numerik terhadap Label"))
cells.append(nbf.v4.new_code_cell("""\
# Kolom yang bukan fitur model
drop_cols = ['id', 'post_id', 'sentence_range', 'text', 'subreddit', 'label', 'confidence']
num_cols = [c for c in train_df.columns if c not in drop_cols]

corr = train_df[num_cols + ['label']].corr()['label'].drop('label').abs().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
corr.head(20).plot(kind='barh', color='steelblue')
plt.title('Top 20 Fitur dengan Korelasi Tertinggi terhadap Label')
plt.xlabel('Korelasi Absolut')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('korelasi_fitur.png', dpi=100, bbox_inches='tight')
plt.show()

print("Top 10 fitur paling berkorelasi dengan label:")
print(corr.head(10))
"""))

cells.append(nbf.v4.new_markdown_cell("### 2.6 Distribusi Beberapa Fitur Penting"))
cells.append(nbf.v4.new_code_cell("""\
top_features = corr.head(6).index.tolist()

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, feat in enumerate(top_features):
    for label_val, color in [(0, '#5bc0eb'), (1, '#e55934')]:
        subset = train_df[train_df['label'] == label_val][feat]
        axes[i].hist(subset, bins=30, alpha=0.6, color=color,
                     label=f'label={label_val}')
    axes[i].set_title(feat)
    axes[i].legend()

plt.tight_layout()
plt.savefig('distribusi_fitur_penting.png', dpi=100, bbox_inches='tight')
plt.show()
"""))

# ── 3. PREPROCESSING ─────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 3. Preprocessing"))

cells.append(nbf.v4.new_markdown_cell("### 3.1 Drop Kolom yang Tidak Digunakan"))
cells.append(nbf.v4.new_code_cell("""\
# Kolom identifier dan teks mentah tidak digunakan untuk model klasik
drop_cols = ['id', 'post_id', 'sentence_range', 'text', 'confidence']

train_clean = train_df.drop(columns=drop_cols)
test_clean  = test_df.drop(columns=drop_cols)

print("Train shape setelah drop:", train_clean.shape)
print("Test shape setelah drop :", test_clean.shape)
"""))

cells.append(nbf.v4.new_markdown_cell("### 3.2 Encode Kolom Kategorikal (subreddit)"))
cells.append(nbf.v4.new_code_cell("""\
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
train_clean['subreddit'] = le.fit_transform(train_clean['subreddit'])
test_clean['subreddit']  = le.transform(test_clean['subreddit'])

print("Subreddit encoding mapping:")
for i, cls in enumerate(le.classes_):
    print(f"  {cls} -> {i}")
"""))

cells.append(nbf.v4.new_markdown_cell("### 3.3 Pisahkan Fitur dan Target"))
cells.append(nbf.v4.new_code_cell("""\
X_train = train_clean.drop(columns=['label'])
y_train = train_clean['label']

X_test  = test_clean.drop(columns=['label'])
y_test  = test_clean['label']

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train distribusi:", y_train.value_counts().to_dict())
print("y_test  distribusi:", y_test.value_counts().to_dict())
"""))

cells.append(nbf.v4.new_markdown_cell("### 3.4 Feature Scaling (StandardScaler)"))
cells.append(nbf.v4.new_code_cell("""\
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Kembalikan ke DataFrame agar lebih mudah dibaca
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled  = pd.DataFrame(X_test_scaled,  columns=X_test.columns)

print("Contoh hasil scaling (5 baris pertama, 5 kolom pertama):")
X_train_scaled.iloc[:5, :5]
"""))

cells.append(nbf.v4.new_markdown_cell("### 3.5 Simpan Hasil Preprocessing"))
cells.append(nbf.v4.new_code_cell("""\
import os

output_dir = 'dreaddit_preprocessing'
os.makedirs(output_dir, exist_ok=True)

# Gabungkan kembali fitur + label untuk disimpan
train_preprocessed = X_train_scaled.copy()
train_preprocessed['label'] = y_train.values

test_preprocessed = X_test_scaled.copy()
test_preprocessed['label'] = y_test.values

train_preprocessed.to_csv(f'{output_dir}/dreaddit-train-preprocessed.csv', index=False)
test_preprocessed.to_csv(f'{output_dir}/dreaddit-test-preprocessed.csv', index=False)

print("Saved:")
print(f"  {output_dir}/dreaddit-train-preprocessed.csv -> {train_preprocessed.shape}")
print(f"  {output_dir}/dreaddit-test-preprocessed.csv  -> {test_preprocessed.shape}")
"""))

# ── 4. SUMMARY ───────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
## 4. Summary Preprocessing

| Tahap | Detail |
|---|---|
| Drop kolom | id, post_id, sentence_range, text, confidence |
| Encode kategorikal | subreddit (LabelEncoder) |
| Feature scaling | StandardScaler pada semua fitur numerik |
| Output train | dreaddit_preprocessing/dreaddit-train-preprocessed.csv |
| Output test | dreaddit_preprocessing/dreaddit-test-preprocessed.csv |
| Jumlah fitur final | 110 fitur + 1 label |
"""))

nb.cells = cells

with open('Eksperimen_squishydal.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook generated: Eksperimen_squishydal.ipynb")
