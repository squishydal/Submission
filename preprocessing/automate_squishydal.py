"""
automate_squishydal.py
Preprocessing otomatis dataset Dreaddit.
Mengembalikan data yang siap dilatih.
"""

import os
import argparse
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


DROP_COLS = ['id', 'post_id', 'sentence_range', 'text', 'confidence']
CATEGORICAL_COLS = ['subreddit']
TARGET_COL = 'label'


def load_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"[load] train={train_df.shape}, test={test_df.shape}")
    return train_df, test_df


def drop_unused_columns(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols_to_drop = [c for c in DROP_COLS if c in train_df.columns]
    train_df = train_df.drop(columns=cols_to_drop)
    test_df = test_df.drop(columns=cols_to_drop)
    print(f"[drop] dropped {cols_to_drop}")
    return train_df, test_df


def encode_categoricals(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    for col in CATEGORICAL_COLS:
        if col not in train_df.columns:
            continue
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col])
        test_df[col] = le.transform(test_df[col])
        print(f"[encode] {col}: {list(le.classes_)}")
    return train_df, test_df


def scale_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = [c for c in train_df.columns if c != TARGET_COL]

    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    print(f"[scale] StandardScaler applied to {len(feature_cols)} features")
    return train_df, test_df


def save_results(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    train_out = os.path.join(output_dir, 'dreaddit-train-preprocessed.csv')
    test_out = os.path.join(output_dir, 'dreaddit-test-preprocessed.csv')
    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)
    print(f"[save] {train_out} {train_df.shape}")
    print(f"[save] {test_out}  {test_df.shape}")


def preprocess(
    train_path: str,
    test_path: str,
    output_dir: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Jalankan seluruh pipeline preprocessing.
    Mengembalikan (X_train_ready, X_test_ready) sebagai DataFrame.
    """
    train_df, test_df = load_data(train_path, test_path)
    train_df, test_df = drop_unused_columns(train_df, test_df)
    train_df, test_df = encode_categoricals(train_df, test_df)
    train_df, test_df = scale_features(train_df, test_df)
    save_results(train_df, test_df, output_dir)
    return train_df, test_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing otomatis Dreaddit')
    parser.add_argument(
        '--train', default='../dreaddit_raw/dreaddit-train.csv',
        help='Path ke file train raw'
    )
    parser.add_argument(
        '--test', default='../dreaddit_raw/dreaddit-test.csv',
        help='Path ke file test raw'
    )
    parser.add_argument(
        '--output', default='dreaddit_preprocessing',
        help='Folder output hasil preprocessing'
    )
    args = parser.parse_args()

    train_ready, test_ready = preprocess(args.train, args.test, args.output)
    print("\n[done] Preprocessing selesai.")
    print(f"  Train: {train_ready.shape}")
    print(f"  Test : {test_ready.shape}")
