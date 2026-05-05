import os
import sys
import torch
import random
import time
from tqdm import tqdm

sys.path.insert(0, os.getcwd())

from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import pandas as pd
from src.models import SentimentClassifier
from src.db import Database
from src.utils import clean_text
from config.settings import MODEL_NAME, DB_PATH, EVAL_DATASET, EVAL_DATASET_SPLIT


def main(args=None):
    if args is None:
        import argparse
        parser = argparse.ArgumentParser(description="Evaluate model and optionally seed database")
        parser.add_argument('--num_samples', type=int, default=500, help='Number of samples to evaluate')
        parser.add_argument('--full', action='store_true', help='Use full test set (overrides --num_samples)')
        parser.add_argument('--store', action='store_true', help='Store results in database')
        parser.add_argument('--clear', action='store_true', help='Clear database before storing')
        args = parser.parse_args()
    
    num_samples = args.num_samples
    use_full = args.full
    store_results = args.store
    clear_db = args.clear
    
    print("=" * 60)
    print("SENTIMENT ANALYSIS MODEL EVALUATION")
    print("=" * 60)
    print(f"\nModel: {MODEL_NAME}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Classes: Negative, Neutral, Positive (3-class)")
    print(f"Target samples: {'FULL TEST SET' if use_full else num_samples}")
    print(f"Store results in DB: {store_results}")
    print("\nPreprocessing steps:")
    print("  - Lowercase")
    print("  - Remove URLs")
    print("  - Remove @mentions")
    print("  - Remove hashtags")
    print("  - Remove punctuation")
    print("  - Normalize whitespace")
    
    print("\nLoading model...")
    clf = SentimentClassifier(MODEL_NAME)
    
    db = None
    if store_results:
        db = Database(DB_PATH)
        if clear_db:
            print("Clearing existing data...")
            db.clear_all()
    
    print(f"\nLoading {EVAL_DATASET} dataset...")
    dataset = load_dataset(EVAL_DATASET, split=EVAL_DATASET_SPLIT)
    
    df = dataset.to_pandas()
    df = df[['text', 'label_text']].rename(columns={'label_text': 'true_label'})
    df = df.dropna()
    df['true_label'] = df['true_label'].str.capitalize()
    
    print(f"Total available samples: {len(df)}")
    
    if use_full:
        print(f"\nUsing full test set ({len(df)} samples)")
    else:
        shuffle_seed = random.randint(1, 999999)
        print(f"\nRandomly sampling {num_samples} samples (seed: {shuffle_seed})...")
        df = df.sample(n=min(num_samples, len(df)), random_state=shuffle_seed)
    
    print(f"\nLabel distribution:")
    print(df['true_label'].value_counts())
    
    print("\nPreprocessing text data...")
    df['text_cleaned'] = df['text'].apply(clean_text)
    df = df[df['text_cleaned'].str.len() > 0]
    print(f"After cleaning: {len(df)}")
    
    print("\nRunning predictions...")
    predictions = []
    correct = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
        text = str(row['text_cleaned'])
        true_label = row['true_label']
        
        result = clf.predict(text, return_probs=True)
        sentiment = result['sentiment']
        confidence = result['confidence']
        probs = result.get('probabilities', {})
        
        predictions.append(sentiment)
        
        if sentiment == true_label:
            correct += 1
        
        if store_results and db:
            db.add_result(
                text=str(row['text']),
                sentiment=sentiment,
                confidence=confidence,
                positive_prob=probs.get('Positive', 0),
                negative_prob=probs.get('Negative', 0),
                neutral_prob=probs.get('Neutral', 0),
                processing_time_ms=0,
                true_label=true_label
            )
    
    df['predicted'] = predictions
    
    y_true = df['true_label'].tolist()
    y_pred = df['predicted'].tolist()
    all_labels = ['Negative', 'Neutral', 'Positive']
    
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT (3-Class)")
    print("=" * 60)
    print(classification_report(y_true, y_pred, labels=all_labels, zero_division=0))
    
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX (3-Class)")
    print("=" * 60)
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    print(f"\n{'True\\Pred':<15} {'Negative':>12} {'Neutral':>12} {'Positive':>12}")
    print("-" * 55)
    for i, label in enumerate(all_labels):
        print(f"{label:<15} {cm[i][0]:>12} {cm[i][1]:>12} {cm[i][2]:>12}")
    print("-" * 55)
    
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print("\n" + "=" * 60)
    print("OVERALL METRICS")
    print("=" * 60)
    print(f"Accuracy:  {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"F1 Score:  {f1:.4f}  ({f1*100:.2f}%)")
    
    print("\n" + "=" * 60)
    print("PREDICTION DISTRIBUTION")
    print("=" * 60)
    pred_counts = pd.Series(y_pred).value_counts()
    for label in all_labels:
        count = pred_counts.get(label, 0)
        pct = count / len(y_pred) * 100
        print(f"  {label}: {count} ({pct:.1f}%)")
    
    if store_results:
        print("\n" + "=" * 60)
        print("DATABASE SEEDING")
        print("=" * 60)
        print(f"Seeded {len(df)} results to database")
        print(f"Correct predictions: {correct}")
        print(f"Accuracy: {correct/len(df)*100:.2f}%")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
