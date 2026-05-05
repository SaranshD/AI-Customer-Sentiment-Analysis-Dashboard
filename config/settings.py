import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DB_PATH = os.path.join(BASE_DIR, "sentiment_analysis.db")

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
EVAL_DATASET = "mteb/tweet_sentiment_extraction"
EVAL_DATASET_SPLIT = "test"

LABEL_MAPPING = {
    "Negative": 0,
    "Neutral": 1,
    "Positive": 2
}

REVERSE_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}

# Training config - reserved for future fine-tuning support
TRAINING_CONFIG = {
    "optimizer": "AdamW",
    "learning_rate": 2e-5,
    "epochs": 3,
    "batch_size": 16,
    "warmup_steps": 500,
    "weight_decay": 0.01,
}
