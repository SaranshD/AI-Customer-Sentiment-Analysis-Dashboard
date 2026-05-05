# Customer Sentiment Analysis Dashboard

AI-powered dashboard for classifying customer reviews as Positive, Neutral, or Negative using a pre-trained BERT transformer model.

## Features

- **ML Model**: Pre-trained `cardiffnlp/twitter-roberta-base-sentiment-latest` (3-class sentiment)
- **REST API**: FastAPI endpoints for programmatic access
- **Dashboard**: Gradio UI with sentiment distribution, timeline, and word frequency charts
- **Database**: SQLite storage for inference history
- **Evaluation**: Accuracy, F1 score, and confusion matrix

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Evaluate model
python main.py eval

# Start dashboard
python main.py dashboard

# Start API
python main.py api
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `python main.py eval` | Evaluate model on test set |
| `python main.py eval --full` | Use full test set (~4k samples) |
| `python main.py eval --store` | Evaluate and seed database |
| `python main.py dashboard` | Launch Gradio dashboard |
| `python main.py api` | Start FastAPI server |
| `python main.py analyze --text "..."` | Single prediction |

## API Endpoints

- `POST /analyze` - Single review classification
- `POST /batch` - Batch classification (up to 100)
- `GET /statistics` - Inference statistics
- `GET /results` - Recent results
- `DELETE /results` - Clear all results
- `GET /health` - Health check

## Configuration

Edit `config/settings.py` to change:
- `MODEL_NAME` - HuggingFace model ID
- `EVAL_DATASET` - Evaluation dataset
- `DB_PATH` - Database location

## Performance

- ~76% accuracy on test set (pre-trained, no fine-tuning)
- CPU-based inference (~80ms per sample)

## Project Structure

```
.
├── main.py                 # CLI entry point
├── evaluate.py             # Model evaluation
├── test_api.py            # API test suite
├── config/
│   └── settings.py         # Configuration
└── src/
    ├── models/            # SentimentClassifier
    ├── db/                # Database ORM
    ├── api/               # FastAPI app
    ├── dashboard/         # Gradio dashboard
    └── utils.py           # Utilities
```

## Requirements

Python 3.10+, transformers, torch, fastapi, gradio, sqlalchemy, scikit-learn