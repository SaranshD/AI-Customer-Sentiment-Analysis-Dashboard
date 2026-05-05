import gradio as gr
from src.db import Database
from src.models import SentimentClassifier
from config.settings import DB_PATH, MODEL_NAME
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from collections import Counter
import re
from PIL import Image
import io


def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img = Image.open(buf)
    img_array = np.array(img)
    plt.close(fig)
    return img_array


db = Database(DB_PATH)
classifier = None


def get_classifier():
    global classifier
    if classifier is None:
        classifier = SentimentClassifier(MODEL_NAME)
    return classifier


def analyze_review(text):
    clf = get_classifier()
    result = clf.predict(text, return_probs=True)
    
    db.add_result(
        text=text,
        sentiment=result['sentiment'],
        confidence=result['confidence'],
        positive_prob=result.get('probabilities', {}).get('Positive', 0),
        negative_prob=result.get('probabilities', {}).get('Negative', 0),
        neutral_prob=result.get('probabilities', {}).get('Neutral', 0)
    )
    
    probs = result.get('probabilities', {})
    prob_text = f"Positive: {probs.get('Positive', 0)*100:.1f}%\n"
    prob_text += f"Neutral: {probs.get('Neutral', 0)*100:.1f}%\n"
    prob_text += f"Negative: {probs.get('Negative', 0)*100:.1f}%"
    
    return result['sentiment'], f"{result['confidence']*100:.1f}%", prob_text


def get_sentiment_distribution():
    counts = db.get_sentiment_counts()
    
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    labels = ['Positive', 'Neutral', 'Negative']
    values = [counts.get(l, 0) for l in labels]
    
    ax.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
    
    return fig_to_image(fig)


def get_sentiment_counts_over_time():
    daily_counts = db.get_daily_sentiment_counts(days=30)
    
    if not daily_counts:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig_to_image(fig)
    
    dates = [d['date'] for d in daily_counts]
    positive = [d.get('Positive', 0) for d in daily_counts]
    neutral = [d.get('Neutral', 0) for d in daily_counts]
    negative = [d.get('Negative', 0) for d in daily_counts]
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    x_pos = list(range(len(dates)))
    ax.plot(x_pos, positive, marker='o', label='Positive', color='#2ecc71', linewidth=2)
    ax.plot(x_pos, neutral, marker='s', label='Neutral', color='#3498db', linewidth=2)
    ax.plot(x_pos, negative, marker='^', label='Negative', color='#e74c3c', linewidth=2)
    
    ax.set_xticks(x_pos[::max(1, len(dates)//7)])
    ax.set_xticklabels(dates[::max(1, len(dates)//7)], rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title('Sentiment Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    return fig_to_image(fig)


def get_negative_words():
    negative_reviews = db.get_negative_reviews(limit=100)
    
    if not negative_reviews:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, 'No negative reviews yet', ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig_to_image(fig)
    
    stopwords = {'the', 'a', 'an', 'is', 'it', 'this', 'that', 'i', 'to', 'and', 'was', 'for', 'on', 'with', 'be', 'are', 'as', 'at', 'in', 'or', 'but', 'have', 'has', 'had', 'not', 'my', 'we', 'they', 'you', 'of', 'so', 'if', 'will', 'just', 'been', 'would', 'can', 'could', 'should', 'their', 'there', 'what', 'all', 'when', 'your', 'which', 'she', 'him', 'his', 'how', 'than', 'them', 'very', 'some', 'do', 'does', 'did', 'up', 'out', 'no', 'about', 'into', 'more', 'only', 'other', 'also', 'me', 'too', 'any', 'these', 'its'}
    
    all_words = []
    for review in negative_reviews:
        words = re.findall(r'\b[a-z]{3,}\b', review.text.lower())
        words = [w for w in words if w not in stopwords]
        all_words.extend(words)
    
    word_counts = Counter(all_words).most_common(15)
    
    if not word_counts:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, 'No words to display', ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig_to_image(fig)
    
    words, counts = zip(*word_counts)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.Reds([(c/max(counts))*0.5 + 0.5 for c in counts])
    ax.barh(range(len(words)), counts, color=colors)
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words)
    ax.set_xlabel('Frequency', fontsize=10)
    ax.set_title('Common Words in Negative Reviews', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    return fig_to_image(fig)


def get_uncertain_predictions():
    incorrect = db.get_incorrect_predictions(limit=3)
    
    if not incorrect:
        return pd.DataFrame(columns=["Text", "Predicted", "Actual", "Confidence"])
    
    data = []
    for pred in incorrect:
        data.append({
            "Text": pred.text,
            "Predicted": pred.sentiment,
            "Actual": pred.true_label,
            "Confidence": f"{pred.confidence*100:.1f}%"
        })
    
    return pd.DataFrame(data)


def get_statistics():
    stats = db.get_statistics()
    
    return f"""**Total Inferences:** {stats['total_inferences']}

**Sentiment Distribution:**
- Positive: {stats['sentiment_distribution'].get('Positive', 0)}
- Neutral: {stats['sentiment_distribution'].get('Neutral', 0)}
- Negative: {stats['sentiment_distribution'].get('Negative', 0)}

**Average Confidence:** {stats['average_confidence']*100:.1f}%

**Average Processing Time:** {stats['average_processing_time_ms']:.1f}ms"""


def create_dashboard():
    with gr.Blocks(title="Customer Sentiment Analysis Dashboard") as demo:
        gr.Markdown("# Customer Sentiment Analysis Dashboard")
        
        with gr.Tab("Analyze Review"):
            with gr.Row():
                with gr.Column():
                    review_input = gr.Textbox(
                        label="Enter Review Text",
                        placeholder="Type or paste a customer review here...",
                        lines=5
                    )
                    analyze_btn = gr.Button("Analyze Sentiment", variant="primary")
                
                with gr.Column():
                    sentiment_output = gr.Textbox(label="Predicted Sentiment", lines=1)
                    confidence_output = gr.Textbox(label="Confidence", lines=1)
                    probabilities_output = gr.Textbox(label="All Probabilities", lines=3)
        
        with gr.Tab("Dashboard"):
            with gr.Row():
                refresh_btn = gr.Button("Refresh Data", variant="secondary")
            
            with gr.Row():
                stats_display = gr.Markdown("*Click Refresh to load statistics*")
            
            gr.Markdown("### Sentiment Distribution")
            dist_plot = gr.Image(height=400)
            
            gr.Markdown("### Negative Review Words")
            words_plot = gr.Image(height=400)
            
            gr.Markdown("### Sentiment Over Time")
            timeline_plot = gr.Image(height=350)
            
            gr.Markdown("### Incorrect Predictions")
            uncertain_display = gr.DataFrame(headers=["Text", "Predicted", "Actual", "Confidence"], interactive=False, wrap=True)
        
        analyze_btn.click(
            fn=analyze_review,
            inputs=[review_input],
            outputs=[sentiment_output, confidence_output, probabilities_output]
        )
        
        refresh_btn.click(
            fn=lambda: (get_sentiment_distribution(), get_negative_words(), get_sentiment_counts_over_time(), get_statistics(), get_uncertain_predictions()),
            inputs=[],
            outputs=[dist_plot, words_plot, timeline_plot, stats_display, uncertain_display]
        )
    
    return demo


if __name__ == "__main__":
    demo = create_dashboard()
    demo.launch(server_name="0.0.0.0", server_port=7860)
