import argparse
import sys
import os

sys.path.insert(0, os.getcwd())


def main():
    parser = argparse.ArgumentParser(
        description="Customer Sentiment Analysis Dashboard - AI-powered sentiment analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    eval_parser = subparsers.add_parser('eval', help='Evaluate model and optionally seed database')
    eval_parser.add_argument('--num_samples', type=int, default=500, help='Number of samples to evaluate')
    eval_parser.add_argument('--full', action='store_true', help='Use full test set (overrides --num_samples)')
    eval_parser.add_argument('--store', action='store_true', help='Store results in database')
    eval_parser.add_argument('--clear', action='store_true', help='Clear database before storing')
    
    api_parser = subparsers.add_parser('api', help='Start FastAPI server')
    api_parser.add_argument('--port', type=int, default=8000, help='API port')
    
    dash_parser = subparsers.add_parser('dashboard', help='Start Gradio dashboard')
    dash_parser.add_argument('--port', type=int, default=7860, help='Dashboard port')
    
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a single review')
    analyze_parser.add_argument('--text', type=str, required=True, help='Review text to analyze')
    
    demo_parser = subparsers.add_parser('demo', help='Start simple Gradio demo')
    demo_parser.add_argument('--port', type=int, default=7861, help='Demo port')
    
    args = parser.parse_args()
    
    if args.command == 'eval':
        from evaluate import main as eval_main
        eval_main(args)
    
    elif args.command == 'api':
        from src.api.app import app
        import uvicorn
        print(f"Starting API server on port {args.port}...")
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    
    elif args.command == 'dashboard':
        from src.dashboard.dashboard import create_dashboard
        print(f"Starting dashboard on port {args.port}...")
        demo = create_dashboard()
        demo.launch(server_name="0.0.0.0", server_port=args.port)
    
    elif args.command == 'analyze':
        from src.models import SentimentClassifier
        from config.settings import MODEL_NAME
        clf = SentimentClassifier(MODEL_NAME)
        result = clf.predict(args.text, return_probs=True)
        print(f"\nText: {args.text}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.2%}")
    
    elif args.command == 'demo':
        import gradio as gr
        from src.models import SentimentClassifier
        from src.db import Database
        from config.settings import MODEL_NAME, DB_PATH
        
        db = Database(DB_PATH)
        clf = SentimentClassifier(MODEL_NAME)
        
        def analyze(text):
            result = clf.predict(text, return_probs=True)
            db.add_result(
                text=text,
                sentiment=result['sentiment'],
                confidence=result['confidence'],
                positive_prob=result.get('probabilities', {}).get('Positive', 0),
                negative_prob=result.get('probabilities', {}).get('Negative', 0),
                neutral_prob=result.get('probabilities', {}).get('Neutral', 0)
            )
            return f"Sentiment: {result['sentiment']}\nConfidence: {result['confidence']:.2%}"
        
        print(f"Starting demo on port {args.port}...")
        demo = gr.Interface(
            fn=analyze,
            inputs=gr.Textbox(label="Enter a review", lines=5),
            outputs=gr.Textbox(label="Result"),
            title="Sentiment Analysis Demo"
        )
        demo.launch(server_name="0.0.0.0", server_port=args.port)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
