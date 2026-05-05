import requests
import json

API_BASE = "http://localhost:8000"


def test_health():
    print("\n=== Health Check ===")
    response = requests.get(f"{API_BASE}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200


def test_analyze_single():
    print("\n=== Analyze Single Review ===")
    
    test_texts = [
        "This product is absolutely amazing! I love it so much.",
        "Terrible experience. Would never buy again.",
        "It's okay, nothing special about it.",
        "The service was good but the product arrived damaged.",
        "Best purchase I've ever made! Highly recommend.",
        "Very disappointed with the quality.",
        "Average product, does what it's supposed to do."
    ]
    
    for text in test_texts:
        response = requests.post(
            f"{API_BASE}/analyze",
            json={"text": text, "store_result": True}
        )
        if response.status_code == 200:
            data = response.json()
            print(f"\nText: {text[:50]}...")
            print(f"  Sentiment: {data['sentiment']}")
            print(f"  Confidence: {data['confidence']:.2%}")
            print(f"  Stored: {data['stored']}")
        else:
            print(f"Error: {response.status_code} - {response.text}")


def test_analyze_no_store():
    print("\n=== Analyze Without Storing ===")
    response = requests.post(
        f"{API_BASE}/analyze",
        json={"text": "Testing without storage", "store_result": False}
    )
    if response.status_code == 200:
        data = response.json()
        print(f"Sentiment: {data['sentiment']}")
        print(f"Stored: {data['stored']}")
    else:
        print(f"Error: {response.status_code}")


def test_batch_analyze():
    print("\n=== Batch Analysis ===")
    texts = [
        "Love this product!",
        "Hate it so much",
        "It's fine, nothing special",
        "Amazing quality and fast shipping",
        "Worst purchase ever, complete waste",
        "Decent for the price",
        "Fantastic customer service",
        "Not worth the money",
        "Pretty good overall",
        "Absolutely terrible experience"
    ]
    
    response = requests.post(
        f"{API_BASE}/batch",
        json={"texts": texts, "store_results": True}
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"Total results: {len(data['results'])}")
        print(f"Total time: {data['total_processing_time_ms']:.2f}ms")
        print(f"Avg time: {data['average_processing_time_ms']:.2f}ms")
        
        sentiment_counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
        for result in data['results']:
            sentiment_counts[result['sentiment']] += 1
        
        print(f"\nSentiment breakdown:")
        for sentiment, count in sentiment_counts.items():
            print(f"  {sentiment}: {count}")
    else:
        print(f"Error: {response.status_code} - {response.text}")


def test_statistics():
    print("\n=== Get Statistics ===")
    response = requests.get(f"{API_BASE}/statistics")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Total inferences: {data['total_inferences']}")
        print(f"Average confidence: {data['average_confidence']:.2%}")
        print(f"Avg processing time: {data['average_processing_time_ms']:.2f}ms")
        print(f"\nSentiment distribution:")
        for sentiment, count in data['sentiment_distribution'].items():
            print(f"  {sentiment}: {count}")
    else:
        print(f"Error: {response.status_code}")


def test_get_results():
    print("\n=== Get Recent Results ===")
    response = requests.get(f"{API_BASE}/results?limit=5")
    
    if response.status_code == 200:
        data = response.json()
        results = data['results']
        print(f"Retrieved {len(results)} results")
        
        for i, result in enumerate(results[:3], 1):
            print(f"\n{i}. Text: {result['text'][:50]}...")
            print(f"   Sentiment: {result['sentiment']} ({result['confidence']:.2%})")
    else:
        print(f"Error: {response.status_code}")


def test_clear_results():
    print("\n=== Clear All Results ===")
    response = requests.delete(f"{API_BASE}/results")
    
    if response.status_code == 200:
        print(f"Response: {response.json()}")
        
        stats_response = requests.get(f"{API_BASE}/statistics")
        if stats_response.status_code == 200:
            data = stats_response.json()
            print(f"Total inferences after clear: {data['total_inferences']}")
    else:
        print(f"Error: {response.status_code}")


def test_root():
    print("\n=== Root Endpoint ===")
    response = requests.get(f"{API_BASE}/")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Message: {data['message']}")
        print(f"Version: {data['version']}")
        print("Available endpoints:")
        for name, endpoint in data['endpoints'].items():
            print(f"  - {name}: {endpoint}")
    else:
        print(f"Error: {response.status_code}")


def test_error_handling():
    print("\n=== Error Handling Tests ===")
    
    tests = [
        ("Empty text", {"text": ""}),
        ("Very long text", {"text": "word " * 1000}),
    ]
    
    for name, payload in tests:
        response = requests.post(f"{API_BASE}/analyze", json=payload)
        print(f"\n{name}: Status {response.status_code}")


def run_all_tests():
    print("=" * 50)
    print("SENTIMENT ANALYSIS API TEST SUITE")
    print("=" * 50)
    
    if not test_health():
        print("\nAPI is not healthy. Make sure it's running on port 8000.")
        print("Start with: python main.py api")
        return
    
    test_root()
    test_analyze_single()
    test_analyze_no_store()
    test_batch_analyze()
    test_statistics()
    test_get_results()
    test_error_handling()
    
    print("\n" + "=" * 50)
    print("TESTS COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    run_all_tests()
