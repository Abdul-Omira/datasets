#!/usr/bin/env python3
"""
Quick validation test for Wikipedia dataset pipeline.
"""

import json
from datasets import Dataset, Features, Sequence, Value

def test_pipeline():
    """Test the dataset creation pipeline."""
    print("🧪 Testing Wikipedia Dataset Pipeline...")

    # Sample data
    data = [
        {
            "id": "12345",
            "title": "Artificial Intelligence",
            "url": "https://en.wikipedia.org/wiki/Artificial_Intelligence",
            "text": "Artificial intelligence (AI) is intelligence demonstrated by machines...",
            "redirects": ["AI", "Machine Intelligence", "Artificial intelligence"],
            "pageviews_2023": 2450000,
            "timestamp": "2023-12-01T10:30:00Z"
        },
        {
            "id": "67890",
            "title": "Machine Learning",
            "url": "https://en.wikipedia.org/wiki/Machine_Learning",
            "text": "Machine learning (ML) is a field of inquiry devoted to understanding...",
            "redirects": ["ML", "Statistical Learning"],
            "pageviews_2023": 1850000,
            "timestamp": "2023-12-01T09:15:00Z"
        }
    ]

    # Define schema
    features = Features({
        'id': Value('string'),
        'title': Value('string'),
        'url': Value('string'),
        'text': Value('string'),
        'redirects': Sequence(Value('string')),
        'pageviews_2023': Value('int32'),
        'timestamp': Value('string'),
    })

    # Create dataset
    dataset = Dataset.from_list(data, features=features)

    print("✅ Dataset created successfully!")
    print(f"   - {len(dataset)} articles")
    print(f"   - Features: {list(dataset.features.keys())}")

    # Test data access
    sample = dataset[0]
    print("\n📄 Sample Article:")
    print(f"   Title: {sample['title']}")
    print(f"   Redirects: {len(sample['redirects'])} aliases")
    print(f"   Pageviews: {sample['pageviews_2023']:,}")

    # Save sample data
    with open('sample_wikipedia_data.json', 'w') as f:
        json.dump(data, f, indent=2)

    print("💾 Sample data saved to sample_wikipedia_data.json")

    return dataset

if __name__ == "__main__":
    test_pipeline()