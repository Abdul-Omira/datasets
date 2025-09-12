#!/usr/bin/env python3
"""
Test script to create a small sample Wikipedia dataset for testing the pipeline.
"""

import json
import logging
from datetime import datetime
from datasets import Dataset, Features, Sequence, Value

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data():
    """Create sample Wikipedia data for testing."""
    sample_articles = [
        {
            "id": "12345",
            "title": "Artificial Intelligence",
            "url": "https://en.wikipedia.org/wiki/Artificial_Intelligence",
            "text": "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of \"intelligent agents\": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals.",
            "redirects": ["AI", "Machine Intelligence", "Artificial intelligence", "Computer Intelligence"],
            "pageviews_2023": 2450000,
            "timestamp": "2023-12-01T10:30:00Z"
        },
        {
            "id": "67890", 
            "title": "Machine Learning",
            "url": "https://en.wikipedia.org/wiki/Machine_Learning",
            "text": "Machine learning (ML) is a field of inquiry devoted to understanding and building methods that 'learn', that is, methods that leverage data to improve performance on some set of tasks. It is seen as a part of artificial intelligence.",
            "redirects": ["ML", "Statistical Learning", "Machine learning"],
            "pageviews_2023": 1850000,
            "timestamp": "2023-12-01T09:15:00Z"
        },
        {
            "id": "11111",
            "title": "Python (programming language)",
            "url": "https://en.wikipedia.org/wiki/Python_(programming_language)",
            "text": "Python is a high-level, interpreted, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. Python is dynamically-typed and garbage-collected.",
            "redirects": ["Python programming", "Python language", "Python programming language"],
            "pageviews_2023": 5200000,
            "timestamp": "2023-11-28T14:22:00Z"
        }
    ]
    
    return sample_articles

def test_dataset_creation_and_upload():
    """Test the complete pipeline with sample data."""
    logger.info("Creating sample dataset for testing...")
    
    # Create sample data
    data = create_sample_data()
    
    # Save sample data to file
    with open('sample_wikipedia_data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info("Sample data saved to sample_wikipedia_data.json")
    
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
    
    logger.info(f"Created test dataset with {len(dataset)} articles")
    
    # Print info about the dataset
    print("\n" + "="*60)
    print("TEST DATASET SUMMARY")
    print("="*60)
    print(f"Number of articles: {len(dataset)}")
    print(f"Features: {list(dataset.features.keys())}")
    print(f"Dataset size: {dataset.num_rows} rows")
    
    # Show samples
    print("\nSAMPLE ARTICLES:")
    print("-" * 40)
    for i, article in enumerate(dataset):
        print(f"{i+1}. {article['title']}")
        print(f"   Redirects: {len(article['redirects'])} ({', '.join(article['redirects'][:2])}...)")
        print(f"   Pageviews: {article['pageviews_2023']:,}")
        print()
    
    # Test upload (uncomment to actually upload)
    # logger.info("Uploading test dataset...")
    # dataset.push_to_hub(
    #     "wikipedia-2023-redirects-test",
    #     private=True,
    #     commit_message="Test upload of Wikipedia redirects dataset"
    # )
    
    print("✅ Test completed successfully!")
    print("To upload to Hub, uncomment the upload lines in the script")
    
    return dataset

if __name__ == "__main__":
    test_dataset_creation_and_upload()