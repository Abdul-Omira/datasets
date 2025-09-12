#!/usr/bin/env python3
"""
Simple upload script for pre-processed Wikipedia 2023 redirects data.

Use this if you already have processed data in the correct format.
"""

import argparse
import json
import logging
from datasets import Dataset, Features, Sequence, Value

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_dataset_card() -> str:
    """Create a comprehensive dataset card."""
    return """---
license: cc-by-sa-3.0
task_categories:
- text-retrieval
- question-answering
language:
- en
tags:
- wikipedia
- redirects
- pageviews
- knowledge-base
size_categories:
- 1M<n<10M
---

# Wikipedia 2023 Redirects Dataset

## Dataset Description

This dataset contains canonical Wikipedia articles enriched with redirect aliases and 2023 pageview statistics. It's designed to support RAG (Retrieval-Augmented Generation) systems and information retrieval tasks by providing:

- **Redirect resolution**: Each article includes all redirect pages that point to it
- **Popularity signals**: 2023 pageview counts aggregated across the canonical page and its redirects
- **Clean text content**: Article text with basic preprocessing applied

## Features

- `id`: Wikipedia page ID (string)
- `title`: Canonical article title (string)  
- `url`: Wikipedia URL (string)
- `text`: Article content (string)
- `redirects`: List of redirect aliases pointing to this article (list of strings)
- `pageviews_2023`: Total 2023 pageviews for this article and its redirects (int32)
- `timestamp`: Last modification timestamp (string)

## Usage

```python
from datasets import load_dataset

# Load the dataset
ds = load_dataset("wikipedia-2023-redirects")

# Example: Find articles with high pageviews
popular_articles = ds.filter(lambda x: x['pageviews_2023'] > 1000000)

# Example: Articles with many redirects (good for query expansion)
redirect_rich = ds.filter(lambda x: len(x['redirects']) > 10)

print(f"Title: {ds[0]['title']}")
print(f"Redirects: {ds[0]['redirects'][:5]}")  # First 5 redirects
print(f"2023 Pageviews: {ds[0]['pageviews_2023']:,}")
```

## Data Sources

- **Wikipedia articles**: English Wikipedia XML dumps from Wikimedia
- **Redirects**: Wikipedia redirect table from database dumps  
- **Pageviews**: 2023 pageview statistics from Wikimedia

## Licensing

- **Wikipedia content**: Licensed under CC BY-SA 3.0 (Creative Commons Attribution-ShareAlike)
- **Pageview data**: Public domain
- **This dataset**: CC BY-SA 3.0

## Citation

If you use this dataset, please cite:

```bibtex
@misc{wikipedia2023redirects,
  title={Wikipedia 2023 Redirects Dataset},
  author={Abdul-Omira, Abdulwahab},
  year={2023},
  howpublished={Hugging Face Datasets},
  url={https://huggingface.co/datasets/wikipedia-2023-redirects}
}
```

## Acknowledgements

- Wikimedia Foundation for providing Wikipedia dumps and pageview data
- Hugging Face for the datasets infrastructure
"""

def upload_dataset(data_file: str, dataset_name: str, private: bool = False):
    """Upload pre-processed data to Hugging Face Hub."""
    
    logger.info(f"Loading data from {data_file}")
    
    # Load data
    with open(data_file, 'r', encoding='utf-8') as f:
        if data_file.endswith('.json'):
            data = json.load(f)
        elif data_file.endswith('.jsonl'):
            data = [json.loads(line) for line in f]
        else:
            raise ValueError("Data file must be .json or .jsonl")
    
    logger.info(f"Loaded {len(data)} articles")
    
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
    
    logger.info("Created Hugging Face Dataset")
    
    # Upload to Hub
    logger.info(f"Uploading to {dataset_name}...")
    dataset.push_to_hub(
        dataset_name,
        private=private,
        commit_message="Add Wikipedia 2023 redirects dataset with pageviews"
    )
    
    logger.info(f"✅ Successfully uploaded to https://huggingface.co/datasets/{dataset_name}")
    
    # Print sample
    sample = dataset[0]
    print(f"\nSample article:")
    print(f"Title: {sample['title']}")
    print(f"Redirects: {len(sample['redirects'])} aliases")
    print(f"Pageviews: {sample['pageviews_2023']:,}")

def main():
    parser = argparse.ArgumentParser(description="Upload pre-processed Wikipedia data to Hub")
    parser.add_argument("data_file", help="Path to JSON/JSONL file with processed data")
    parser.add_argument("--dataset_name", default="wikipedia-2023-redirects", 
                       help="Dataset name for Hugging Face Hub")
    parser.add_argument("--private", action="store_true", help="Make dataset private")
    
    args = parser.parse_args()
    
    upload_dataset(args.data_file, args.dataset_name, args.private)

if __name__ == "__main__":
    main()