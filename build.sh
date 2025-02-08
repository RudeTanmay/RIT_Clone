#!/bin/bash
# Install dependencies
pip install -r requirements.txt

# Create NLTK data directory
mkdir -p nltk_data

# Download NLTK data
python -m nltk.downloader -d nltk_data punkt
python -m nltk.downloader -d nltk_data wordnet
