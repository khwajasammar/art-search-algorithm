# Visual Art Search Algorithm — CNN-Based Similarity Model

This repository contains a deep-learning model I built to search and compare digital artwork by visual similarity.  
It powers my art-platform work and is designed for fast, scalable embedding generation.

## Features
- CNN-based feature extractor (PyTorch/TensorFlow)
- Computes image embeddings for similarity search
- Cosine-similarity ranking
- Easy to integrate into web or mobile apps
- Clean modular design (`src/NEWMODEL.py`)

## Model Overview
The model takes an input image → preprocesses it → generates an embedding → compares that embedding against a vector database to surface the closest matches.
