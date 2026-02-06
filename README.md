# Text Stack Ensemble with Vision-Language Models

## Overview

This notebook implements an advanced ensemble learning approach for price prediction in the Amazon ML Challenge 2025. The solution combines multiple specialized models with vision-language model (VLM) predictions to create a robust and accurate price estimator.

## Architecture

### Three-Expert Ensemble System

The notebook uses a stacked ensemble architecture with three diverse base models, each serving as a specialist in different aspects of price prediction:

#### 1. **XGBoost - Keyword Expert**
- **Features**: TF-IDF (1,2-grams) + Engineered Text Features
- **Purpose**: Captures keyword patterns and term frequencies that directly influence pricing
- **Engineered Features**:
  - Text length
  - Word count
  - Digit count (for product codes, specifications, etc.)

#### 2. **LightGBM - Semantic & Market Expert**
- **Features**: Transformer Embeddings + Engineered Features + KNN Market Features
- **Transformer Models Used**:
  - sentence-transformers/all-MiniLM-L12-v2
  - BAAI/bge-base-en-v1.5 (optimized for semantic similarity)
  - sentence-transformers/all-distilroberta-v1
  - sentence-transformers/paraphrase-mpnet-base-v2
- **Purpose**: Understands semantic meaning and market patterns through neighbor analysis
- **KNN Features**: Mean, median, and std of 10 nearest neighbors' prices (from BGE embeddings)

#### 3. **DenseNet (1D CNN) - Deep Learning Specialist**
- **Features**: Text Sequences → Learned Embeddings → Conv1D Features
- **Architecture**:
  - Embedding Layer (vocab_size: 30K, embedding_dim: 128)
  - Conv1D (64 filters, kernel_size: 3)
  - Global Max Pooling
  - Dense layers (64 units) with Dropout (0.5)
  - Output regression layer
- **Purpose**: Learns complex non-linear patterns through deep neural network

### Meta-Model

A **Ridge Regression** meta-model (α=1.0) blends predictions from all three base models in a weighted manner:
- Learns optimal weights for each expert during cross-validation
- Creates final predictions via weighted averaging

### Vision-Language Models Integration

**Florence & CLIP Models** provide additional predictive signal:
- Vision-language embeddings capture visual-semantic relationships
- Predictions from Florence and CLIP are generated independently
- **These VLM predictions are added to the text stack predictions** for final submission
- Combines multimodal information (text semantics + visual features) for improved accuracy

## Pipeline Phases

### Phase 1: Feature Engineering
- Generate engineered text features (length, word count, digits)
- Create transformer embeddings from 4 different models
- Compute KNN features based on market similarity

### Phase 2: Cross-Validation Training
- 5-fold stratified cross-validation
- Each base model trained and evaluated on fold validation sets
- Out-of-fold (OOF) predictions collected for meta-model training

### Phase 3: Meta-Model Training
- Ridge regression trained on OOF predictions from base models
- Optimal blending weights determined

### Phase 4: Final Predictions
- All three base models retrained on 100% of training data
- Predictions generated on test set
- Meta-model makes final blended predictions
- VLM predictions added for enhanced accuracy

## Usage

### Requirements
```
pandas
numpy
scikit-learn
xgboost
lightgbm
tensorflow
sentence-transformers
faiss
```

### Running the Notebook

1. **Setup Configuration**: Modify the `CFG` class for:
   - Data paths
   - Number of cross-validation folds
   - KNN neighbors count
   - CNN architecture parameters

2. **Execute Phases Sequentially**:
   - Phase 1: Feature Engineering
   - Phase 2: Cross-Validation (obtains OOF predictions)
   - Phase 3: Meta-Model Training
   - Phase 4: Final Predictions & Submission

3. **Output**: `submission1.csv` with predicted prices for test set

## Key Features

✅ **Ensemble Diversity**: Three completely different model types reduce variance  
✅ **Cross-Validation**: Proper out-of-fold evaluation prevents overfitting  
✅ **Feature Rich**: Combines keyword patterns, semantic embeddings, and learned representations  
✅ **GPU Acceleration**: XGBoost and LightGBM configured for GPU training  
✅ **Market Awareness**: KNN features leverage similar items' prices  
✅ **Multimodal**: Incorporates both text and visual features via VLMs  
✅ **Robust Blending**: Meta-model learns optimal combination strategy  

## Performance Metrics

- **Evaluation Metric**: SMAPE (Symmetric Mean Absolute Percentage Error)
- OOF SMAPE tracked across all three models
- Final ensemble SMAPE reported after meta-model training

## Notes

- Negative predictions clipped to 0 (no negative prices in submission)
- GPU FAISS index used for KNN if available, falls back to CPU
- Early stopping applied during CNN training to prevent overfitting
- Combined embeddings saved to `combined_embeddings.npy` for efficiency


**Note: Combined Embeddings not added to github due to large file size**