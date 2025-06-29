# 🔧 Scripts Directory

This directory contains all core executable scripts for the clickbait spoiler classification project.

## 📋 **Purpose & Functionality**

The `scripts/` folder serves as the **main execution hub** containing optimized Python scripts that implement the complete machine learning pipeline from data preprocessing to model evaluation.

## 📄 **Script Components (7 Files)**

### 🔨 **Data Processing Scripts**
- **`preprocess_data.py`** - Core data preprocessing pipeline
  - Converts raw JSONL data to ML-ready features
  - Generates SBERT embeddings (384-dim) for 4 text fields
  - Creates numerical features (5 dimensions)
  - Total output: 1,541 features per sample
  
- **`clean_unicode_auto.py`** - Automated Unicode character cleaning
  - Removes 36+ types of problematic Unicode characters
  - Fixes encoding issues and normalizes text
  - Automatic backup and safe replacement

### 🚀 **Model Training & Evaluation**
- **`train_classifier.py`** - ML model training with cross-validation
  - Trains 3 models: Random Forest, SVM, Logistic Regression
  - 5-fold stratified cross-validation
  - Hyperparameter optimization and model comparison
  
- **`evaluate_classification.py`** - Comprehensive model evaluation
  - Multi-model performance comparison
  - Generates confusion matrices, ROC curves
  - Error analysis and feature importance
  - Saves detailed evaluation reports

### 📊 **Analysis & Visualization**
- **`analyze_processed_data.py`** - Processed data analysis & reporting
  - Statistical analysis of processed features
  - Data quality assessment and validation
  - Generates comprehensive analysis reports

- **`create_advanced_visualizations.py`** - Advanced data visualizations
  - PCA and t-SNE embeddings visualization
  - Task-specific analysis charts
  - Comprehensive comparison plots

### ⚙️ **System Management**
- **`check_system_readiness.py`** - System validation & environment check
  - CUDA availability and GPU memory check
  - Dependencies validation
  - Training time estimation
  - Directory structure verification

## 🚀 **Complete Workflow**

```bash
# 1. System Check
python scripts/check_system_readiness.py

# 2. Data Cleaning  
python scripts/clean_unicode_auto.py

# 3. Data Processing
python scripts/preprocess_data.py

# 4. Data Analysis
python scripts/analyze_processed_data.py

# 5. Advanced Visualizations
python scripts/create_advanced_visualizations.py

# 6. Model Training
python scripts/train_classifier.py

# 7. Model Evaluation
python scripts/evaluate_classification.py
```

## 🎯 **Design Philosophy**

- **Modular**: Each script has a single, well-defined responsibility
- **Sequential**: Scripts build upon each other in logical order
- **Robust**: Error handling and validation at each step
- **Reproducible**: Consistent random seeds and configurations
- **Professional**: Production-ready code with comprehensive logging

## 📈 **Expected Outputs**

- **Preprocessed datasets** in `data/processed/`
- **Trained models** in `models/task2_classification/`
- **Evaluation results** in `results/task2_classification/`
- **Visualizations** in `results/preprocessing_analysis/`
- **Analysis reports** in JSON and CSV formats

---

**🏆 Result**: SVM model achieves **85.5% accuracy** with 0.8152 F1-macro score 