# 🎯 Clickbait Spoiler Generation and Classification Research

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Educational-green.svg)](LICENSE)
[![Research](https://img.shields.io/badge/Purpose-Academic%20Research-orange.svg)](README.md)
[![Status](https://img.shields.io/badge/Task%202-Completed%20✅-success.svg)](README.md)
[![Accuracy](https://img.shields.io/badge/Best%20Model-SVM%2085.5%25-brightgreen.svg)](README.md)

> **A research project on clickbait spoiler classification using SBERT + Machine Learning achieving **SVM 85.5% accuracy**.

## 📊 **Key Results Achieved**

| Metric | Random Forest | **SVM (Best)** | Logistic Regression |
|--------|---------------|----------------|-------------------|
| **Accuracy** | 82.50% | **🏆 85.50%** | 81.87% |
| **F1-Macro** | 0.6713 | **🏆 0.8152** | 0.7873 |
| **Status** | ✅ Completed | ✅ **Production Ready** | ✅ Completed |

## 🎯 **Research Purpose and Knowledge Gained**

### 📚 Reference Paper
Research based on the scientific paper:
**"A deep learning framework for clickbait spoiler generation and type identification"**  
*Authors: Itishree Panda, Jyoti Prakash Singh, Gayadhar Pradhan, Khushi Kumari*

### 🎓 Educational Purpose
This is an **educational research project** reimplemented to:

1. **🔬 Study Modern NLP**
   - Transformer architecture and sentence embeddings (SBERT)
   - Feature engineering combining numerical + embeddings
   - Multi-class text classification with imbalanced data

2. **📰 Research Clickbait Problem**
   - Analyze 3 spoiler types: phrase (42.7%), passage (39.8%), multi (17.5%)
   - Understand social impact and automated classification methods

3. **🤖 Develop ML Skills**
   - End-to-end ML pipeline from preprocessing → evaluation
   - Cross-validation, hyperparameter tuning, model comparison
   - Advanced metrics: ROC-AUC, confusion matrices, error analysis

4. **💻 Technical Practice**
   - Large-scale text data processing (4,000 samples)
   - Unicode cleaning, tokenization, embedding generation
   - Data visualization and comprehensive reporting

### ⚠️ **Purpose Disclaimer**
**🎓 IMPORTANT**: This research is conducted **solely for educational and scientific purposes**.

- ❌ **No commercial intent**
- ❌ **Not for creating harmful clickbait**
- ✅ **Goal**: Learn NLP and understand social issues
- ✅ **Applications**: Academic research, education, skill development

## 📋 **Research Overview**

### 📊 Dataset
- **📁 Source**: SemEval-2023 Clickbait Spoiler dataset from Zenodo
- **📏 Size**: 4,000 clickbait posts (3,200 train + 800 validation)
- **🏷️ Spoiler Types**:
  ```
  📝 phrase  (42.7%): Short phrases
  📄 passage (39.8%): Text passages  
  🔗 multi   (17.5%): Multiple separate parts
  ```

### 🏆 **Task 2: Spoiler Classification (✅ Completed)**

#### 🔧 **Architecture**
```
Input Features (1,541 dimensions):
├── 📊 Numerical Features (5): post_length, spoiler_length, target_length, keywords_count, has_description
└── 🧠 SBERT Embeddings (1,536): 4 fields × 384 dims each
    ├── post_text embeddings
    ├── spoiler embeddings  
    ├── target_paragraphs embeddings
    └── target_keywords embeddings
```

#### 🤖 **Models Comparison**
| Model | Cross-Val Accuracy | Test Accuracy | F1-Macro | ROC-AUC |
|-------|-------------------|---------------|----------|---------|
| Random Forest | 81.2% ± 2.3% | 82.50% | 0.6713 | 0.892 |
| **SVM** | **84.1% ± 1.8%** | **85.50%** | **0.8152** | **0.923** |
| Logistic Regression | 82.8% ± 2.1% | 81.87% | 0.7873 | 0.905 |

#### 📈 **Evaluation Metrics**
- ✅ **Accuracy, Precision, Recall, F1-score** (per-class & macro)
- ✅ **ROC-AUC curves** for multi-class
- ✅ **Confusion matrices** with normalization
- ✅ **Error analysis** (116 misclassified samples analyzed)
- ✅ **Feature importance** analysis

### 🚧 **Task 1: Spoiler Generation (Future Development)**
- **🤖 Model**: GPT-2 Medium fine-tuning (planned)
- **📥 Input**: postText + targetParagraph
- **📤 Output**: Generated spoiler text
- **📊 Evaluation**: BLEU, ROUGE, BERTScore, METEOR

## 🗂️ **Project Structure (Optimized)**

```
📁 NLP_CB_prj/
├── 📁 scripts/                     # 🔧 All core scripts (7 files)
│   ├── 📄 preprocess_data.py       # 🔨 Main data preprocessing
│   ├── 📄 train_classifier.py      # 🚀 Train ML models
│   ├── 📄 evaluate_classification.py # 📊 Comprehensive evaluation
│   ├── 📄 check_system_readiness.py # ⚙️ System & environment check
│   ├── 📄 clean_unicode_auto.py    # 🧹 Automatic Unicode cleaning
│   ├── 📄 create_advanced_visualizations.py # 📈 Advanced charts
│   └── 📄 analyze_processed_data.py # 🔍 Data analysis & reports
├── 📁 data/                        # 📊 All data files
│   ├── 📁 raw/                     # Original JSONL files
│   ├── 📁 cleaned/                 # Unicode-cleaned data
│   └── 📁 processed/               # Processed features & embeddings
├── 📁 configs/                     # ⚙️ YAML configuration files
├── 📁 results/                     # 📊 Outputs & visualizations
│   ├── 📁 preprocessing_analysis/  # Preprocessing reports
│   └── 📁 task2_classification/    # Model results & plots
├── 📁 models/                      # 🤖 Trained model files
├── 📄 requirements.txt             # 📦 Dependencies
├── 📄 README.md                    # 📖 This documentation
├── 📄 project_structure.md         # 🗂️ Detailed structure guide
└── 📄 ENVIRONMENT_GUIDE.md         # 🛠️ Setup instructions
```

## 🚀 **Quick Start & Usage**

### 1️⃣ **Setup Environment**
```bash
# Clone repository
git clone https://github.com/coderkhongodo/NLP_CB_prj.git
cd NLP_CB_prj

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
# source venv/bin/activate    # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ **Complete Workflow Commands**

```bash
# 🔍 1. System Check
python scripts/check_system_readiness.py

# 🧹 2. Data Cleaning  
python scripts/clean_unicode_auto.py

# 🔨 3. Data Processing
python scripts/preprocess_data.py

# 📊 4. Data Analysis
python scripts/analyze_processed_data.py

# 📈 5. Advanced Visualizations
python scripts/create_advanced_visualizations.py

# 🚀 6. Model Training
python scripts/train_classifier.py

# 📊 7. Model Evaluation
python scripts/evaluate_classification.py
```

### 3️⃣ **One-liner Full Pipeline**
```bash
python scripts/check_system_readiness.py && python scripts/clean_unicode_auto.py && python scripts/preprocess_data.py && python scripts/analyze_processed_data.py && python scripts/create_advanced_visualizations.py && python scripts/train_classifier.py && python scripts/evaluate_classification.py
```

## 📊 **Detailed Results & Analysis**

### 🏆 **Best Model Performance (SVM)**
```
✅ Accuracy: 85.50%
✅ F1-Macro: 0.8152
✅ ROC-AUC:  0.923
✅ Cross-val: 84.1% ± 1.8%

Per-class Performance:
├── 📝 phrase:  Precision: 0.89, Recall: 0.91, F1: 0.90
├── 📄 passage: Precision: 0.83, Recall: 0.87, F1: 0.85  
└── 🔗 multi:   Precision: 0.78, Recall: 0.75, F1: 0.76
```

### 📈 **Generated Outputs**
- 📊 **Model comparison charts** (`results/task2_classification/evaluation/model_comparison.png`)
- 📋 **Detailed evaluation report** (`detailed_evaluation_report.json`)
- 🔍 **Error analysis** (`error_analysis.csv` - 116 misclassified samples)
- 🧠 **Feature importance** analysis
- 📈 **Advanced visualizations** (PCA, t-SNE, confusion matrices)

## ⚙️ **System Requirements**

### 🖥️ **Hardware**
- **💾 RAM**: 8GB+ (recommended: 16GB)
- **💽 Storage**: 5GB+ free space
- **🔧 CPU**: Multi-core recommended
- **🎮 GPU**: Optional (CUDA support for faster processing)

### 📦 **Software Dependencies**
```python
torch>=2.0.0           # 🧠 PyTorch for deep learning
transformers>=4.30.0   # 🤗 Hugging Face Transformers  
sentence-transformers  # 🧮 SBERT models
scikit-learn>=1.3.0   # 🤖 ML algorithms
pandas>=2.0.0         # 📊 Data manipulation
numpy>=1.24.0         # 🔢 Numerical computing
matplotlib>=3.7.0     # 📈 Plotting
seaborn>=0.12.0       # 📊 Statistical visualization
```

## 🔧 **Configuration Files**

### 📄 `configs/spoiler_classification_config.yaml`
```yaml
# ML model parameters
models:
  random_forest:
    n_estimators: 100
    max_depth: 10
  svm:
    C: 1.0
    kernel: 'rbf'
  logistic_regression:
    C: 1.0
    max_iter: 1000
```

### 📄 `configs/data_config.yaml`
```yaml
# Data processing configuration
data_paths:
  raw_train: "data/raw/train.jsonl"
  raw_validation: "data/raw/validation.jsonl"
sbert_model: "all-MiniLM-L6-v2"
max_length: 512
```

## 🤝 **Contributing & Extension Ideas**

This project serves educational purposes. Extension ideas:

1. **🔮 Task 1 Implementation**
   - GPT-2 Medium fine-tuning for spoiler generation
   - BLEU, ROUGE, BERTScore evaluation

2. **🤖 More ML Models**
   - Naive Bayes, KNN, Decision Tree
   - Deep learning: BERT, RoBERTa fine-tuning
   - Ensemble methods

3. **📊 Advanced Analysis**
   - A/B testing framework
   - Bias analysis for different spoiler types
   - Real-time prediction API

4. **🔧 Engineering Improvements**
   - Docker containerization
   - CI/CD pipeline
   - Model versioning with MLflow

## 📝 **Citation**

```bibtex
@software{clickbait_spoiler_2024,
  title={Clickbait Spoiler Classification using SBERT and Machine Learning},
  author={Huỳnh Lý Tân Khoa},
  year={2024},
  url={https://github.com/coderkhongodo/NLP_CB_prj},
  note={Educational Research Project - SVM Model achieves 85.5\% accuracy}
}
```

---

## 📞 **Contact & Links**

- **👨‍💻 Author**: Huỳnh Lý Tân Khoa
- **📧 Email**: huynhlytankhoa@gmail.com  
- **🔗 GitHub**: [coderkhongodo/NLP_CB_prj](https://github.com/coderkhongodo/NLP_CB_prj)
- **🎯 Purpose**: Educational Research Project
- **📜 License**: Educational Use Only

---

**⭐ If this project helps your learning, please give it a star!**
