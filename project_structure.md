# Clickbait Spoiler Generation and Classification Project Structure

## Cấu trúc thư mục tổng quan

```
clickbait-spoiler-research/
│
├── data/                           # Thư mục chứa dữ liệu
│   ├── raw/                        # Dữ liệu thô từ SemEval-2023
│   ├── processed/                  # Dữ liệu đã xử lý
│   └── cleaned/                    # Dữ liệu đã làm sạch Unicode
│
├── scripts/                        # Scripts xử lý và thí nghiệm (7 files)
│   ├── preprocess_data.py          # Tiền xử lý dữ liệu chính
│   ├── train_classifier.py         # Huấn luyện mô hình phân loại
│   ├── evaluate_classification.py  # Đánh giá mô hình phân loại
│   ├── check_system_readiness.py   # Kiểm tra system cho training
│   ├── clean_unicode_auto.py       # Tự động làm sạch Unicode
│   ├── create_advanced_visualizations.py  # Tạo visualizations chi tiết
│   └── analyze_processed_data.py   # Phân tích dữ liệu đã xử lý
│
├── configs/                        # File cấu hình YAML
├── results/                        # Kết quả thí nghiệm và visualizations
├── models/                         # Mô hình đã huấn luyện
├── venv/                          # Virtual environment
├── requirements.txt               # Dependencies
├── README.md                      # Hướng dẫn project
├── ENVIRONMENT_GUIDE.md           # Hướng dẫn setup môi trường
└── .gitignore                     # Git ignore rules
```

## Mô tả chi tiết các Scripts chính

### 1. Data Processing & Preprocessing
- **`preprocess_data.py`**: Script chính cho tiền xử lý dữ liệu
  - Task 1: Tạo features cho GPT-2 spoiler generation
  - Task 2: Tạo features + embeddings cho SBERT classification
  - Tokenization, embedding generation với SBERT

- **`clean_unicode_auto.py`**: Tự động làm sạch ký tự Unicode problematic
  - Thay thế 36,000+ ký tự Unicode ambiguous
  - Automatic backup và replacement

### 2. Training & Evaluation
- **`train_classifier.py`**: Huấn luyện mô hình phân loại Task 2
  - SBERT embeddings + ML classifiers (RF, SVM, LR)
  - Cross-validation, feature importance analysis
  - WandB integration cho experiment tracking

- **`evaluate_classification.py`**: Đánh giá comprehensive cho Task 2
  - Multi-model comparison và metrics
  - Confusion matrices, ROC curves
  - Error analysis và detailed reports

### 3. Analysis & Visualization
- **`analyze_processed_data.py`**: Phân tích dữ liệu đã processed
  - Task 1 và Task 2 data analysis
  - Length distributions, embeddings analysis
  - Automated report generation

- **`create_advanced_visualizations.py`**: Tạo visualizations toàn diện
  - Task-specific analysis charts
  - PCA/t-SNE embeddings visualization
  - Comprehensive comparison plots

### 4. System Management
- **`check_system_readiness.py`**: Kiểm tra system cho training
  - CUDA availability và GPU memory
  - Dependencies validation
  - Training time estimation
  - Directory structure verification

## Workflow đề xuất

1. **Setup**: `check_system_readiness.py`
2. **Data Cleaning**: `clean_unicode_auto.py`
3. **Preprocessing**: `preprocess_data.py`
4. **Analysis**: `analyze_processed_data.py`
5. **Visualization**: `create_advanced_visualizations.py`
6. **Training**: `train_classifier.py`
7. **Evaluation**: `evaluate_classification.py` 