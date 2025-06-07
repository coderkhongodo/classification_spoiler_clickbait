# Clickbait Spoiler Generation and Classification Project Structure

## Cấu trúc thư mục tổng quan

```
clickbait-spoiler-research/
│
├── data/                           # Thư mục chứa dữ liệu
│   ├── raw/                        # Dữ liệu thô từ SemEval-2023
│   ├── processed/                  # Dữ liệu đã xử lý
│   └── splits/                     # Dữ liệu đã chia train/test
│
├── src/                            # Mã nguồn chính
│   ├── data/                       # Xử lý dữ liệu
│   ├── models/                     # Định nghĩa mô hình
│   ├── training/                   # Scripts huấn luyện
│   ├── evaluation/                 # Scripts đánh giá
│   └── utils/                      # Các utility functions
│
├── configs/                        # File cấu hình
├── notebooks/                      # Jupyter notebooks để EDA và thử nghiệm
├── scripts/                        # Scripts chạy thí nghiệm
├── results/                        # Kết quả thí nghiệm
├── models/                         # Mô hình đã huấn luyện
├── logs/                           # Log files
└── requirements.txt                # Dependencies
```

## Mô tả chi tiết từng thành phần

### 1. Data Processing (`src/data/`)
- `data_loader.py`: Load và parse dữ liệu SemEval-2023
- `preprocessor.py`: Tiền xử lý văn bản
- `dataset.py`: Tạo PyTorch Dataset classes
- `data_splitter.py`: Chia dữ liệu train/test

### 2. Models (`src/models/`)
- `gpt2_spoiler_generator.py`: Mô hình GPT-2 cho tạo spoiler
- `sbert_classifier.py`: Mô hình SBERT + ML classifiers
- `base_model.py`: Abstract base class cho các mô hình

### 3. Training (`src/training/`)
- `spoiler_generator_trainer.py`: Huấn luyện mô hình tạo spoiler
- `spoiler_classifier_trainer.py`: Huấn luyện mô hình phân loại
- `trainer_utils.py`: Utility functions cho training

### 4. Evaluation (`src/evaluation/`)
- `generation_metrics.py`: BLEU, ROUGE, BERTScore, METEOR
- `classification_metrics.py`: Accuracy, Precision, Recall, F1, MCC
- `evaluator.py`: Main evaluation class

### 5. Configuration (`configs/`)
- `spoiler_generation_config.yaml`: Cấu hình cho tạo spoiler
- `spoiler_classification_config.yaml`: Cấu hình cho phân loại
- `data_config.yaml`: Cấu hình xử lý dữ liệu

### 6. Scripts (`scripts/`)
- `run_spoiler_generation.py`: Chạy thí nghiệm tạo spoiler
- `run_spoiler_classification.py`: Chạy thí nghiệm phân loại
- `evaluate_models.py`: Đánh giá các mô hình

### 7. Notebooks (`notebooks/`)
- `01_data_exploration.ipynb`: Khám phá dữ liệu
- `02_spoiler_generation_experiments.ipynb`: Thí nghiệm tạo spoiler
- `03_spoiler_classification_experiments.ipynb`: Thí nghiệm phân loại
- `04_results_analysis.ipynb`: Phân tích kết quả 