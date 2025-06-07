# Clickbait Spoiler Generation and Classification Research

Dự án nghiên cứu về sinh spoiler và phân loại spoiler cho clickbait sử dụng các mô hình học sâu và machine learning.

## Mục đích nghiên cứu và kiến thức đạt được

### Nguồn tham khảo
Nghiên cứu này được thực hiện dựa trên bài báo khoa học:
**"A deep learning framework for clickbait spoiler generation and type identification"** 
*Tác giả: Itishree Panda, Jyoti Prakash Singh, Gayadhar Pradhan, Khushi Kumari*

### Mục đích học tập
Đây là một **bài nghiên cứu học tập** được thực hiện lại nhằm:

1. **Tìm hiểu về các mô hình NLP hiện đại**
   - Hiểu sâu về kiến trúc Transformer và sentence embeddings
   - Nghiên cứu SBERT (Sentence-BERT) cho text classification
   - Khám phá các kỹ thuật feature engineering với embeddings

2. **Nghiên cứu vấn đề clickbait trong truyền thông**
   - Phân tích hiện tượng clickbait và tác động xã hội
   - Hiểu các loại spoiler: phrase (cụm từ), passage (đoạn), multi (nhiều phần)
   - Nghiên cứu phương pháp tự động phân loại spoiler

3. **Phát triển kỹ năng Machine Learning**
   - Thực hành quy trình ML end-to-end từ preprocessing đến evaluation
   - So sánh hiệu suất các thuật toán machine learning
   - Áp dụng các metrics đánh giá phù hợp cho multi-class classification

4. **Kiến thức kỹ thuật đạt được**
   - Xử lý dữ liệu văn bản quy mô lớn với pandas và numpy
   - Kỹ thuật tokenization và text preprocessing
   - Feature engineering kết hợp numerical features và embeddings
   - Cross-validation và hyperparameter tuning
   - Data visualization và phân tích kết quả

### Tuyên bố về mục đích
**⚠️ QUAN TRỌNG**: Nghiên cứu này được thực hiện **chỉ với mục đích học tập và nghiên cứu khoa học**. 

- **Không có mục đích thương mại**
- **Không được sử dụng để tạo clickbait có hại**
- **Mục tiêu**: Hiểu biết sâu hơn về NLP và các vấn đề xã hội liên quan
- **Ứng dụng**: Nghiên cứu học thuật, giáo dục, và phát triển kỹ năng kỹ thuật

## Tổng quan nghiên cứu

### Bộ dữ liệu
- **Nguồn**: SemEval-2023 Clickbait Spoiler dataset từ Zenodo
- **Kích thước**: 4.000 bài đăng clickbait
- **Phân chia**: 3.200 mẫu huấn luyện, 800 mẫu validation
- **Loại spoiler**: 
  - **phrase** (42.7%): Cụm từ ngắn
  - **passage** (39.8%): Đoạn văn bản
  - **multi** (17.5%): Nhiều phần riêng biệt

### Phương pháp đã implement

#### Task 2: Phân loại loại spoiler (Đã hoàn thành)
- **Mô hình**: SBERT + Machine Learning Classifiers
- **Đặc trưng**: 
  - 5 numerical features (post_length, spoiler_length, target_length, keywords_count, has_description)
  - 4 SBERT embeddings (384-dim mỗi field): post_text, spoiler, target_paragraphs, target_keywords
  - **Tổng**: 1541 features (5 + 384×4 = 1541)
- **Bộ phân loại so sánh**: 3 models
  - Random Forest (RF)
  - Support Vector Machine (SVM)
  - Logistic Regression (LR)
- **Đánh giá**: Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix

#### Task 1: Tạo spoiler (Sẽ cập nhật sau)
- **Mô hình**: GPT-2 Medium (dự định implement)
- **Đầu vào**: postText + targetParagraph  
- **Đầu ra**: Văn bản spoiler
- **Đánh giá**: BLEU, ROUGE, BERTScore, METEOR

## Cấu trúc Project

```
clickbait-spoiler-research/
├── data/                           # Dữ liệu
│   ├── raw/                       # Dữ liệu gốc (train.jsonl, validation.jsonl)
│   ├── cleaned/                   # Dữ liệu đã clean unicode
│   ├── processed/                 # Dữ liệu đã xử lý
│   └── backup/                    # Backup dữ liệu
├── src/                           # Mã nguồn chính
│   ├── data/                      # Data processing modules
│   ├── models/                    # Model definitions
│   ├── training/                  # Training scripts
│   ├── evaluation/                # Evaluation scripts
│   └── utils/                     # Utility functions
├── scripts/                       # Scripts thực thi
│   ├── data_exploration.py        # Khám phá dữ liệu
│   ├── preprocess_data.py         # Tiền xử lý dữ liệu
│   ├── check_environment.py       # Kiểm tra môi trường
│   └── setup_project.py           # Thiết lập project
├── configs/                       # Configuration files
│   ├── data_config.yaml          # Cấu hình dữ liệu
│   └── spoiler_classification_config.yaml  # Cấu hình phân loại
├── results/                       # Kết quả thí nghiệm
│   ├── data_exploration/          # Biểu đồ phân tích dữ liệu
│   ├── task2_classification/      # Kết quả phân loại
│   └── preprocessing_analysis/    # Phân tích tiền xử lý
├── models/                        # Mô hình đã huấn luyện
├── logs/                          # Log files
└── venv/                          # Virtual environment
```

## Cài đặt và thiết lập

### 1. Yêu cầu hệ thống
- Python 3.8+
- RAM: 8GB+ (recommended 16GB)
- Storage: 5GB+ free space

### 2. Thiết lập môi trường
```bash
# Tạo virtual environment
python -m venv venv

# Kích hoạt (Windows)
.\venv\Scripts\Activate.ps1

# Cài đặt dependencies
pip install -r requirements.txt
```

### 3. Kiểm tra môi trường
```bash
python scripts/check_environment.py
```

## Sử dụng

### 1. Khám phá dữ liệu
```bash
python scripts/data_exploration.py
```
- Tạo các biểu đồ phân tích trong `results/data_exploration/`
- Thống kê chi tiết về bộ dữ liệu

### 2. Tiền xử lý dữ liệu
```bash
python scripts/preprocess_data.py
```
- Clean unicode characters
- Tạo numerical features
- Sinh SBERT embeddings
- Lưu dữ liệu processed trong `data/processed/`

### 3. Huấn luyện mô hình phân loại
```bash
python src/training/train_classifier.py
```
- Huấn luyện 3 models với cross-validation
- Lưu models trong `models/task2_classification/`
- Tạo báo cáo kết quả trong `results/task2_classification/`

### 4. Đánh giá mô hình
```bash
python src/evaluation/evaluate_classification.py
```
- So sánh hiệu suất các models
- Tạo confusion matrices và biểu đồ
- Phân tích feature importance

## Kết quả hiện tại

### Phân loại spoiler (Task 2)
✅ **Đã hoàn thành và đánh giá**
- So sánh 3 bộ phân loại: Random Forest, SVM, Logistic Regression
- Features: 1541-dimensional (5 numerical + 1536 embeddings)
- Evaluation metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC
- Visualizations: Confusion matrices, feature importance, model comparison

### Tạo spoiler (Task 1)
🔄 **Đang phát triển**
- Sẽ implement GPT-2 Medium fine-tuning
- Text generation với BLEU, ROUGE evaluation

## Files cấu hình

### `configs/spoiler_classification_config.yaml`
- Cấu hình 3 ML models (RF, SVM, LR)
- Parameters cho training và evaluation
- Feature configuration

### `configs/data_config.yaml`
- Đường dẫn dữ liệu
- Text processing parameters
- SBERT model configuration

## Dependencies chính

```txt
torch>=2.0.0           # PyTorch cho deep learning
transformers>=4.30.0   # Hugging Face Transformers
sentence-transformers  # SBERT models
scikit-learn>=1.3.0   # Machine learning algorithms
pandas>=2.0.0         # Data manipulation
numpy>=1.24.0         # Numerical computing
matplotlib>=3.7.0     # Plotting
seaborn>=0.12.0       # Statistical visualization
```

Xem file `requirements.txt` để biết danh sách đầy đủ.

## Cách đóng góp

Dự án này phục vụ mục đích học tập. Nếu muốn mở rộng:
1. Implement Task 1 (spoiler generation) với GPT-2
2. Thêm các bộ phân loại khác (Naive Bayes, KNN, Decision Tree, etc.)
3. Thử nghiệm với các pre-trained models khác
4. Cải thiện feature engineering

## Trích dẫn

```bibtex
@article{clickbait_spoiler_2024,
  title={Clickbait Spoiler Generation and Classification using GPT-2 and SBERT: A Deep Learning Study},
  author={Huỳnh Lý Tân Khoa},
  email={huynhlytankhoa@gmail.com},
  type={Academic Research Project}
}
```

---

**📧 Contact**: huynhlytankhoa@gmail.com  
**🎯 Purpose**: Educational Research Project  
