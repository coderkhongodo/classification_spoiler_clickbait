# Clickbait Spoiler Generation and Classification Research

Nghiên cứu này tập trung vào hai nhiệm vụ chính:
1. **Tạo spoiler cho clickbait** sử dụng mô hình GPT-2 Medium
2. **Phân loại loại spoiler** sử dụng SBERT và các bộ phân loại máy học

## Tổng quan

### Bộ dữ liệu
- **Nguồn**: SemEval-2023 dataset từ Zenodo
- **Kích thước**: 4.000 bài đăng clickbait
- **Phân chia**: 3.200 mẫu huấn luyện, 800 mẫu kiểm tra
- **Loại spoiler**: phrase (cụm từ), passage (đoạn), multipart (nhiều phần)

### Phương pháp

#### 1. Tạo spoiler (Spoiler Generation)
- **Mô hình**: GPT-2 Medium được fine-tuned
- **Đầu vào**: postText + targetParagraph
- **Đầu ra**: Văn bản spoiler
- **Đánh giá**: BLEU, ROUGE, BERTScore, METEOR

#### 2. Phân loại loại spoiler (Spoiler Type Classification)
- **Mô hình**: SBERT + ML Classifiers
- **Đặc trưng**: Embedding từ 5 cột văn bản (kích thước 1920)
- **Bộ phân loại**: NB, KNN, DT, AdaBoost, LR, GB, RF, SVM
- **Đánh giá**: Accuracy, Precision, Recall, F1-score, MCC

## Cấu trúc Project

```
clickbait-spoiler-research/
├── data/                    # Dữ liệu
├── src/                     # Mã nguồn
├── configs/                 # Cấu hình
├── notebooks/               # Jupyter notebooks
├── scripts/                 # Scripts thực thi
├── results/                 # Kết quả thí nghiệm
├── models/                  # Mô hình đã huấn luyện
└── logs/                    # Log files
```

## Cài đặt

1. **Clone repository**:
```bash
git clone <repository-url>
cd clickbait-spoiler-research
```

2. **Cài đặt dependencies**:
```bash
pip install -r requirements.txt
```

3. **Tải xuống pre-trained models**:
```bash
python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('gpt2-medium'); AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')"
```

## Sử dụng

### 1. Chuẩn bị dữ liệu
```bash
python scripts/prepare_data.py --config configs/data_config.yaml
```

### 2. Huấn luyện mô hình tạo spoiler
```bash
python scripts/run_spoiler_generation.py --config configs/spoiler_generation_config.yaml
```

### 3. Huấn luyện mô hình phân loại
```bash
python scripts/run_spoiler_classification.py --config configs/spoiler_classification_config.yaml
```

### 4. Đánh giá mô hình
```bash
python scripts/evaluate_models.py
```

## Kết quả mong đợi

### Tạo spoiler
- GPT-2 Medium fine-tuned vượt trội hơn T5 và các baseline khác
- Đánh giá trên nhiều metrics: BLEU, ROUGE, BERTScore, METEOR

### Phân loại spoiler
- SBERT + SVM đạt hiệu suất tốt nhất
- So sánh 8 bộ phân loại máy học khác nhau
- Đánh giá comprehensive với accuracy, precision, recall, F1, MCC

## Notebooks

1. `01_data_exploration.ipynb`: Khám phá và phân tích dữ liệu
2. `02_spoiler_generation_experiments.ipynb`: Thí nghiệm tạo spoiler
3. `03_spoiler_classification_experiments.ipynb`: Thí nghiệm phân loại
4. `04_results_analysis.ipynb`: Phân tích kết quả chi tiết

## Cấu hình

- `data_config.yaml`: Cấu hình xử lý dữ liệu
- `spoiler_generation_config.yaml`: Cấu hình GPT-2 fine-tuning
- `spoiler_classification_config.yaml`: Cấu hình SBERT + ML classifiers

## Requirements

Xem file `requirements.txt` để biết danh sách đầy đủ các thư viện cần thiết.

## Trích dẫn

```bibtex
@article{clickbait_spoiler_2023,
  title={Clickbait Spoiler Generation and Classification using GPT-2 and SBERT},
  author={Your Name},
  journal={Your Conference/Journal},
  year={2023}
}
```
