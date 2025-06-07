# 📊 Báo cáo Khám phá Dữ liệu Clickbait Spoiler

## 🎯 Tổng quan Dataset

### Thông tin cơ bản
- **Nguồn**: SemEval-2023 Clickbait Spoiling Dataset
- **Tổng số mẫu**: 4,000
- **Training set**: 3,200 mẫu (80%)
- **Validation set**: 800 mẫu (20%)

### Cấu trúc dữ liệu
Dataset chứa 14 trường thông tin chính:
- `uuid`: ID duy nhất
- `postText`: Nội dung clickbait post (list)
- `targetTitle`: Tiêu đề bài viết gốc
- `targetParagraphs`: Nội dung bài viết gốc (list)
- `targetDescription`: Mô tả bài viết
- `targetKeywords`: Từ khóa liên quan
- `spoiler`: Spoiler text (list)
- `tags`: Loại spoiler (list)

## 📋 Phân tích Loại Spoiler

### Phân phối các loại spoiler:
1. **Phrase** (Cụm từ): 1,367 mẫu (42.7%)
   - Spoiler ngắn, thường 1-3 từ
   - Độ dài trung bình: 2.5 từ
   - Ví dụ: "2070", "intellectual stimulation"

2. **Passage** (Đoạn văn): 1,274 mẫu (39.8%)
   - Spoiler dài hơn, thường là câu hoặc đoạn ngắn
   - Độ dài trung bình: 21.2 từ
   - Ví dụ: "how about that morning we go throw?"

3. **Multi** (Nhiều phần): 559 mẫu (17.5%)
   - Spoiler phức tạp nhất, nhiều thành phần
   - Độ dài trung bình: 29.1 từ
   - Độ dài có thể lên đến 255 từ

### Insights:
- Dataset **tương đối cân bằng** giữa phrase và passage
- Multi class ít hơn nhưng vẫn đủ để training
- **Imbalanced dataset** - cần xem xét khi training classification model

## 📏 Phân tích Độ dài Văn bản

### Thống kê chi tiết:

| Trường | Trung bình | Trung vị | Min-Max | Độ lệch chuẩn |
|--------|------------|----------|---------|---------------|
| Post Text | 10.8 từ | 11.0 từ | 1-28 từ | 3.5 |
| Spoiler | 14.6 từ | 7.0 từ | 1-255 từ | 19.3 |
| Target Title | 11.3 từ | 11.0 từ | 0-33 từ | 3.5 |
| Target Paragraphs | 514.8 từ | 352.5 từ | 7-14,059 từ | 583.3 |

### Key Findings:
- **Post Text** rất ngắn (trung bình 11 từ) - đặc trưng của clickbait
- **Target Paragraphs** rất dài và có độ biến thiên lớn
- **Spoiler length** có phân phối skewed (median < mean)
- **Target Title** tương tự Post Text về độ dài

## ❓ Phân tích Dữ liệu Thiếu

### Missing Values:
- **postText**: 0% (hoàn hảo)
- **targetTitle**: 0.03% (1/3200 - negligible)
- **targetParagraphs**: 0% (hoàn hảo)
- **targetDescription**: 10.4% (332/3200)
- **targetKeywords**: 41.1% (1314/3200)
- **spoiler**: 0% (hoàn hảo)

### Implications:
- **targetKeywords** thiếu nhiều nhất - cần xử lý đặc biệt
- **targetDescription** thiếu ít hơn nhưng vẫn đáng kể
- Các trường chính (postText, spoiler, targetParagraphs) đầy đủ

## 🎯 Phân tích theo Task cụ thể

### Task 1: Spoiler Generation
**Input**: postText + targetParagraphs → **Output**: spoiler

#### Thống kê Input:
- **Độ dài trung bình**: 525.6 từ
- **≤ 512 tokens**: 66.9% (cần truncation cho 33.1%)
- **Phù hợp với GPT-2 max_length = 512**

#### Thống kê Output:
- **Độ dài trung bình**: 14.6 từ
- **≤ 128 tokens**: 99.6% (rất phù hợp)
- **Max target length = 128 là đủ**

### Task 2: Spoiler Type Classification
**Features**: 5 text fields → SBERT embeddings → **Output**: spoiler type

#### Feature Analysis:
- **postText**: 10.8 từ (ngắn, đặc trưng)
- **targetTitle**: 11.3 từ (tương tự postText)
- **targetParagraphs**: 514.8 từ (dài nhất, chứa nhiều thông tin)
- **targetDescription**: 20.9 từ (trung bình)
- **targetKeywords**: 7.6 từ (ngắn)

#### Embedding Strategy:
- **SBERT model**: all-MiniLM-L6-v2 (384 dimensions)
- **Combined features**: 384 × 5 = 1,920 dimensions
- **Handle missing values** trong targetDescription và targetKeywords

## 💡 Khuyến nghị Implementation

### Task 1 - Spoiler Generation:
1. **Model**: GPT-2 Medium fine-tuned
2. **Preprocessing**:
   - `max_input_length = 512` (cover 67% data, truncate rest)
   - `max_target_length = 128` (cover 99.6% data)
   - Combine postText + targetParagraphs as input
3. **Training**:
   - Learning rate: 5e-5
   - Batch size: 8 với gradient accumulation
   - Early stopping based on validation loss

### Task 2 - Spoiler Type Classification:
1. **Feature Engineering**:
   - Use SBERT để encode 5 text fields
   - Handle missing values:
     - targetDescription: fill với ""
     - targetKeywords: fill với ""
   - Concatenate embeddings → 1920-dim vector

2. **Models to test**:
   - Naive Bayes, KNN, Decision Tree
   - AdaBoost, Logistic Regression
   - Gradient Boosting, Random Forest
   - **SVM** (expected best performer)

3. **Evaluation**:
   - Cross-validation với 5 folds
   - Metrics: Accuracy, Precision, Recall, F1, MCC
   - Handle class imbalance nếu cần

### Data Preprocessing:
1. **Text cleaning**: Remove HTML, normalize whitespace
2. **Tokenization**: Sử dụng GPT-2 tokenizer cho generation
3. **Missing value handling**: Strategy khác nhau cho từng field
4. **Data splitting**: Maintain class distribution trong splits

## 📊 Biểu đồ được tạo

1. **spoiler_types_distribution.png**: Phân phối loại spoiler
2. **text_length_distributions.png**: Phân phối độ dài văn bản
3. **spoiler_analysis_by_type.png**: So sánh spoiler theo loại
4. **missing_data_analysis.png**: Phân tích dữ liệu thiếu
5. **task_specific_analysis.png**: Phân tích theo từng task

## 🎯 Kết luận

Dataset SemEval-2023 Clickbait Spoiling có chất lượng tốt với:
- ✅ **Dữ liệu đầy đủ** cho các trường chính
- ✅ **Phân phối hợp lý** giữa các loại spoiler
- ✅ **Độ dài phù hợp** với các model hiện tại
- ⚠️ **Cần xử lý** missing values trong targetKeywords
- ⚠️ **Cần truncation** cho input dài trong generation task

Dataset sẵn sàng cho việc implement cả hai task với các insights và khuyến nghị đã được đưa ra. 