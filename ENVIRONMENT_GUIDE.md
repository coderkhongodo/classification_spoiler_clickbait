# 🐍 Hướng dẫn sử dụng môi trường ảo

## ✅ Môi trường đã được thiết lập thành công!

### 📦 Đã cài đặt:
- **Python**: 3.12.3
- **PyTorch**: 2.7.1+cpu
- **Transformers**: 4.52.4
- **Sentence Transformers**: 4.1.0
- **Scikit-learn**: 1.7.0
- **Và 50+ thư viện khác** cần thiết cho nghiên cứu

### 🔧 Kích hoạt môi trường ảo

#### Windows PowerShell:
```powershell
.\venv\Scripts\Activate.ps1
```

#### Windows CMD:
```cmd
venv\Scripts\activate.bat
```

#### Mac/Linux:
```bash
source venv/bin/activate
```

Khi kích hoạt thành công, bạn sẽ thấy `(venv)` xuất hiện ở đầu dòng lệnh.

### 🚀 Kiểm tra môi trường

```bash
# Kiểm tra toàn bộ môi trường
python scripts/check_environment.py

# Kiểm tra PyTorch
python -c "import torch; print('PyTorch:', torch.__version__)"

# Kiểm tra Transformers
python -c "import transformers; print('Transformers:', transformers.__version__)"
```

### 📊 Dữ liệu

Dữ liệu đã được giải nén vào:
- `data/raw/train.jsonl` (18.3 MB, 3,200 samples)
- `data/raw/validation.jsonl` (4.8 MB, 800 samples)

### 🎯 Bắt đầu nghiên cứu

1. **Khám phá dữ liệu**:
```bash
jupyter lab
# Mở notebooks/01_data_exploration.ipynb
```

2. **Setup project**:
```bash
python scripts/setup_project.py
```

3. **Kiểm tra môi trường**:
```bash
python scripts/check_environment.py
```

### 📁 Cấu trúc project

```
clickbait-spoiler-research/
├── data/                    # ✅ Đã có dữ liệu
│   ├── raw/                 # train.jsonl, validation.jsonl
│   ├── processed/           # ✅ Đã tạo
│   └── splits/              # ✅ Đã tạo
├── src/                     # ✅ Đã tạo modules
├── configs/                 # ✅ Đã có config files
├── notebooks/               # ✅ Đã có notebook đầu tiên
├── scripts/                 # ✅ Đã có setup scripts
├── models/                  # ✅ Đã tạo
├── results/                 # ✅ Đã tạo
└── logs/                    # ✅ Đã tạo
```

### 🔄 Tắt môi trường ảo

```bash
deactivate
```

### 📝 Ghi chú quan trọng

1. **Luôn kích hoạt môi trường ảo** trước khi làm việc
2. **PyTorch CPU version** - phù hợp cho nghiên cứu trên laptop
3. **Tất cả dependencies** đã được cài đặt và kiểm tra
4. **Project structure** đã sẵn sàng cho implementation

### 🚨 Khắc phục sự cố

#### Nếu gặp lỗi kích hoạt PowerShell:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### Nếu thiếu dependencies:
```bash
pip install -r requirements.txt
```

#### Nếu muốn cài đặt lại hoàn toàn:
```bash
# Xóa môi trường cũ
rmdir /s venv

# Tạo lại
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## 🎉 Môi trường sẵn sàng cho nghiên cứu!

**Bước tiếp theo**: Mở Jupyter Lab và bắt đầu khám phá dữ liệu:
```bash
jupyter lab
``` 