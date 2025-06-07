#!/usr/bin/env python3
"""
Script để kiểm tra môi trường và các dependencies cho nghiên cứu clickbait spoiler
"""

import sys
import importlib.util

def check_package(package_name, import_name=None):
    """Kiểm tra xem package có được cài đặt và import được không"""
    if import_name is None:
        import_name = package_name
    
    try:
        spec = importlib.util.find_spec(import_name)
        if spec is None:
            return False, "Not found"
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        version = getattr(module, '__version__', 'Unknown')
        return True, version
    except Exception as e:
        return False, str(e)

def main():
    print("🔍 Đang kiểm tra môi trường cho nghiên cứu Clickbait Spoiler...")
    print("=" * 60)
    
    # Danh sách các packages cần thiết
    required_packages = [
        ("torch", "torch"),
        ("transformers", "transformers"), 
        ("datasets", "datasets"),
        ("accelerate", "accelerate"),
        ("nltk", "nltk"),
        ("spacy", "spacy"),
        ("sentence-transformers", "sentence_transformers"),
        ("scikit-learn", "sklearn"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("rouge-score", "rouge_score"),
        ("bert-score", "bert_score"),
        ("sacrebleu", "sacrebleu"),
        ("tqdm", "tqdm"),
        ("wandb", "wandb"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("yaml", "yaml"),
        ("jupyter", "jupyter"),
        ("ipykernel", "IPython")
    ]
    
    print(f"Python version: {sys.version}")
    print("-" * 60)
    
    all_good = True
    
    for package_name, import_name in required_packages:
        success, version = check_package(package_name, import_name)
        status = "✅" if success else "❌"
        print(f"{status} {package_name:<20} {version}")
        
        if not success:
            all_good = False
    
    print("-" * 60)
    
    if all_good:
        print("🎉 Tất cả dependencies đã được cài đặt thành công!")
        print("\n📋 Bước tiếp theo:")
        print("1. Copy dữ liệu train.jsonl và validation.jsonl vào thư mục data/raw/")
        print("2. Chạy notebooks để khám phá dữ liệu: jupyter lab")
        print("3. Implement các class model trong src/")
        print("4. Chạy thí nghiệm với scripts/")
    else:
        print("⚠️  Một số dependencies chưa được cài đặt đúng cách!")
        print("Hãy chạy: pip install -r requirements.txt")
    
    print("\n🔗 Để kích hoạt môi trường này trong tương lai:")
    print("   .\\venv\\Scripts\\Activate.ps1")

if __name__ == "__main__":
    main() 