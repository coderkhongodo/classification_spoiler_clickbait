#!/usr/bin/env python3
"""
Script setup ban đầu cho project nghiên cứu clickbait spoiler
"""

import os
import json
import pandas as pd
from pathlib import Path

def create_directories():
    """Tạo các thư mục cần thiết nếu chưa tồn tại"""
    directories = [
        "data/raw",
        "data/processed", 
        "data/splits",
        "models/spoiler_generator",
        "models/spoiler_classifier",
        "logs/spoiler_generation",
        "logs/spoiler_classification", 
        "results/spoiler_generation",
        "results/spoiler_classification",
        "data/processed/embeddings"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def check_data_files():
    """Kiểm tra xem các file dữ liệu có tồn tại không"""
    data_files = {
        "data/raw/train.jsonl": "Training data",
        "data/raw/validation.jsonl": "Validation data", 
        "data/raw/README.md": "Dataset documentation"
    }
    
    print("\n📊 Checking data files:")
    for file_path, description in data_files.items():
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"✅ {description}: {file_path} ({size_mb:.1f} MB)")
        else:
            print(f"❌ {description}: {file_path} - NOT FOUND")

def explore_data_sample():
    """Khám phá mẫu dữ liệu đầu tiên"""
    train_file = "data/raw/train.jsonl"
    
    if not os.path.exists(train_file):
        print("❌ Training data not found!")
        return
        
    print("\n🔍 Exploring sample data:")
    print("-" * 50)
    
    # Đọc 5 dòng đầu tiên
    with open(train_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 3:  # Chỉ xem 3 mẫu đầu
                break
            
            data = json.loads(line)
            print(f"\n📝 Sample {i+1}:")
            print(f"UUID: {data.get('uuid', 'N/A')}")
            print(f"Post Text: {data.get('postText', 'N/A')[:100]}...")
            print(f"Target Title: {data.get('targetTitle', 'N/A')[:100]}...")
            print(f"Spoiler: {data.get('spoiler', 'N/A')[:100]}...")
            print(f"Tags: {data.get('tags', 'N/A')}")
    
    # Đếm tổng số dòng
    with open(train_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    print(f"\n📊 Total training samples: {total_lines}")

def create_initial_notebook():
    """Tạo notebook khám phá dữ liệu đầu tiên"""
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Clickbait Spoiler Data Exploration\n",
                    "\n",
                    "Notebook này dùng để khám phá và phân tích bộ dữ liệu clickbait spoiler từ SemEval-2023.\n",
                    "\n",
                    "## Overview\n",
                    "- **Dataset**: SemEval-2023 Clickbait Spoiling\n", 
                    "- **Task 1**: Spoiler Generation (GPT-2)\n",
                    "- **Task 2**: Spoiler Type Classification (SBERT + ML)\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "import json\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "from collections import Counter\n",
                    "\n",
                    "# Set style\n",
                    "plt.style.use('seaborn-v0_8')\n",
                    "sns.set_palette('husl')"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Load training data\n",
                    "def load_jsonl(file_path):\n",
                    "    data = []\n",
                    "    with open(file_path, 'r', encoding='utf-8') as f:\n",
                    "        for line in f:\n",
                    "            data.append(json.loads(line))\n",
                    "    return data\n",
                    "\n",
                    "train_data = load_jsonl('../data/raw/train.jsonl')\n",
                    "val_data = load_jsonl('../data/raw/validation.jsonl')\n",
                    "\n",
                    "print(f\"Training samples: {len(train_data)}\")\n",
                    "print(f\"Validation samples: {len(val_data)}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Data Analysis\n", "Phân tích cấu trúc và phân phối dữ liệu"]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python", 
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    notebook_path = "notebooks/01_data_exploration.ipynb"
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook_content, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Created notebook: {notebook_path}")

def main():
    print("🚀 Setting up Clickbait Spoiler Research Project...")
    print("=" * 60)
    
    # Tạo thư mục
    create_directories()
    
    # Kiểm tra dữ liệu
    check_data_files()
    
    # Khám phá dữ liệu mẫu
    explore_data_sample()
    
    # Tạo notebook
    create_initial_notebook()
    
    print("\n" + "=" * 60)
    print("🎉 Project setup completed!")
    print("\n📋 Next steps:")
    print("1. Run: jupyter lab")
    print("2. Open: notebooks/01_data_exploration.ipynb")
    print("3. Explore the dataset and understand the structure")
    print("4. Implement model classes in src/")
    
if __name__ == "__main__":
    main() 