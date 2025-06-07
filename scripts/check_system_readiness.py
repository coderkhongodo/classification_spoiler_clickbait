#!/usr/bin/env python3
"""
Script kiểm tra system readiness cho training
"""

import torch
import sys
import psutil
import os
from pathlib import Path
import pkg_resources
import json
import pandas as pd

def check_cuda_availability():
    """Kiểm tra CUDA availability"""
    print("🔍 CHECKING CUDA AVAILABILITY")
    print("=" * 50)
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"CUDA Devices: {device_count}")
        
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  Device {i}: {device_name} ({memory:.1f} GB)")
            
        current_device = torch.cuda.current_device()
        print(f"Current Device: {current_device}")
    else:
        print("❌ CUDA not available - will use CPU training")
    
    return cuda_available

def check_memory():
    """Kiểm tra system memory"""
    print("\n💾 CHECKING SYSTEM MEMORY")
    print("=" * 50)
    
    memory = psutil.virtual_memory()
    total_gb = memory.total / 1024**3
    available_gb = memory.available / 1024**3
    used_percent = memory.percent
    
    print(f"Total Memory: {total_gb:.1f} GB")
    print(f"Available Memory: {available_gb:.1f} GB")
    print(f"Memory Usage: {used_percent:.1f}%")
    
    # Check if enough memory for training
    if available_gb < 4:
        print("⚠️  WARNING: Less than 4GB available memory")
        print("   Consider closing other applications before training")
    else:
        print("✅ Sufficient memory available")
    
    return available_gb

def check_dependencies():
    """Kiểm tra required packages"""
    print("\n📦 CHECKING DEPENDENCIES")
    print("=" * 50)
    
    required_packages = [
        'torch', 'transformers', 'sentence-transformers',
        'scikit-learn', 'pandas', 'numpy', 'matplotlib',
        'seaborn', 'tqdm', 'wandb'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            version = pkg_resources.get_distribution(package).version
            print(f"✅ {package}: {version}")
        except pkg_resources.DistributionNotFound:
            print(f"❌ {package}: NOT INSTALLED")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install " + " ".join(missing_packages))
    else:
        print("\n✅ All dependencies installed")
    
    return len(missing_packages) == 0

def check_data_readiness():
    """Kiểm tra data readiness"""
    print("\n📊 CHECKING DATA READINESS")
    print("=" * 50)
    
    checks = {
        "Raw data extracted": Path("data/train.jsonl").exists(),
        "Task 1 processed": Path("data/processed/task1/train_processed.csv").exists(),
        "Task 2 processed": Path("data/processed/task2/train_processed.csv").exists(),
        "Embeddings created": Path("data/processed/task2/train_embeddings.pkl").exists(),
        "Validation data": Path("data/processed/task1/validation_processed.csv").exists()
    }
    
    all_ready = True
    for check, status in checks.items():
        icon = "✅" if status else "❌"
        print(f"{icon} {check}")
        if not status:
            all_ready = False
    
    if all_ready:
        # Check data sizes
        try:
            train_t1 = pd.read_csv("data/processed/task1/train_processed.csv")
            train_t2 = pd.read_csv("data/processed/task2/train_processed.csv")
            
            print(f"\n📈 Data Statistics:")
            print(f"  Task 1 training samples: {len(train_t1):,}")
            print(f"  Task 2 training samples: {len(train_t2):,}")
            print(f"  Total processed samples: {len(train_t1) + len(train_t2):,}")
            
        except Exception as e:
            print(f"⚠️  Could not read data files: {e}")
    
    return all_ready

def check_directory_structure():
    """Kiểm tra directory structure"""
    print("\n📁 CHECKING DIRECTORY STRUCTURE")
    print("=" * 50)
    
    required_dirs = [
        "data/processed/task1",
        "data/processed/task2",
        "results",
        "models",
        "logs",
        "src/training",
        "src/models",
        "src/evaluation",
        "configs"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} - MISSING")
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"\n⚠️  Creating missing directories...")
        for dir_path in missing_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            print(f"   Created: {dir_path}")
    
    return len(missing_dirs) == 0

def check_config_files():
    """Kiểm tra config files"""
    print("\n⚙️  CHECKING CONFIG FILES")
    print("=" * 50)
    
    config_files = [
        "configs/spoiler_generation_config.yaml",
        "configs/spoiler_classification_config.yaml",
        "configs/data_config.yaml"
    ]
    
    all_configs = True
    for config_file in config_files:
        if Path(config_file).exists():
            print(f"✅ {config_file}")
        else:
            print(f"❌ {config_file} - MISSING")
            all_configs = False
    
    return all_configs

def estimate_training_time():
    """Ước tính training time"""
    print("\n⏱️  TRAINING TIME ESTIMATION")
    print("=" * 50)
    
    cuda_available = torch.cuda.is_available()
    
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        if "RTX" in device_name or "GTX" in device_name:
            task1_time = "2-4 hours"
            task2_time = "30-60 minutes"
        else:
            task1_time = "4-8 hours"
            task2_time = "1-2 hours"
    else:
        task1_time = "8-16 hours"
        task2_time = "2-4 hours"
    
    print(f"Task 1 (GPT-2 Fine-tuning): {task1_time}")
    print(f"Task 2 (SBERT + ML): {task2_time}")
    print(f"Total estimated time: {task1_time} + {task2_time}")

def generate_readiness_report():
    """Tạo readiness report"""
    print("\n📋 GENERATING READINESS REPORT")
    print("=" * 60)
    
    # Run all checks
    cuda_ok = check_cuda_availability()
    memory_gb = check_memory()
    deps_ok = check_dependencies()
    data_ok = check_data_readiness()
    dirs_ok = check_directory_structure()
    configs_ok = check_config_files()
    
    estimate_training_time()
    
    # Overall assessment
    print("\n🎯 OVERALL READINESS ASSESSMENT")
    print("=" * 60)
    
    readiness_score = 0
    total_checks = 6
    
    checks_results = [
        ("CUDA/GPU Setup", cuda_ok),
        ("Memory Available", memory_gb >= 4),
        ("Dependencies", deps_ok),
        ("Data Processing", data_ok),
        ("Directory Structure", dirs_ok),
        ("Config Files", configs_ok)
    ]
    
    for check_name, status in checks_results:
        icon = "✅" if status else "❌"
        print(f"{icon} {check_name}")
        if status:
            readiness_score += 1
    
    readiness_percentage = (readiness_score / total_checks) * 100
    
    print(f"\n🏆 READINESS SCORE: {readiness_score}/{total_checks} ({readiness_percentage:.0f}%)")
    
    if readiness_percentage >= 90:
        print("🚀 EXCELLENT! Ready to start training immediately")
        print("💡 Recommendation: Begin with Task 2 (faster) then Task 1")
    elif readiness_percentage >= 70:
        print("⚠️  GOOD! Minor issues need attention before training")
        print("💡 Recommendation: Fix missing items then start training")
    else:
        print("❌ NOT READY! Major issues need to be resolved")
        print("💡 Recommendation: Address all issues before attempting training")
    
    # Create summary report
    report = {
        "timestamp": str(pd.Timestamp.now()),
        "readiness_score": f"{readiness_score}/{total_checks}",
        "readiness_percentage": readiness_percentage,
        "cuda_available": cuda_ok,
        "memory_gb": memory_gb,
        "dependencies_ok": deps_ok,
        "data_ready": data_ok,
        "directories_ok": dirs_ok,
        "configs_ok": configs_ok,
        "next_steps": get_next_steps(readiness_percentage)
    }
    
    # Save report
    with open("TRAINING_READINESS_REPORT.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Report saved: TRAINING_READINESS_REPORT.json")
    
    return readiness_percentage

def get_next_steps(readiness_percentage):
    """Get next steps based on readiness"""
    if readiness_percentage >= 90:
        return [
            "Ready to start training!",
            "Begin with Task 2 (SBERT + ML classification) - faster training",
            "Then proceed with Task 1 (GPT-2 fine-tuning) - longer training",
            "Set up experiment tracking (wandb) if desired",
            "Ensure adequate time for training completion"
        ]
    elif readiness_percentage >= 70:
        return [
            "Fix missing dependencies or configurations",
            "Verify data integrity",
            "Check available system resources",
            "Then proceed with training"
        ]
    else:
        return [
            "Install missing dependencies",
            "Complete data preprocessing",
            "Set up proper directory structure",
            "Configure training parameters",
            "Verify system requirements"
        ]

def main():
    """Main function"""
    print("🔍 SYSTEM READINESS CHECK FOR CLICKBAIT SPOILER TRAINING")
    print("=" * 70)
    print("Checking all prerequisites for model training...")
    print()
    
    readiness_score = generate_readiness_report()
    
    print("\n" + "=" * 70)
    print("🎉 SYSTEM CHECK COMPLETED!")
    print(f"📊 Final Readiness: {readiness_score:.0f}%")
    print("=" * 70)

if __name__ == "__main__":
    main() 