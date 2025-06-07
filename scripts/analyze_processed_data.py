#!/usr/bin/env python3
"""
Script phân tích và kiểm tra dữ liệu đã được processed
"""

import json
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_task1_data():
    """Phân tích dữ liệu Task 1"""
    logger.info("🔍 PHÂN TÍCH DỮ LIỆU TASK 1 (Spoiler Generation)")
    logger.info("=" * 60)
    
    for split in ['train', 'validation']:
        csv_file = f"data/processed/task1/{split}_processed.csv"
        if not Path(csv_file).exists():
            logger.warning(f"⚠️  File không tồn tại: {csv_file}")
            continue
            
        logger.info(f"\n📊 Phân tích {split} data:")
        df = pd.read_csv(csv_file)
        
        logger.info(f"  📈 Số samples: {len(df)}")
        logger.info(f"  📏 Input length - Mean: {df['input_length'].mean():.1f}, Max: {df['input_length'].max()}")
        logger.info(f"  📏 Target length - Mean: {df['target_length'].mean():.1f}, Max: {df['target_length'].max()}")
        
        # Phân tích độ dài input tokens
        input_stats = df['input_length'].describe()
        logger.info(f"  📊 Input length distribution:")
        logger.info(f"     Min: {input_stats['min']:.0f}, 25%: {input_stats['25%']:.0f}")
        logger.info(f"     50%: {input_stats['50%']:.0f}, 75%: {input_stats['75%']:.0f}")
        logger.info(f"     Max: {input_stats['max']:.0f}")
        
        # Kiểm tra truncation
        truncated = (df['input_length'] >= 512).sum()
        logger.info(f"  ✂️  Samples bị truncate (≥512 tokens): {truncated} ({truncated/len(df)*100:.1f}%)")
        
        # Phân tích spoiler types
        spoiler_types = []
        for types_str in df['spoiler_type']:
            if pd.notna(types_str):
                try:
                    types_list = eval(types_str) if isinstance(types_str, str) else types_str
                    if isinstance(types_list, list):
                        spoiler_types.extend(types_list)
                except:
                    pass
        
        type_counts = Counter(spoiler_types)
        logger.info(f"  🏷️  Spoiler types distribution:")
        for stype, count in type_counts.most_common():
            logger.info(f"     {stype}: {count} ({count/len(df)*100:.1f}%)")

def analyze_task2_data():
    """Phân tích dữ liệu Task 2"""
    logger.info("\n🔍 PHÂN TÍCH DỮ LIỆU TASK 2 (Spoiler Classification)")
    logger.info("=" * 60)
    
    for split in ['train', 'validation']:
        csv_file = f"data/processed/task2/{split}_processed.csv"
        embeddings_file = f"data/processed/task2/{split}_embeddings.pkl"
        
        if not Path(csv_file).exists():
            logger.warning(f"⚠️  File không tồn tại: {csv_file}")
            continue
            
        logger.info(f"\n📊 Phân tích {split} data:")
        df = pd.read_csv(csv_file)
        
        logger.info(f"  📈 Số samples: {len(df)}")
        
        # Phân tích text features
        logger.info(f"  📏 Text lengths:")
        logger.info(f"     Post text - Mean: {df['post_length'].mean():.1f}, Max: {df['post_length'].max()}")
        logger.info(f"     Spoiler - Mean: {df['spoiler_length'].mean():.1f}, Max: {df['spoiler_length'].max()}")
        logger.info(f"     Target - Mean: {df['target_length'].mean():.1f}, Max: {df['target_length'].max()}")
        
        # Phân tích numerical features
        logger.info(f"  🔢 Numerical features:")
        logger.info(f"     Keywords count - Mean: {df['keywords_count'].mean():.1f}, Max: {df['keywords_count'].max()}")
        logger.info(f"     Has description: {df['has_description'].sum()} ({df['has_description'].mean()*100:.1f}%)")
        
        # Phân tích labels
        label_counts = df['spoiler_type'].value_counts()
        logger.info(f"  🏷️  Label distribution:")
        for label, count in label_counts.items():
            logger.info(f"     {label}: {count} ({count/len(df)*100:.1f}%)")
        
        # Phân tích embeddings
        if Path(embeddings_file).exists():
            logger.info(f"  🧠 Embeddings analysis:")
            with open(embeddings_file, 'rb') as f:
                embeddings = pickle.load(f)
            
            for field, emb in embeddings.items():
                logger.info(f"     {field}: shape {emb.shape}, dtype {emb.dtype}")
                logger.info(f"       Mean: {emb.mean():.4f}, Std: {emb.std():.4f}")
                logger.info(f"       Min: {emb.min():.4f}, Max: {emb.max():.4f}")

def create_visualizations():
    """Tạo visualizations cho dữ liệu đã processed"""
    logger.info("\n📊 TẠO VISUALIZATIONS")
    logger.info("=" * 60)
    
    # Create output directory
    viz_dir = Path("results/preprocessing_analysis")
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Task 1 visualization
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Task 1 (Spoiler Generation) - Data Analysis', fontsize=16)
        
        for i, split in enumerate(['train', 'validation']):
            csv_file = f"data/processed/task1/{split}_processed.csv"
            if Path(csv_file).exists():
                df = pd.read_csv(csv_file)
                
                # Input length distribution
                axes[i, 0].hist(df['input_length'], bins=50, alpha=0.7, color='skyblue')
                axes[i, 0].axvline(512, color='red', linestyle='--', label='Max length (512)')
                axes[i, 0].set_title(f'{split.title()} - Input Length Distribution')
                axes[i, 0].set_xlabel('Input Length (tokens)')
                axes[i, 0].set_ylabel('Frequency')
                axes[i, 0].legend()
                
                # Target length distribution
                axes[i, 1].hist(df['target_length'], bins=30, alpha=0.7, color='lightgreen')
                axes[i, 1].set_title(f'{split.title()} - Target Length Distribution')
                axes[i, 1].set_xlabel('Target Length (tokens)')
                axes[i, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'task1_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"💾 Saved: {viz_dir / 'task1_analysis.png'}")
        
    except Exception as e:
        logger.error(f"❌ Error creating Task 1 visualization: {e}")
    
    # Task 2 visualization
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Task 2 (Spoiler Classification) - Data Analysis', fontsize=16)
        
        for i, split in enumerate(['train', 'validation']):
            csv_file = f"data/processed/task2/{split}_processed.csv"
            if Path(csv_file).exists():
                df = pd.read_csv(csv_file)
                
                # Text length distributions
                axes[i, 0].hist([df['post_length'], df['spoiler_length'], df['target_length']], 
                               bins=30, alpha=0.7, label=['Post', 'Spoiler', 'Target'])
                axes[i, 0].set_title(f'{split.title()} - Text Length Distribution')
                axes[i, 0].set_xlabel('Length (words)')
                axes[i, 0].set_ylabel('Frequency')
                axes[i, 0].legend()
                axes[i, 0].set_yscale('log')
                
                # Label distribution
                label_counts = df['spoiler_type'].value_counts()
                axes[i, 1].pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%')
                axes[i, 1].set_title(f'{split.title()} - Label Distribution')
                
                # Features correlation
                numeric_cols = ['post_length', 'spoiler_length', 'target_length', 'keywords_count']
                corr_matrix = df[numeric_cols].corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[i, 2])
                axes[i, 2].set_title(f'{split.title()} - Feature Correlation')
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'task2_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"💾 Saved: {viz_dir / 'task2_analysis.png'}")
        
    except Exception as e:
        logger.error(f"❌ Error creating Task 2 visualization: {e}")

def create_summary_report():
    """Tạo báo cáo tóm tắt"""
    logger.info("\n📋 TẠO BÁO CÁO TÓM TẮT")
    logger.info("=" * 60)
    
    report_lines = []
    report_lines.append("# PREPROCESSING DATA ANALYSIS REPORT")
    report_lines.append("=" * 50)
    report_lines.append("")
    
    # Task 1 summary
    report_lines.append("## Task 1: Spoiler Generation")
    report_lines.append("")
    
    for split in ['train', 'validation']:
        csv_file = f"data/processed/task1/{split}_processed.csv"
        if Path(csv_file).exists():
            df = pd.read_csv(csv_file)
            report_lines.append(f"### {split.title()} Data")
            report_lines.append(f"- Samples: {len(df):,}")
            report_lines.append(f"- Input Length: μ={df['input_length'].mean():.1f}, σ={df['input_length'].std():.1f}")
            report_lines.append(f"- Target Length: μ={df['target_length'].mean():.1f}, σ={df['target_length'].std():.1f}")
            
            truncated = (df['input_length'] >= 512).sum()
            report_lines.append(f"- Truncated Samples: {truncated} ({truncated/len(df)*100:.1f}%)")
            report_lines.append("")
    
    # Task 2 summary
    report_lines.append("## Task 2: Spoiler Classification")
    report_lines.append("")
    
    for split in ['train', 'validation']:
        csv_file = f"data/processed/task2/{split}_processed.csv"
        if Path(csv_file).exists():
            df = pd.read_csv(csv_file)
            report_lines.append(f"### {split.title()} Data")
            report_lines.append(f"- Samples: {len(df):,}")
            
            # Label distribution
            label_counts = df['spoiler_type'].value_counts()
            report_lines.append("- Label Distribution:")
            for label, count in label_counts.items():
                report_lines.append(f"  - {label}: {count} ({count/len(df)*100:.1f}%)")
            
            report_lines.append(f"- Text Features:")
            report_lines.append(f"  - Post Length: μ={df['post_length'].mean():.1f}")
            report_lines.append(f"  - Spoiler Length: μ={df['spoiler_length'].mean():.1f}")
            report_lines.append(f"  - Target Length: μ={df['target_length'].mean():.1f}")
            report_lines.append("")
    
    # Embeddings info
    report_lines.append("## Embeddings")
    report_lines.append("")
    embeddings_info = {
        'train': Path("data/processed/task2/train_embeddings.pkl"),
        'validation': Path("data/processed/task2/validation_embeddings.pkl")
    }
    
    for split, emb_file in embeddings_info.items():
        if emb_file.exists():
            with open(emb_file, 'rb') as f:
                embeddings = pickle.load(f)
            
            report_lines.append(f"### {split.title()} Embeddings")
            for field, emb in embeddings.items():
                report_lines.append(f"- {field}: {emb.shape} ({emb.nbytes / 1024 / 1024:.1f} MB)")
            report_lines.append("")
    
    # Save report
    report_file = Path("results/preprocessing_analysis/PREPROCESSING_REPORT.md")
    report_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"💾 Saved report: {report_file}")

def main():
    """Hàm chính"""
    logger.info("🔍 PHÂN TÍCH DỮ LIỆU ĐÃ PROCESSED")
    logger.info("=" * 60)
    
    # Analyze Task 1 data
    analyze_task1_data()
    
    # Analyze Task 2 data
    analyze_task2_data()
    
    # Create visualizations
    create_visualizations()
    
    # Create summary report
    create_summary_report()
    
    logger.info(f"\n{'='*60}")
    logger.info("🎉 PHÂN TÍCH HOÀN THÀNH!")
    logger.info("📁 Kết quả lưu trong: results/preprocessing_analysis/")

if __name__ == "__main__":
    main() 