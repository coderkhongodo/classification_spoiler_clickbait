#!/usr/bin/env python3
"""
Script khám phá dữ liệu clickbait spoiler với biểu đồ chi tiết
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette('husl')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_jsonl(file_path):
    """Load dữ liệu từ file JSONL"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def basic_data_overview(train_data, val_data):
    """Tổng quan cơ bản về dữ liệu"""
    print("📊 TỔNG QUAN DỮ LIỆU")
    print("=" * 50)
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Total samples: {len(train_data) + len(val_data)}")
    
    # Kiểm tra cấu trúc dữ liệu
    sample = train_data[0]
    print(f"\nCác trường dữ liệu có sẵn:")
    for key in sample.keys():
        print(f"  - {key}: {type(sample[key])}")
    
    return len(train_data), len(val_data)

def analyze_spoiler_types(data):
    """Phân tích các loại spoiler"""
    print(f"\n📋 PHÂN TÍCH LOẠI SPOILER")
    print("=" * 50)
    
    # Đếm tags
    all_tags = []
    for item in data:
        tags = item.get('tags', [])
        if isinstance(tags, list):
            all_tags.extend(tags)
        else:
            all_tags.append(tags)
    
    tag_counts = Counter(all_tags)
    print("Phân phối loại spoiler:")
    total = sum(tag_counts.values())
    for tag, count in tag_counts.most_common():
        print(f"  {tag}: {count} ({count/total*100:.1f}%)")
    
    # Tạo biểu đồ
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Pie chart
    labels, sizes = zip(*tag_counts.most_common())
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax1.set_title('Phân phối loại Spoiler', fontsize=14, fontweight='bold')
    
    # Bar chart
    ax2.bar(labels, sizes, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_title('Số lượng mẫu theo loại Spoiler', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Số lượng')
    ax2.set_xlabel('Loại Spoiler')
    
    # Thêm số lượng lên bar
    for i, v in enumerate(sizes):
        ax2.text(i, v + 10, str(v), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/data_exploration/spoiler_types_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return tag_counts

def analyze_text_lengths(data):
    """Phân tích độ dài văn bản"""
    print(f"\n📏 PHÂN TÍCH ĐỘ DÀI VĂN BẢN")
    print("=" * 50)
    
    # Tính độ dài các trường văn bản
    post_lengths = []
    spoiler_lengths = []
    title_lengths = []
    paragraph_lengths = []
    
    for item in data:
        # Post text
        post_text = item.get('postText', [])
        if isinstance(post_text, list):
            post_text = ' '.join(post_text)
        post_lengths.append(len(post_text.split()) if post_text else 0)
        
        # Spoiler
        spoiler = item.get('spoiler', [])
        if isinstance(spoiler, list):
            spoiler = ' '.join(spoiler)
        spoiler_lengths.append(len(spoiler.split()) if spoiler else 0)
        
        # Target title
        title = item.get('targetTitle', '')
        title_lengths.append(len(title.split()) if title else 0)
        
        # Target paragraphs
        paragraphs = item.get('targetParagraphs', [])
        if isinstance(paragraphs, list):
            paragraph_text = ' '.join(paragraphs)
        else:
            paragraph_text = paragraphs or ''
        paragraph_lengths.append(len(paragraph_text.split()) if paragraph_text else 0)
    
    # Thống kê
    stats = {
        'Post Text': post_lengths,
        'Spoiler': spoiler_lengths,
        'Target Title': title_lengths,
        'Target Paragraphs': paragraph_lengths
    }
    
    print("Thống kê độ dài (số từ):")
    for field, lengths in stats.items():
        print(f"\n{field}:")
        print(f"  Trung bình: {np.mean(lengths):.1f}")
        print(f"  Trung vị: {np.median(lengths):.1f}")
        print(f"  Min-Max: {np.min(lengths)}-{np.max(lengths)}")
        print(f"  Độ lệch chuẩn: {np.std(lengths):.1f}")
    
    # Tạo biểu đồ
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
    
    for i, (field, lengths) in enumerate(stats.items()):
        # Histogram
        axes[i].hist(lengths, bins=50, alpha=0.7, color=colors[i], edgecolor='black')
        axes[i].axvline(np.mean(lengths), color='red', linestyle='--', linewidth=2,
                       label=f'Trung bình: {np.mean(lengths):.1f}')
        axes[i].axvline(np.median(lengths), color='blue', linestyle='--', linewidth=2,
                       label=f'Trung vị: {np.median(lengths):.1f}')
        axes[i].set_title(f'Phân phối độ dài {field}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Số từ')
        axes[i].set_ylabel('Tần suất')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/data_exploration/text_length_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return stats

def analyze_spoiler_by_type(data):
    """Phân tích spoiler theo từng loại"""
    print(f"\n🎯 PHÂN TÍCH SPOILER THEO LOẠI")
    print("=" * 50)
    
    # Nhóm dữ liệu theo tag
    spoiler_by_type = defaultdict(list)
    
    for item in data:
        tags = item.get('tags', [])
        spoiler = item.get('spoiler', [])
        
        if isinstance(spoiler, list):
            spoiler_text = ' '.join(spoiler)
        else:
            spoiler_text = spoiler or ''
        
        spoiler_length = len(spoiler_text.split()) if spoiler_text else 0
        
        if isinstance(tags, list):
            for tag in tags:
                spoiler_by_type[tag].append({
                    'text': spoiler_text,
                    'length': spoiler_length
                })
        else:
            spoiler_by_type[tags].append({
                'text': spoiler_text,
                'length': spoiler_length
            })
    
    # Thống kê theo loại
    print("Thống kê spoiler theo loại:")
    for spoiler_type, spoilers in spoiler_by_type.items():
        lengths = [s['length'] for s in spoilers]
        print(f"\n{spoiler_type.upper()}:")
        print(f"  Số lượng: {len(spoilers)}")
        print(f"  Độ dài trung bình: {np.mean(lengths):.1f} từ")
        print(f"  Độ dài trung vị: {np.median(lengths):.1f} từ")
        print(f"  Min-Max: {np.min(lengths)}-{np.max(lengths)} từ")
        
        # Ví dụ spoiler
        print(f"  Ví dụ:")
        for i, spoiler in enumerate(spoilers[:3]):
            print(f"    {i+1}. {spoiler['text'][:100]}...")
    
    # Tạo biểu đồ so sánh
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Box plot so sánh độ dài
    type_names = list(spoiler_by_type.keys())
    length_data = [
        [s['length'] for s in spoiler_by_type[t]] 
        for t in type_names
    ]
    
    bp = ax1.boxplot(length_data, labels=type_names, patch_artist=True)
    colors = ['lightblue', 'lightcoral', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax1.set_title('So sánh độ dài Spoiler theo loại', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Số từ')
    ax1.set_xlabel('Loại Spoiler')
    ax1.grid(True, alpha=0.3)
    
    # Violin plot
    parts = ax2.violinplot(length_data, positions=range(1, len(type_names)+1))
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    
    ax2.set_title('Phân phối độ dài Spoiler theo loại', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Số từ')
    ax2.set_xlabel('Loại Spoiler')
    ax2.set_xticks(range(1, len(type_names)+1))
    ax2.set_xticklabels(type_names)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/data_exploration/spoiler_analysis_by_type.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return spoiler_by_type

def analyze_missing_data(data):
    """Phân tích dữ liệu thiếu"""
    print(f"\n❓ PHÂN TÍCH DỮ LIỆU THIẾU")
    print("=" * 50)
    
    fields_to_check = [
        'postText', 'targetTitle', 'targetParagraphs', 
        'targetDescription', 'targetKeywords', 'spoiler'
    ]
    
    missing_stats = {}
    
    for field in fields_to_check:
        missing_count = 0
        total_count = len(data)
        
        for item in data:
            value = item.get(field, None)
            if value is None or value == '' or value == []:
                missing_count += 1
            elif isinstance(value, list) and len(value) == 0:
                missing_count += 1
            elif isinstance(value, str) and value.strip() == '':
                missing_count += 1
        
        missing_percentage = (missing_count / total_count) * 100
        missing_stats[field] = {
            'count': missing_count,
            'percentage': missing_percentage
        }
        
        print(f"{field}: {missing_count}/{total_count} ({missing_percentage:.1f}%)")
    
    # Tạo biểu đồ
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    fields = list(missing_stats.keys())
    counts = [missing_stats[f]['count'] for f in fields]
    percentages = [missing_stats[f]['percentage'] for f in fields]
    
    # Bar chart số lượng
    bars1 = ax1.bar(fields, counts, color='orange', alpha=0.7, edgecolor='black')
    ax1.set_title('Số lượng dữ liệu thiếu theo trường', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Số lượng')
    ax1.set_xlabel('Trường dữ liệu')
    ax1.tick_params(axis='x', rotation=45)
    
    # Thêm số lượng lên bar
    for bar, count in zip(bars1, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(count), ha='center', va='bottom', fontweight='bold')
    
    # Bar chart phần trăm
    bars2 = ax2.bar(fields, percentages, color='red', alpha=0.7, edgecolor='black')
    ax2.set_title('Phần trăm dữ liệu thiếu theo trường', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Phần trăm (%)')
    ax2.set_xlabel('Trường dữ liệu')
    ax2.tick_params(axis='x', rotation=45)
    
    # Thêm phần trăm lên bar
    for bar, pct in zip(bars2, percentages):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/data_exploration/missing_data_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return missing_stats

def analyze_task_specific_data(data):
    """Phân tích dữ liệu cho từng task cụ thể"""
    print(f"\n🎯 PHÂN TÍCH THEO TASK CỤ THỂ")
    print("=" * 50)
    
    # Task 1: Generation (postText + targetParagraphs -> spoiler)
    print("TASK 1 - SPOILER GENERATION:")
    generation_inputs = []
    generation_outputs = []
    
    for item in data:
        # Input: postText + targetParagraphs
        post_text = item.get('postText', [])
        if isinstance(post_text, list):
            post_text = ' '.join(post_text)
        
        target_paragraphs = item.get('targetParagraphs', [])
        if isinstance(target_paragraphs, list):
            target_text = ' '.join(target_paragraphs)
        else:
            target_text = target_paragraphs or ''
        
        combined_input = f"{post_text} {target_text}".strip()
        generation_inputs.append(len(combined_input.split()))
        
        # Output: spoiler
        spoiler = item.get('spoiler', [])
        if isinstance(spoiler, list):
            spoiler = ' '.join(spoiler)
        generation_outputs.append(len(spoiler.split()) if spoiler else 0)
    
    print(f"  Input length (postText + targetParagraphs):")
    print(f"    Trung bình: {np.mean(generation_inputs):.1f} từ")
    print(f"    <= 512 tokens: {np.mean([l <= 512 for l in generation_inputs])*100:.1f}%")
    print(f"  Output length (spoiler):")
    print(f"    Trung bình: {np.mean(generation_outputs):.1f} từ")
    print(f"    <= 128 tokens: {np.mean([l <= 128 for l in generation_outputs])*100:.1f}%")
    
    # Task 2: Classification (5 text fields -> spoiler type)
    print(f"\nTASK 2 - SPOILER TYPE CLASSIFICATION:")
    classification_features = {
        'postText': [],
        'targetTitle': [],
        'targetParagraphs': [],
        'targetDescription': [],
        'targetKeywords': []
    }
    
    for item in data:
        # postText
        post_text = item.get('postText', [])
        if isinstance(post_text, list):
            post_text = ' '.join(post_text)
        classification_features['postText'].append(len(post_text.split()) if post_text else 0)
        
        # targetTitle
        title = item.get('targetTitle', '')
        classification_features['targetTitle'].append(len(title.split()) if title else 0)
        
        # targetParagraphs
        paragraphs = item.get('targetParagraphs', [])
        if isinstance(paragraphs, list):
            paragraph_text = ' '.join(paragraphs)
        else:
            paragraph_text = paragraphs or ''
        classification_features['targetParagraphs'].append(len(paragraph_text.split()) if paragraph_text else 0)
        
        # targetDescription
        description = item.get('targetDescription', '')
        classification_features['targetDescription'].append(len(description.split()) if description else 0)
        
        # targetKeywords
        keywords = item.get('targetKeywords', [])
        if isinstance(keywords, list):
            keyword_text = ' '.join(keywords)
        else:
            keyword_text = keywords or ''
        classification_features['targetKeywords'].append(len(keyword_text.split()) if keyword_text else 0)
    
    print(f"  Feature lengths (for SBERT embedding):")
    for feature, lengths in classification_features.items():
        print(f"    {feature}: trung bình {np.mean(lengths):.1f} từ")
    
    print(f"  Combined embedding dimension: 384 × 5 = 1920")
    
    # Tạo biểu đồ task-specific
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Task 1: Input vs Output length
    axes[0,0].scatter(generation_inputs, generation_outputs, alpha=0.6, color='blue')
    axes[0,0].set_title('Task 1: Input vs Output Length', fontsize=12, fontweight='bold')
    axes[0,0].set_xlabel('Input Length (words)')
    axes[0,0].set_ylabel('Output Length (words)')
    axes[0,0].grid(True, alpha=0.3)
    
    # Task 1: Input length distribution
    axes[0,1].hist(generation_inputs, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,1].axvline(512, color='red', linestyle='--', label='GPT-2 limit (512)')
    axes[0,1].axvline(np.mean(generation_inputs), color='orange', linestyle='--', 
                     label=f'Mean: {np.mean(generation_inputs):.1f}')
    axes[0,1].set_title('Task 1: Input Length Distribution', fontsize=12, fontweight='bold')
    axes[0,1].set_xlabel('Input Length (words)')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Task 2: Feature lengths comparison
    feature_names = list(classification_features.keys())
    feature_means = [np.mean(classification_features[f]) for f in feature_names]
    
    bars = axes[1,0].bar(range(len(feature_names)), feature_means, 
                        color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'pink'],
                        alpha=0.8, edgecolor='black')
    axes[1,0].set_title('Task 2: Average Feature Lengths', fontsize=12, fontweight='bold')
    axes[1,0].set_ylabel('Average Length (words)')
    axes[1,0].set_xticks(range(len(feature_names)))
    axes[1,0].set_xticklabels([f.replace('target', '') for f in feature_names], rotation=45)
    
    # Thêm giá trị lên bar
    for bar, mean_val in zip(bars, feature_means):
        axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                      f'{mean_val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Task 2: Feature length distributions (boxplot)
    feature_data = [classification_features[f] for f in feature_names]
    bp = axes[1,1].boxplot(feature_data, labels=[f.replace('target', '') for f in feature_names])
    axes[1,1].set_title('Task 2: Feature Length Distributions', fontsize=12, fontweight='bold')
    axes[1,1].set_ylabel('Length (words)')
    axes[1,1].tick_params(axis='x', rotation=45)
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/data_exploration/task_specific_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return generation_inputs, generation_outputs, classification_features

def create_summary_report(train_size, val_size, tag_counts, text_stats, missing_stats):
    """Tạo báo cáo tổng kết"""
    print(f"\n📋 BÁO CÁO TỔNG KẾT")
    print("=" * 70)
    
    print(f"📊 TỔNG QUAN:")
    print(f"  • Tổng số mẫu: {train_size + val_size}")
    print(f"  • Training: {train_size} mẫu")
    print(f"  • Validation: {val_size} mẫu")
    
    print(f"\n🏷️  PHÂN PHỐI LOẠI SPOILER:")
    total_tags = sum(tag_counts.values())
    for tag, count in tag_counts.most_common():
        print(f"  • {tag}: {count} ({count/total_tags*100:.1f}%)")
    
    print(f"\n📏 ĐỘ DÀI VĂN BẢN TRUNG BÌNH:")
    for field, lengths in text_stats.items():
        print(f"  • {field}: {np.mean(lengths):.1f} từ")
    
    print(f"\n❓ DỮ LIỆU THIẾU:")
    for field, stats in missing_stats.items():
        if stats['percentage'] > 0:
            print(f"  • {field}: {stats['percentage']:.1f}%")
    
    print(f"\n💡 KHUYẾN NGHỊ:")
    print(f"  • Task 1 (Generation): Sử dụng max_input_length=512, max_output_length=128")
    print(f"  • Task 2 (Classification): Xử lý missing values, sử dụng SBERT embedding 1920-dim")
    print(f"  • Cân bằng dữ liệu giữa các loại spoiler nếu cần")
    print(f"  • Fine-tune GPT-2 medium cho generation task")
    print(f"  • Test multiple ML algorithms cho classification task")

def main():
    """Hàm chính"""
    print("🔍 BẮT ĐẦU KHÁM PHÁ DỮ LIỆU CLICKBAIT SPOILER")
    print("=" * 80)
    
    # Tạo thư mục kết quả
    Path("results/data_exploration").mkdir(parents=True, exist_ok=True)
    
    # Load dữ liệu
    try:
        train_data = load_jsonl("data/raw/train.jsonl")
        val_data = load_jsonl("data/raw/validation.jsonl")
        print("✅ Đã load dữ liệu thành công!")
    except FileNotFoundError as e:
        print(f"❌ Không tìm thấy file dữ liệu: {e}")
        return
    
    # Phân tích từng khía cạnh
    train_size, val_size = basic_data_overview(train_data, val_data)
    tag_counts = analyze_spoiler_types(train_data)
    text_stats = analyze_text_lengths(train_data)
    spoiler_by_type = analyze_spoiler_by_type(train_data)
    missing_stats = analyze_missing_data(train_data)
    gen_inputs, gen_outputs, cls_features = analyze_task_specific_data(train_data)
    
    # Tạo báo cáo tổng kết
    create_summary_report(train_size, val_size, tag_counts, text_stats, missing_stats)
    
    print(f"\n{'='*80}")
    print("🎉 HOÀN THÀNH KHÁM PHÁ DỮ LIỆU!")
    print("📊 Đã tạo các biểu đồ chi tiết trong thư mục results/data_exploration/")
    print("📋 Có thể bắt đầu implement models dựa trên insights này")
    print(f"{'='*80}")

if __name__ == "__main__":
    main() 