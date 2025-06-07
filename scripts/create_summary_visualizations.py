#!/usr/bin/env python3
"""
Script tạo biểu đồ tổng hợp và summary cho báo cáo
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

def create_comprehensive_summary(train_data, val_data):
    """Tạo biểu đồ tổng hợp toàn diện"""
    
    # Combine data for overall analysis
    all_data = train_data + val_data
    
    # Create a comprehensive figure
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Dataset Overview (top left)
    ax1 = plt.subplot(3, 4, 1)
    sizes = [len(train_data), len(val_data)]
    labels = ['Training\n(3,200)', 'Validation\n(800)']
    colors = ['#ff9999', '#66b3ff']
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax1.set_title('Dataset Split', fontsize=14, fontweight='bold')
    
    # 2. Spoiler Types Distribution (top center-left)
    ax2 = plt.subplot(3, 4, 2)
    all_tags = []
    for item in train_data:
        tags = item.get('tags', [])
        if isinstance(tags, list):
            all_tags.extend(tags)
        else:
            all_tags.append(tags)
    
    tag_counts = Counter(all_tags)
    labels, sizes = zip(*tag_counts.most_common())
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    ax2.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax2.set_title('Spoiler Types', fontsize=14, fontweight='bold')
    
    # 3. Text Length Comparison (top center-right)
    ax3 = plt.subplot(3, 4, 3)
    
    # Calculate lengths
    post_lengths = []
    spoiler_lengths = []
    title_lengths = []
    
    for item in train_data:
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
    
    means = [np.mean(post_lengths), np.mean(spoiler_lengths), np.mean(title_lengths)]
    fields = ['Post Text', 'Spoiler', 'Target Title']
    
    bars = ax3.bar(fields, means, color=['skyblue', 'lightcoral', 'lightgreen'], alpha=0.8)
    ax3.set_title('Average Text Lengths', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Words')
    
    # Add values on bars
    for bar, mean_val in zip(bars, means):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{mean_val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Missing Data Overview (top right)
    ax4 = plt.subplot(3, 4, 4)
    
    fields_to_check = ['targetDescription', 'targetKeywords']
    missing_percentages = []
    
    for field in fields_to_check:
        missing_count = 0
        for item in train_data:
            value = item.get(field, None)
            if value is None or value == '' or value == []:
                missing_count += 1
            elif isinstance(value, str) and value.strip() == '':
                missing_count += 1
        missing_percentages.append((missing_count / len(train_data)) * 100)
    
    bars = ax4.bar(fields_to_check, missing_percentages, color='orange', alpha=0.7)
    ax4.set_title('Missing Data (%)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Percentage')
    ax4.tick_params(axis='x', rotation=45)
    
    for bar, pct in zip(bars, missing_percentages):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 5. Task 1 Analysis - Input Length Distribution (middle left)
    ax5 = plt.subplot(3, 4, 5)
    
    generation_inputs = []
    for item in train_data:
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
    
    ax5.hist(generation_inputs, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax5.axvline(512, color='red', linestyle='--', linewidth=2, label='GPT-2 limit (512)')
    ax5.axvline(np.mean(generation_inputs), color='orange', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(generation_inputs):.1f}')
    ax5.set_title('Task 1: Input Length Distribution', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Input Length (words)')
    ax5.set_ylabel('Frequency')
    ax5.legend()
    
    # 6. Task 1 Analysis - Output Length Distribution (middle center-left)
    ax6 = plt.subplot(3, 4, 6)
    
    generation_outputs = []
    for item in train_data:
        spoiler = item.get('spoiler', [])
        if isinstance(spoiler, list):
            spoiler = ' '.join(spoiler)
        generation_outputs.append(len(spoiler.split()) if spoiler else 0)
    
    ax6.hist(generation_outputs, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    ax6.axvline(128, color='red', linestyle='--', linewidth=2, label='Target limit (128)')
    ax6.axvline(np.mean(generation_outputs), color='orange', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(generation_outputs):.1f}')
    ax6.set_title('Task 1: Output Length Distribution', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Output Length (words)')
    ax6.set_ylabel('Frequency')
    ax6.legend()
    
    # 7. Task 2 Analysis - Feature Lengths (middle center-right)
    ax7 = plt.subplot(3, 4, 7)
    
    feature_lengths = {
        'postText': [],
        'targetTitle': [],
        'targetDescription': [],
        'targetKeywords': []
    }
    
    for item in train_data:
        # postText
        post_text = item.get('postText', [])
        if isinstance(post_text, list):
            post_text = ' '.join(post_text)
        feature_lengths['postText'].append(len(post_text.split()) if post_text else 0)
        
        # targetTitle
        title = item.get('targetTitle', '')
        feature_lengths['targetTitle'].append(len(title.split()) if title else 0)
        
        # targetDescription
        description = item.get('targetDescription', '')
        feature_lengths['targetDescription'].append(len(description.split()) if description else 0)
        
        # targetKeywords
        keywords = item.get('targetKeywords', '')
        feature_lengths['targetKeywords'].append(len(keywords.split()) if keywords else 0)
    
    feature_names = list(feature_lengths.keys())
    feature_means = [np.mean(feature_lengths[f]) for f in feature_names]
    
    bars = ax7.bar(range(len(feature_names)), feature_means,
                  color=['skyblue', 'lightcoral', 'lightgreen', 'gold'], alpha=0.8)
    ax7.set_title('Task 2: Feature Lengths', fontsize=12, fontweight='bold')
    ax7.set_ylabel('Average Length (words)')
    ax7.set_xticks(range(len(feature_names)))
    ax7.set_xticklabels([f.replace('target', '') for f in feature_names], rotation=45)
    
    for bar, mean_val in zip(bars, feature_means):
        ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{mean_val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 8. Spoiler Length by Type (middle right)
    ax8 = plt.subplot(3, 4, 8)
    
    spoiler_by_type = defaultdict(list)
    for item in train_data:
        tags = item.get('tags', [])
        spoiler = item.get('spoiler', [])
        
        if isinstance(spoiler, list):
            spoiler_text = ' '.join(spoiler)
        else:
            spoiler_text = spoiler or ''
        
        spoiler_length = len(spoiler_text.split()) if spoiler_text else 0
        
        if isinstance(tags, list):
            for tag in tags:
                spoiler_by_type[tag].append(spoiler_length)
        else:
            spoiler_by_type[tags].append(spoiler_length)
    
    type_names = list(spoiler_by_type.keys())
    length_data = [spoiler_by_type[t] for t in type_names]
    
    bp = ax8.boxplot(length_data, labels=type_names, patch_artist=True)
    colors = ['lightblue', 'lightcoral', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax8.set_title('Spoiler Length by Type', fontsize=12, fontweight='bold')
    ax8.set_ylabel('Length (words)')
    
    # 9-12. Key Statistics Summary (bottom row)
    
    # 9. Task 1 Statistics
    ax9 = plt.subplot(3, 4, 9)
    ax9.axis('off')
    
    task1_stats = f"""
TASK 1: SPOILER GENERATION

Input (postText + targetParagraphs):
• Average: {np.mean(generation_inputs):.1f} words
• ≤ 512 tokens: {np.mean([l <= 512 for l in generation_inputs])*100:.1f}%

Output (spoiler):
• Average: {np.mean(generation_outputs):.1f} words  
• ≤ 128 tokens: {np.mean([l <= 128 for l in generation_outputs])*100:.1f}%

Model: GPT-2 Medium fine-tuned
Evaluation: BLEU, ROUGE, BERTScore, METEOR
"""
    
    ax9.text(0.05, 0.95, task1_stats, transform=ax9.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 10. Task 2 Statistics
    ax10 = plt.subplot(3, 4, 10)
    ax10.axis('off')
    
    task2_stats = f"""
TASK 2: SPOILER TYPE CLASSIFICATION

Features: 5 text fields
• postText: {np.mean(feature_lengths['postText']):.1f} words
• targetTitle: {np.mean(feature_lengths['targetTitle']):.1f} words
• targetDescription: {np.mean(feature_lengths['targetDescription']):.1f} words
• targetKeywords: {np.mean(feature_lengths['targetKeywords']):.1f} words

Embedding: SBERT (384 × 5 = 1920 dim)
Models: 8 ML algorithms (best: SVM)
Evaluation: Accuracy, Precision, Recall, F1, MCC
"""
    
    ax10.text(0.05, 0.95, task2_stats, transform=ax10.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # 11. Dataset Quality
    ax11 = plt.subplot(3, 4, 11)
    ax11.axis('off')
    
    quality_stats = f"""
DATASET QUALITY

✅ Complete data:
• postText: 100%
• spoiler: 100%
• targetParagraphs: 100%

⚠️ Missing data:
• targetDescription: {missing_percentages[0]:.1f}%
• targetKeywords: {missing_percentages[1]:.1f}%

Class distribution:
• phrase: {tag_counts['phrase']} (42.7%)
• passage: {tag_counts['passage']} (39.8%)
• multi: {tag_counts['multi']} (17.5%)
"""
    
    ax11.text(0.05, 0.95, quality_stats, transform=ax11.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # 12. Implementation Recommendations
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')
    
    recommendations = f"""
IMPLEMENTATION RECOMMENDATIONS

Task 1 (Generation):
• max_input_length = 512
• max_target_length = 128
• Learning rate = 5e-5
• Batch size = 8

Task 2 (Classification):
• Handle missing values
• Use SBERT embeddings
• Test 8 ML algorithms
• Focus on SVM

Data preprocessing:
• Text cleaning & normalization
• Missing value imputation
• Class balancing if needed
"""
    
    ax12.text(0.05, 0.95, recommendations, transform=ax12.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    plt.suptitle('Clickbait Spoiler Dataset - Comprehensive Analysis Summary', 
                 fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    plt.savefig('results/data_exploration/comprehensive_summary.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_task_comparison_chart(train_data):
    """Tạo biểu đồ so sánh hai task"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Task 1: Input-Output relationship
    generation_inputs = []
    generation_outputs = []
    
    for item in train_data:
        # Input
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
        
        # Output
        spoiler = item.get('spoiler', [])
        if isinstance(spoiler, list):
            spoiler = ' '.join(spoiler)
        generation_outputs.append(len(spoiler.split()) if spoiler else 0)
    
    # Scatter plot with density
    ax1.scatter(generation_inputs, generation_outputs, alpha=0.6, color='blue', s=20)
    ax1.set_title('Task 1: Input vs Output Length Relationship', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Input Length (words)')
    ax1.set_ylabel('Output Length (words)')
    ax1.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    correlation = np.corrcoef(generation_inputs, generation_outputs)[0, 1]
    ax1.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=ax1.transAxes,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Task 1: Length distribution comparison
    ax2.hist(generation_inputs, bins=50, alpha=0.7, label='Input', color='skyblue', density=True)
    ax2.hist(generation_outputs, bins=50, alpha=0.7, label='Output', color='lightcoral', density=True)
    ax2.set_title('Task 1: Input vs Output Length Distributions', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Length (words)')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Task 2: Feature importance visualization
    feature_lengths = {
        'postText': [],
        'targetTitle': [],
        'targetParagraphs': [],
        'targetDescription': [],
        'targetKeywords': []
    }
    
    for item in train_data:
        # postText
        post_text = item.get('postText', [])
        if isinstance(post_text, list):
            post_text = ' '.join(post_text)
        feature_lengths['postText'].append(len(post_text.split()) if post_text else 0)
        
        # targetTitle
        title = item.get('targetTitle', '')
        feature_lengths['targetTitle'].append(len(title.split()) if title else 0)
        
        # targetParagraphs
        paragraphs = item.get('targetParagraphs', [])
        if isinstance(paragraphs, list):
            paragraph_text = ' '.join(paragraphs)
        else:
            paragraph_text = paragraphs or ''
        feature_lengths['targetParagraphs'].append(len(paragraph_text.split()) if paragraph_text else 0)
        
        # targetDescription
        description = item.get('targetDescription', '')
        feature_lengths['targetDescription'].append(len(description.split()) if description else 0)
        
        # targetKeywords
        keywords = item.get('targetKeywords', '')
        feature_lengths['targetKeywords'].append(len(keywords.split()) if keywords else 0)
    
    # Box plot for feature comparison
    feature_names = list(feature_lengths.keys())
    feature_data = [feature_lengths[f] for f in feature_names]
    
    bp = ax3.boxplot(feature_data, labels=[f.replace('target', '') for f in feature_names], patch_artist=True)
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'pink']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax3.set_title('Task 2: Feature Length Distributions', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Length (words)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Task comparison summary
    ax4.axis('off')
    
    comparison_text = f"""
TASK COMPARISON SUMMARY

TASK 1: SPOILER GENERATION
• Objective: Generate spoiler text
• Input: postText + targetParagraphs ({np.mean(generation_inputs):.1f} words avg)
• Output: spoiler ({np.mean(generation_outputs):.1f} words avg)
• Model: GPT-2 Medium (fine-tuned)
• Challenge: Variable input length, need truncation

TASK 2: SPOILER TYPE CLASSIFICATION  
• Objective: Classify spoiler type (phrase/passage/multi)
• Input: 5 text features → SBERT embeddings (1920-dim)
• Output: 3 classes (imbalanced: 42.7%, 39.8%, 17.5%)
• Model: SBERT + ML classifiers (SVM expected best)
• Challenge: Missing values, class imbalance

KEY INSIGHTS:
• Input-output correlation: {correlation:.3f} (weak)
• targetParagraphs dominates feature length
• Missing data mainly in targetKeywords (41.1%)
• Both tasks feasible with current dataset quality
"""
    
    ax4.text(0.05, 0.95, comparison_text, transform=ax4.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('results/data_exploration/task_comparison_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Hàm chính"""
    print("🎨 TẠO BIỂU ĐỒ TỔNG HỢP VÀ SUMMARY")
    print("=" * 60)
    
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
    
    # Tạo biểu đồ tổng hợp
    print("📊 Đang tạo biểu đồ tổng hợp...")
    create_comprehensive_summary(train_data, val_data)
    
    print("📈 Đang tạo biểu đồ so sánh task...")
    create_task_comparison_chart(train_data)
    
    print("\n" + "=" * 60)
    print("🎉 HOÀN THÀNH TẠO BIỂU ĐỒ TỔNG HỢP!")
    print("📊 Đã tạo thêm 2 biểu đồ summary trong results/data_exploration/:")
    print("  • comprehensive_summary.png - Tổng hợp toàn diện")
    print("  • task_comparison_analysis.png - So sánh hai task")
    print("=" * 60)

if __name__ == "__main__":
    main() 