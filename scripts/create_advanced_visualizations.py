#!/usr/bin/env python3
"""
Script tạo visualizations chi tiết và toàn diện cho dữ liệu đã tiền xử lý
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data():
    """Load processed data"""
    data = {}
    
    # Load Task 1 data
    for split in ['train', 'validation']:
        csv_file = f"data/processed/task1/{split}_processed.csv"
        if Path(csv_file).exists():
            data[f'task1_{split}'] = pd.read_csv(csv_file)
    
    # Load Task 2 data and embeddings
    for split in ['train', 'validation']:
        csv_file = f"data/processed/task2/{split}_processed.csv"
        emb_file = f"data/processed/task2/{split}_embeddings.pkl"
        
        if Path(csv_file).exists():
            data[f'task2_{split}'] = pd.read_csv(csv_file)
            
        if Path(emb_file).exists():
            with open(emb_file, 'rb') as f:
                data[f'embeddings_{split}'] = pickle.load(f)
    
    return data

def create_task1_visualizations(data):
    """Tạo visualizations cho Task 1"""
    print("📊 Creating Task 1 Visualizations...")
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('Task 1: Spoiler Generation - Comprehensive Analysis', fontsize=16, fontweight='bold')
    
    train_df = data['task1_train']
    val_df = data['task1_validation']
    
    # 1. Input Length Distribution with Statistics
    axes[0, 0].hist(train_df['input_length'], bins=50, alpha=0.7, label='Train', color='skyblue', density=True)
    axes[0, 0].hist(val_df['input_length'], bins=50, alpha=0.7, label='Validation', color='orange', density=True)
    axes[0, 0].axvline(512, color='red', linestyle='--', linewidth=2, label='Max Length (512)')
    axes[0, 0].axvline(train_df['input_length'].mean(), color='blue', linestyle=':', label=f"Train Mean ({train_df['input_length'].mean():.1f})")
    axes[0, 0].set_title('Input Length Distribution (Density)', fontweight='bold')
    axes[0, 0].set_xlabel('Input Length (tokens)')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Target Length Distribution
    axes[0, 1].hist(train_df['target_length'], bins=40, alpha=0.7, label='Train', color='lightgreen', density=True)
    axes[0, 1].hist(val_df['target_length'], bins=40, alpha=0.7, label='Validation', color='salmon', density=True)
    axes[0, 1].axvline(train_df['target_length'].mean(), color='green', linestyle=':', label=f"Train Mean ({train_df['target_length'].mean():.1f})")
    axes[0, 1].set_title('Target Length Distribution (Density)', fontweight='bold')
    axes[0, 1].set_xlabel('Target Length (tokens)')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Truncation Analysis
    train_truncated = (train_df['input_length'] >= 512).sum()
    val_truncated = (val_df['input_length'] >= 512).sum()
    
    truncation_data = {
        'Split': ['Train', 'Validation'],
        'Truncated': [train_truncated, val_truncated],
        'Not Truncated': [len(train_df) - train_truncated, len(val_df) - val_truncated]
    }
    
    x = np.arange(len(truncation_data['Split']))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, truncation_data['Not Truncated'], width, label='Not Truncated', color='lightgreen')
    axes[1, 0].bar(x + width/2, truncation_data['Truncated'], width, label='Truncated', color='lightcoral')
    axes[1, 0].set_title('Truncation Analysis', fontweight='bold')
    axes[1, 0].set_xlabel('Dataset Split')
    axes[1, 0].set_ylabel('Number of Samples')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(truncation_data['Split'])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add percentage annotations
    for i, (trunc, total) in enumerate(zip(truncation_data['Truncated'], [len(train_df), len(val_df)])):
        pct = trunc / total * 100
        axes[1, 0].text(i, trunc + 50, f'{pct:.1f}%', ha='center', fontweight='bold')
    
    # 4. Input vs Target Length Scatter
    sample_indices = np.random.choice(len(train_df), 1000, replace=False)
    sample_train = train_df.iloc[sample_indices]
    
    scatter = axes[1, 1].scatter(sample_train['input_length'], sample_train['target_length'], 
                                alpha=0.6, c=range(len(sample_train)), cmap='viridis')
    axes[1, 1].set_title('Input vs Target Length (Sample)', fontweight='bold')
    axes[1, 1].set_xlabel('Input Length (tokens)')
    axes[1, 1].set_ylabel('Target Length (tokens)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add correlation coefficient
    corr = train_df['input_length'].corr(train_df['target_length'])
    axes[1, 1].text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=axes[1, 1].transAxes, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 5. Length Statistics Comparison
    stats_data = {
        'Metric': ['Mean', 'Median', 'Std', 'Min', 'Max'],
        'Input Length (Train)': [
            train_df['input_length'].mean(),
            train_df['input_length'].median(),
            train_df['input_length'].std(),
            train_df['input_length'].min(),
            train_df['input_length'].max()
        ],
        'Target Length (Train)': [
            train_df['target_length'].mean(),
            train_df['target_length'].median(),
            train_df['target_length'].std(),
            train_df['target_length'].min(),
            train_df['target_length'].max()
        ]
    }
    
    stats_df = pd.DataFrame(stats_data)
    
    # Create table
    axes[2, 0].axis('tight')
    axes[2, 0].axis('off')
    table = axes[2, 0].table(cellText=[[f'{val:.1f}' for val in row] for row in stats_df.iloc[:, 1:].values],
                            rowLabels=stats_df['Metric'],
                            colLabels=stats_df.columns[1:],
                            cellLoc='center',
                            loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    axes[2, 0].set_title('Length Statistics Summary', fontweight='bold', pad=20)
    
    # 6. Token Length Percentiles
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    input_percentiles = [np.percentile(train_df['input_length'], p) for p in percentiles]
    target_percentiles = [np.percentile(train_df['target_length'], p) for p in percentiles]
    
    axes[2, 1].plot(percentiles, input_percentiles, 'o-', label='Input Length', linewidth=2, markersize=6)
    axes[2, 1].plot(percentiles, target_percentiles, 's-', label='Target Length', linewidth=2, markersize=6)
    axes[2, 1].set_title('Length Percentiles', fontweight='bold')
    axes[2, 1].set_xlabel('Percentile')
    axes[2, 1].set_ylabel('Length (tokens)')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/preprocessing_analysis/task1_comprehensive_analysis.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("✅ Task 1 comprehensive visualization saved!")

def create_task2_visualizations(data):
    """Tạo visualizations cho Task 2"""
    print("📊 Creating Task 2 Visualizations...")
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    fig.suptitle('Task 2: Spoiler Classification - Comprehensive Analysis', fontsize=16, fontweight='bold')
    
    train_df = data['task2_train']
    val_df = data['task2_validation']
    
    # 1. Label Distribution with Exact Numbers
    train_labels = train_df['spoiler_type'].value_counts()
    val_labels = val_df['spoiler_type'].value_counts()
    
    labels = train_labels.index
    x = np.arange(len(labels))
    width = 0.35
    
    train_bars = axes[0, 0].bar(x - width/2, train_labels.values, width, label='Train', color='skyblue')
    val_bars = axes[0, 0].bar(x + width/2, val_labels.values, width, label='Validation', color='orange')
    
    axes[0, 0].set_title('Label Distribution by Split', fontweight='bold')
    axes[0, 0].set_xlabel('Spoiler Type')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in train_bars:
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 10,
                       f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    for bar in val_bars:
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 10,
                       f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Text Length Distributions (Log Scale)
    text_columns = ['post_length', 'spoiler_length', 'target_length']
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    
    for i, (col, color) in enumerate(zip(text_columns, colors)):
        axes[0, 1].hist(train_df[col], bins=50, alpha=0.7, label=col.replace('_', ' ').title(), 
                       color=color, density=True)
    
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_title('Text Length Distributions (Log Scale)', fontweight='bold')
    axes[0, 1].set_xlabel('Length (words)')
    axes[0, 1].set_ylabel('Density (log scale)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Feature Correlation Heatmap
    numeric_cols = ['post_length', 'spoiler_length', 'target_length', 'keywords_count', 'has_description']
    corr_matrix = train_df[numeric_cols].corr()
    
    im = axes[0, 2].imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    axes[0, 2].set_xticks(range(len(numeric_cols)))
    axes[0, 2].set_yticks(range(len(numeric_cols)))
    axes[0, 2].set_xticklabels([col.replace('_', '\n') for col in numeric_cols], rotation=45)
    axes[0, 2].set_yticklabels([col.replace('_', '\n') for col in numeric_cols])
    axes[0, 2].set_title('Feature Correlation Matrix', fontweight='bold')
    
    # Add correlation values
    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            text = axes[0, 2].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)
    cbar.set_label('Correlation Coefficient')
    
    # 4. Box Plots by Label
    combined_df = pd.concat([train_df, val_df])
    
    box_data = []
    labels_list = []
    for label in combined_df['spoiler_type'].unique():
        mask = combined_df['spoiler_type'] == label
        box_data.append(combined_df[mask]['spoiler_length'])
        labels_list.append(f"{label}\n(n={mask.sum()})")
    
    axes[1, 0].boxplot(box_data, labels=labels_list)
    axes[1, 0].set_title('Spoiler Length by Type', fontweight='bold')
    axes[1, 0].set_ylabel('Spoiler Length (words)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Keywords Count Analysis
    axes[1, 1].hist(train_df['keywords_count'], bins=50, alpha=0.7, color='lightpink', density=True)
    axes[1, 1].axvline(train_df['keywords_count'].mean(), color='red', linestyle='--', 
                      label=f"Mean: {train_df['keywords_count'].mean():.1f}")
    axes[1, 1].axvline(train_df['keywords_count'].median(), color='blue', linestyle='--', 
                      label=f"Median: {train_df['keywords_count'].median():.1f}")
    axes[1, 1].set_title('Keywords Count Distribution', fontweight='bold')
    axes[1, 1].set_xlabel('Number of Keywords')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Description Availability
    desc_counts = train_df.groupby('spoiler_type')['has_description'].agg(['count', 'sum']).reset_index()
    desc_counts['percentage'] = desc_counts['sum'] / desc_counts['count'] * 100
    
    bars = axes[1, 2].bar(desc_counts['spoiler_type'], desc_counts['percentage'], 
                         color=['lightblue', 'lightgreen', 'lightcoral'])
    axes[1, 2].set_title('Description Availability by Type', fontweight='bold')
    axes[1, 2].set_xlabel('Spoiler Type')
    axes[1, 2].set_ylabel('Percentage with Description')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Add percentage labels
    for bar, pct in zip(bars, desc_counts['percentage']):
        axes[1, 2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                       f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 7. Text Length Relationships
    sample_indices = np.random.choice(len(train_df), 1000, replace=False)
    sample_train = train_df.iloc[sample_indices]
    
    scatter = axes[2, 0].scatter(sample_train['post_length'], sample_train['spoiler_length'], 
                                c=sample_train['target_length'], cmap='viridis', alpha=0.6)
    axes[2, 0].set_title('Post vs Spoiler Length\n(colored by Target Length)', fontweight='bold')
    axes[2, 0].set_xlabel('Post Length (words)')
    axes[2, 0].set_ylabel('Spoiler Length (words)')
    cbar = plt.colorbar(scatter, ax=axes[2, 0])
    cbar.set_label('Target Length')
    axes[2, 0].grid(True, alpha=0.3)
    
    # 8. Length Statistics by Label
    stats_by_label = train_df.groupby('spoiler_type')[['post_length', 'spoiler_length', 'target_length']].agg(['mean', 'std']).round(1)
    
    # Flatten column names
    stats_by_label.columns = [f'{col[0]}_{col[1]}' for col in stats_by_label.columns]
    
    axes[2, 1].axis('tight')
    axes[2, 1].axis('off')
    table = axes[2, 1].table(cellText=stats_by_label.values,
                            rowLabels=stats_by_label.index,
                            colLabels=stats_by_label.columns,
                            cellLoc='center',
                            loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    axes[2, 1].set_title('Length Statistics by Label', fontweight='bold', pad=20)
    
    # 9. Feature Distribution by Label
    label_colors = {'phrase': 'lightblue', 'passage': 'lightgreen', 'multi': 'lightcoral'}
    
    for label in train_df['spoiler_type'].unique():
        mask = train_df['spoiler_type'] == label
        axes[2, 2].hist(train_df[mask]['keywords_count'], bins=30, alpha=0.6, 
                       label=f'{label} (n={mask.sum()})', color=label_colors[label], density=True)
    
    axes[2, 2].set_title('Keywords Count by Label', fontweight='bold')
    axes[2, 2].set_xlabel('Number of Keywords')
    axes[2, 2].set_ylabel('Density')
    axes[2, 2].legend()
    axes[2, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/preprocessing_analysis/task2_comprehensive_analysis.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("✅ Task 2 comprehensive visualization saved!")

def create_embeddings_visualizations(data):
    """Tạo visualizations cho embeddings"""
    print("📊 Creating Embeddings Visualizations...")
    
    if 'embeddings_train' not in data:
        print("❌ No embeddings data found!")
        return
    
    embeddings = data['embeddings_train']
    train_df = data['task2_train']
    
    # Create PCA and t-SNE for each embedding type
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.suptitle('SBERT Embeddings Analysis - Dimensionality Reduction', fontsize=16, fontweight='bold')
    
    # Color mapping for labels
    label_colors = {'phrase': 0, 'passage': 1, 'multi': 2}
    colors = [label_colors[label] for label in train_df['spoiler_type']]
    
    # Sample data for visualization (t-SNE is computationally expensive)
    sample_size = min(1000, len(train_df))
    sample_indices = np.random.choice(len(train_df), sample_size, replace=False)
    
    embedding_fields = ['post_text', 'spoiler', 'target_paragraphs', 'target_keywords']
    
    for i, field in enumerate(embedding_fields):
        # PCA
        pca = PCA(n_components=2, random_state=42)
        pca_result = pca.fit_transform(embeddings[field][sample_indices])
        
        scatter = axes[0, i].scatter(pca_result[:, 0], pca_result[:, 1], 
                                   c=[colors[idx] for idx in sample_indices], 
                                   cmap='Set1', alpha=0.6, s=20)
        axes[0, i].set_title(f'PCA: {field.replace("_", " ").title()}\n'
                           f'Explained Variance: {pca.explained_variance_ratio_.sum():.1%}', 
                           fontweight='bold')
        axes[0, i].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        axes[0, i].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        axes[0, i].grid(True, alpha=0.3)
        
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        tsne_result = tsne.fit_transform(embeddings[field][sample_indices])
        
        scatter = axes[1, i].scatter(tsne_result[:, 0], tsne_result[:, 1], 
                                   c=[colors[idx] for idx in sample_indices], 
                                   cmap='Set1', alpha=0.6, s=20)
        axes[1, i].set_title(f't-SNE: {field.replace("_", " ").title()}', fontweight='bold')
        axes[1, i].set_xlabel('t-SNE 1')
        axes[1, i].set_ylabel('t-SNE 2')
        axes[1, i].grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=plt.cm.Set1(i), label=label) 
                      for i, label in enumerate(['phrase', 'passage', 'multi'])]
    fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(0.98, 0.5))
    
    plt.tight_layout()
    plt.savefig('results/preprocessing_analysis/embeddings_analysis.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("✅ Embeddings visualization saved!")
    
    # Create embeddings statistics visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('SBERT Embeddings - Statistical Analysis', fontsize=16, fontweight='bold')
    
    # 1. Embedding norms by field
    norms_data = {}
    for field in embedding_fields:
        norms = np.linalg.norm(embeddings[field], axis=1)
        norms_data[field] = norms
    
    axes[0, 0].boxplot(norms_data.values(), labels=[f.replace('_', '\n') for f in norms_data.keys()])
    axes[0, 0].set_title('Embedding Norms by Field', fontweight='bold')
    axes[0, 0].set_ylabel('L2 Norm')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Pairwise similarities
    from sklearn.metrics.pairwise import cosine_similarity
    
    similarity_matrix = np.zeros((len(embedding_fields), len(embedding_fields)))
    for i, field1 in enumerate(embedding_fields):
        for j, field2 in enumerate(embedding_fields):
            # Sample for computational efficiency
            sample_indices = np.random.choice(len(embeddings[field1]), min(500, len(embeddings[field1])), replace=False)
            sim = cosine_similarity(embeddings[field1][sample_indices], embeddings[field2][sample_indices])
            similarity_matrix[i, j] = np.mean(np.diag(sim))
    
    im = axes[0, 1].imshow(similarity_matrix, cmap='RdYlBu_r', vmin=0, vmax=1)
    axes[0, 1].set_xticks(range(len(embedding_fields)))
    axes[0, 1].set_yticks(range(len(embedding_fields)))
    axes[0, 1].set_xticklabels([f.replace('_', '\n') for f in embedding_fields], rotation=45)
    axes[0, 1].set_yticklabels([f.replace('_', '\n') for f in embedding_fields])
    axes[0, 1].set_title('Cross-Field Cosine Similarity', fontweight='bold')
    
    # Add similarity values
    for i in range(len(embedding_fields)):
        for j in range(len(embedding_fields)):
            text = axes[0, 1].text(j, i, f'{similarity_matrix[i, j]:.3f}',
                                 ha="center", va="center", color="black", fontweight='bold')
    
    cbar = plt.colorbar(im, ax=axes[0, 1])
    cbar.set_label('Cosine Similarity')
    
    # 3. Embedding dimensions analysis
    variances = []
    for field in embedding_fields:
        var = np.var(embeddings[field], axis=0)
        variances.append(var)
    
    for i, (field, var) in enumerate(zip(embedding_fields, variances)):
        axes[1, 0].plot(var, alpha=0.7, label=field.replace('_', ' ').title())
    
    axes[1, 0].set_title('Embedding Dimensions Variance', fontweight='bold')
    axes[1, 0].set_xlabel('Dimension Index')
    axes[1, 0].set_ylabel('Variance')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Embeddings summary statistics
    stats_data = []
    for field in embedding_fields:
        emb = embeddings[field]
        stats_data.append([
            field.replace('_', ' ').title(),
            f'{emb.mean():.4f}',
            f'{emb.std():.4f}',
            f'{emb.min():.4f}',
            f'{emb.max():.4f}',
            f'{np.linalg.norm(emb, axis=1).mean():.4f}'
        ])
    
    axes[1, 1].axis('tight')
    axes[1, 1].axis('off')
    table = axes[1, 1].table(cellText=stats_data,
                            colLabels=['Field', 'Mean', 'Std', 'Min', 'Max', 'Avg Norm'],
                            cellLoc='center',
                            loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    axes[1, 1].set_title('Embeddings Summary Statistics', fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('results/preprocessing_analysis/embeddings_statistics.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("✅ Embeddings statistics visualization saved!")

def create_comparison_visualizations(data):
    """Tạo visualizations so sánh train/validation"""
    print("📊 Creating Train/Validation Comparison...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Train vs Validation Comparison', fontsize=16, fontweight='bold')
    
    # Task 1 comparisons
    train_t1 = data['task1_train']
    val_t1 = data['task1_validation']
    
    # 1. Input length comparison
    axes[0, 0].hist(train_t1['input_length'], bins=50, alpha=0.7, label='Train', density=True, color='skyblue')
    axes[0, 0].hist(val_t1['input_length'], bins=50, alpha=0.7, label='Validation', density=True, color='orange')
    axes[0, 0].set_title('Task 1: Input Length Distribution', fontweight='bold')
    axes[0, 0].set_xlabel('Input Length (tokens)')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Target length comparison
    axes[0, 1].hist(train_t1['target_length'], bins=30, alpha=0.7, label='Train', density=True, color='lightgreen')
    axes[0, 1].hist(val_t1['target_length'], bins=30, alpha=0.7, label='Validation', density=True, color='salmon')
    axes[0, 1].set_title('Task 1: Target Length Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Target Length (tokens)')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Task 1 Labels comparison
    train_labels_t1 = train_t1['spoiler_type'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    val_labels_t1 = val_t1['spoiler_type'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    
    # Flatten labels
    train_flat = []
    val_flat = []
    for labels in train_labels_t1:
        if isinstance(labels, list):
            train_flat.extend(labels)
    for labels in val_labels_t1:
        if isinstance(labels, list):
            val_flat.extend(labels)
    
    from collections import Counter
    train_counts = Counter(train_flat)
    val_counts = Counter(val_flat)
    
    all_labels = list(set(train_flat + val_flat))
    train_values = [train_counts.get(label, 0) for label in all_labels]
    val_values = [val_counts.get(label, 0) for label in all_labels]
    
    x = np.arange(len(all_labels))
    width = 0.35
    
    axes[0, 2].bar(x - width/2, train_values, width, label='Train', color='skyblue')
    axes[0, 2].bar(x + width/2, val_values, width, label='Validation', color='orange')
    axes[0, 2].set_title('Task 1: Label Distribution', fontweight='bold')
    axes[0, 2].set_xlabel('Spoiler Type')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(all_labels)
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Task 2 comparisons
    train_t2 = data['task2_train']
    val_t2 = data['task2_validation']
    
    # 4. Spoiler length comparison
    axes[1, 0].hist(train_t2['spoiler_length'], bins=40, alpha=0.7, label='Train', density=True, color='lightblue')
    axes[1, 0].hist(val_t2['spoiler_length'], bins=40, alpha=0.7, label='Validation', density=True, color='lightcoral')
    axes[1, 0].set_title('Task 2: Spoiler Length Distribution', fontweight='bold')
    axes[1, 0].set_xlabel('Spoiler Length (words)')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Keywords count comparison
    axes[1, 1].hist(train_t2['keywords_count'], bins=50, alpha=0.7, label='Train', density=True, color='lightgreen')
    axes[1, 1].hist(val_t2['keywords_count'], bins=50, alpha=0.7, label='Validation', density=True, color='lightyellow')
    axes[1, 1].set_title('Task 2: Keywords Count Distribution', fontweight='bold')
    axes[1, 1].set_xlabel('Number of Keywords')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Task 2 Labels comparison
    train_labels_t2 = train_t2['spoiler_type'].value_counts()
    val_labels_t2 = val_t2['spoiler_type'].value_counts()
    
    labels = train_labels_t2.index
    x = np.arange(len(labels))
    width = 0.35
    
    train_bars = axes[1, 2].bar(x - width/2, train_labels_t2.values, width, label='Train', color='lightblue')
    val_bars = axes[1, 2].bar(x + width/2, val_labels_t2.values, width, label='Validation', color='lightcoral')
    
    axes[1, 2].set_title('Task 2: Label Distribution', fontweight='bold')
    axes[1, 2].set_xlabel('Spoiler Type')
    axes[1, 2].set_ylabel('Count')
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(labels)
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    # Add percentage annotations
    for bar, total in zip(train_bars, [len(train_t2)] * len(labels)):
        height = bar.get_height()
        pct = height / total * 100
        axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 20,
                       f'{pct:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    for bar, total in zip(val_bars, [len(val_t2)] * len(labels)):
        height = bar.get_height()
        pct = height / total * 100
        axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 20,
                       f'{pct:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/preprocessing_analysis/train_validation_comparison.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("✅ Train/Validation comparison visualization saved!")

def main():
    """Hàm chính"""
    print("🎨 CREATING ADVANCED VISUALIZATIONS")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path("results/preprocessing_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("📂 Loading processed data...")
    data = load_data()
    
    if not data:
        print("❌ No data found! Please run preprocessing first.")
        return
    
    # Create visualizations
    create_task1_visualizations(data)
    create_task2_visualizations(data)
    create_embeddings_visualizations(data)
    create_comparison_visualizations(data)
    
    print(f"\n{'='*60}")
    print("🎉 ALL VISUALIZATIONS COMPLETED!")
    print("📁 Saved files:")
    print("  📊 task1_comprehensive_analysis.png")
    print("  📊 task2_comprehensive_analysis.png")
    print("  📊 embeddings_analysis.png")
    print("  📊 embeddings_statistics.png")
    print("  📊 train_validation_comparison.png")
    print(f"📂 Location: {output_dir.absolute()}")

if __name__ == "__main__":
    main() 