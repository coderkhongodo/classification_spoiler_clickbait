#!/usr/bin/env python3
"""
Script preprocessing dữ liệu cho cả Task 1 (Spoiler Generation) và Task 2 (Spoiler Classification)
"""

import json
import pandas as pd
from pathlib import Path
from transformers import GPT2Tokenizer, AutoTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
from tqdm import tqdm
import torch
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.gpt2_tokenizer = None
        self.sbert_model = None
        self.setup_models()
        
    def setup_models(self):
        """Khởi tạo các models cần thiết"""
        logger.info("🔧 Khởi tạo tokenizers và models...")
        
        # GPT-2 tokenizer cho Task 1
        logger.info("📝 Loading GPT-2 tokenizer...")
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
        self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token
        
        # SBERT model cho Task 2
        logger.info("🧠 Loading SBERT model...")
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        logger.info("✅ Models loaded successfully!")
        
    def load_data(self, file_path):
        """Load dữ liệu từ JSONL file"""
        logger.info(f"📂 Loading data from {file_path}")
        
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        
        logger.info(f"✅ Loaded {len(data)} samples")
        return data
    
    def create_task1_features(self, data):
        """Tạo features cho Task 1 (Spoiler Generation)"""
        logger.info("🔨 Creating Task 1 features...")
        
        processed_data = []
        
        for item in tqdm(data, desc="Processing Task 1"):
            # Input: postText + targetParagraph
            post_text = item.get('postText', [])
            target_paragraphs = item.get('targetParagraphs', [])
            spoiler = item.get('spoiler', [])
            
            # Combine post text
            if isinstance(post_text, list):
                post_combined = ' '.join(post_text)
            else:
                post_combined = str(post_text)
            
            # Combine target paragraphs
            if isinstance(target_paragraphs, list):
                target_combined = ' '.join(target_paragraphs)
            else:
                target_combined = str(target_paragraphs)
            
            # Combine spoiler
            if isinstance(spoiler, list):
                spoiler_combined = ' '.join(spoiler)
            else:
                spoiler_combined = str(spoiler)
            
            # Create input text for GPT-2
            input_text = f"POST: {post_combined} TARGET: {target_combined} SPOILER:"
            
            # Tokenize
            input_tokens = self.gpt2_tokenizer.encode(input_text, max_length=512, truncation=True)
            target_tokens = self.gpt2_tokenizer.encode(spoiler_combined, max_length=128, truncation=True)
            
            processed_item = {
                'id': item.get('id'),
                'post_text': post_combined,
                'target_paragraphs': target_combined,
                'spoiler': spoiler_combined,
                'input_text': input_text,
                'input_tokens': input_tokens,
                'target_tokens': target_tokens,
                'input_length': len(input_tokens),
                'target_length': len(target_tokens),
                'spoiler_type': item.get('tags', [])
            }
            
            processed_data.append(processed_item)
        
        logger.info(f"✅ Created {len(processed_data)} Task 1 samples")
        return processed_data
    
    def create_task2_features(self, data):
        """Tạo features cho Task 2 (Spoiler Classification)"""
        logger.info("🔨 Creating Task 2 features...")
        
        processed_data = []
        
        for item in tqdm(data, desc="Processing Task 2"):
            post_text = item.get('postText', [])
            spoiler = item.get('spoiler', [])
            target_paragraphs = item.get('targetParagraphs', [])
            target_keywords = item.get('targetKeywords', [])
            target_description = item.get('targetDescription', '')
            
            # Text features
            if isinstance(post_text, list):
                post_combined = ' '.join(post_text)
            else:
                post_combined = str(post_text)
                
            if isinstance(spoiler, list):
                spoiler_combined = ' '.join(spoiler)
            else:
                spoiler_combined = str(spoiler)
                
            if isinstance(target_paragraphs, list):
                target_para_combined = ' '.join(target_paragraphs)
            else:
                target_para_combined = str(target_paragraphs)
                
            if isinstance(target_keywords, list):
                keywords_combined = ' '.join(target_keywords) if target_keywords else ''
            else:
                keywords_combined = str(target_keywords) if target_keywords else ''
            
            # Numerical features
            post_length = len(post_combined.split())
            spoiler_length = len(spoiler_combined.split())
            target_length = len(target_para_combined.split())
            keywords_count = len(target_keywords) if target_keywords else 0
            has_description = 1 if target_description else 0
            
            # Spoiler type (label)
            spoiler_types = item.get('tags', [])
            if 'phrase' in spoiler_types:
                label = 'phrase'
            elif 'passage' in spoiler_types:
                label = 'passage'  
            elif 'multi' in spoiler_types:
                label = 'multi'
            else:
                label = 'phrase'  # default
            
            processed_item = {
                'id': item.get('id'),
                'post_text': post_combined,
                'spoiler': spoiler_combined,
                'target_paragraphs': target_para_combined,
                'target_keywords': keywords_combined,
                'target_description': target_description,
                'post_length': post_length,
                'spoiler_length': spoiler_length,
                'target_length': target_length,
                'keywords_count': keywords_count,
                'has_description': has_description,
                'spoiler_type': label
            }
            
            processed_data.append(processed_item)
        
        logger.info(f"✅ Created {len(processed_data)} Task 2 samples")
        return processed_data
    
    def create_embeddings(self, processed_data, text_fields):
        """Tạo SBERT embeddings cho các text fields"""
        logger.info("🧠 Creating SBERT embeddings...")
        
        embeddings = {}
        
        for field in text_fields:
            logger.info(f"📊 Creating embeddings for: {field}")
            texts = [item[field] for item in processed_data]
            
            # Tạo embeddings với batch processing
            field_embeddings = self.sbert_model.encode(
                texts,
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            embeddings[field] = field_embeddings
            logger.info(f"✅ Created embeddings for {field}: {field_embeddings.shape}")
        
        return embeddings
    
    def save_processed_data(self, data, embeddings, output_dir, split_name):
        """Lưu processed data và embeddings"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save processed data
        data_file = output_path / f"{split_name}_processed.json"
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 Saved processed data: {data_file}")
        
        # Save embeddings
        embeddings_file = None
        if embeddings:
            embeddings_file = output_path / f"{split_name}_embeddings.pkl"
            with open(embeddings_file, 'wb') as f:
                pickle.dump(embeddings, f)
            logger.info(f"💾 Saved embeddings: {embeddings_file}")
        
        # Save as pandas DataFrame for easier analysis
        df = pd.DataFrame(data)
        csv_file = output_path / f"{split_name}_processed.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8')
        logger.info(f"💾 Saved CSV: {csv_file}")
        
        return data_file, embeddings_file, csv_file

def main():
    """Hàm chính"""
    logger.info("🚀 STARTING DATA PREPROCESSING")
    logger.info("=" * 60)
    
    # Khởi tạo preprocessor
    preprocessor = DataPreprocessor()
    
    # Files to process
    files_to_process = [
        ("data/raw/train.jsonl", "train"),
        ("data/raw/validation.jsonl", "validation")
    ]
    
    for file_path, split_name in files_to_process:
        if not Path(file_path).exists():
            logger.warning(f"⚠️  File không tồn tại: {file_path}")
            continue
            
        logger.info(f"\n{'='*60}")
        logger.info(f"📁 Processing {split_name} data")
        logger.info(f"{'='*60}")
        
        # Load raw data
        raw_data = preprocessor.load_data(file_path)
        
        # Process Task 1 data
        logger.info("\n🎯 TASK 1: Spoiler Generation")
        task1_data = preprocessor.create_task1_features(raw_data)
        
        # Save Task 1 data
        preprocessor.save_processed_data(
            task1_data, 
            None,  # No embeddings for Task 1
            f"data/processed/task1",
            split_name
        )
        
        # Process Task 2 data
        logger.info("\n🎯 TASK 2: Spoiler Classification")
        task2_data = preprocessor.create_task2_features(raw_data)
        
        # Create embeddings for Task 2
        text_fields = ['post_text', 'spoiler', 'target_paragraphs', 'target_keywords']
        embeddings = preprocessor.create_embeddings(task2_data, text_fields)
        
        # Save Task 2 data
        preprocessor.save_processed_data(
            task2_data,
            embeddings,
            f"data/processed/task2", 
            split_name
        )
    
    # Tạo summary report
    logger.info(f"\n{'='*60}")
    logger.info("📋 PREPROCESSING SUMMARY")
    logger.info(f"{'='*60}")
    
    # Check output directories
    task1_dir = Path("data/processed/task1")
    task2_dir = Path("data/processed/task2")
    
    if task1_dir.exists():
        task1_files = list(task1_dir.glob("*"))
        logger.info(f"\n📁 Task 1 Output Files ({len(task1_files)}):")
        for f in task1_files:
            logger.info(f"  📄 {f.name} ({f.stat().st_size / 1024:.1f} KB)")
    
    if task2_dir.exists():
        task2_files = list(task2_dir.glob("*"))
        logger.info(f"\n📁 Task 2 Output Files ({len(task2_files)}):")
        for f in task2_files:
            logger.info(f"  📄 {f.name} ({f.stat().st_size / 1024:.1f} KB)")
    
    logger.info(f"\n🎉 PREPROCESSING COMPLETED!")
    logger.info("🔸 Task 1 data ready for GPT-2 fine-tuning")
    logger.info("🔸 Task 2 data ready for classification training")

if __name__ == "__main__":
    main() 