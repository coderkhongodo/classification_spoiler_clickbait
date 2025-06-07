#!/usr/bin/env python3
"""
Training script cho Task 2: SBERT + ML Classification
"""

import pandas as pd
import numpy as np
import pickle
import json
import yaml
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import logging
from datetime import datetime
import wandb
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SpoilerClassifier:
    def __init__(self, config_path="configs/spoiler_classification_config.yaml"):
        """Initialize classifier với config"""
        self.config = self.load_config(config_path)
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.results = {}
        
        # Setup experiment tracking
        if self.config.get('use_wandb', False):
            wandb.init(
                project=self.config.get('wandb_project', 'clickbait-spoiler-classification'),
                config=self.config
            )
    
    def load_config(self, config_path):
        """Load configuration file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def load_data(self):
        """Load processed data và embeddings"""
        logger.info("📂 Loading processed data và embeddings...")
        
        # Load CSV data
        train_df = pd.read_csv("data/processed/task2/train_processed.csv")
        val_df = pd.read_csv("data/processed/task2/validation_processed.csv")
        
        # Load embeddings
        with open("data/processed/task2/train_embeddings.pkl", 'rb') as f:
            train_embeddings = pickle.load(f)
        
        with open("data/processed/task2/validation_embeddings.pkl", 'rb') as f:
            val_embeddings = pickle.load(f)
        
        logger.info(f"✅ Loaded {len(train_df)} training samples, {len(val_df)} validation samples")
        
        return train_df, val_df, train_embeddings, val_embeddings
    
    def prepare_features(self, df, embeddings):
        """Prepare combined features từ embeddings và numerical features"""
        logger.info("🔧 Preparing combined features...")
        
        # Numerical features
        numerical_features = [
            'post_length', 'spoiler_length', 'target_length', 
            'keywords_count', 'has_description'
        ]
        
        # Get numerical features
        X_numerical = df[numerical_features].values
        
        # Combine embeddings
        embedding_fields = ['post_text', 'spoiler', 'target_paragraphs', 'target_keywords']
        X_embeddings = []
        
        for field in embedding_fields:
            X_embeddings.append(embeddings[field])
        
        # Concatenate all embeddings
        X_embeddings_combined = np.concatenate(X_embeddings, axis=1)
        
        # Combine numerical và embeddings
        X_combined = np.concatenate([X_numerical, X_embeddings_combined], axis=1)
        
        # Feature names for interpretability
        feature_names = numerical_features.copy()
        for field in embedding_fields:
            for i in range(384):  # SBERT embedding dimension
                feature_names.append(f"{field}_emb_{i}")
        
        self.feature_columns = feature_names
        
        logger.info(f"✅ Combined features shape: {X_combined.shape}")
        logger.info(f"   - Numerical features: {len(numerical_features)}")
        logger.info(f"   - Embedding features: {X_embeddings_combined.shape[1]}")
        logger.info(f"   - Total features: {X_combined.shape[1]}")
        
        return X_combined
    
    def prepare_labels(self, df):
        """Prepare labels"""
        # Map string labels to integers
        label_map = {'phrase': 0, 'passage': 1, 'multi': 2}
        y = df['spoiler_type'].map(label_map).values
        
        logger.info(f"✅ Label distribution:")
        unique, counts = np.unique(y, return_counts=True)
        for label_idx, count in zip(unique, counts):
            label_name = [k for k, v in label_map.items() if v == label_idx][0]
            logger.info(f"   - {label_name}: {count} ({count/len(y)*100:.1f}%)")
        
        return y, label_map
    
    def create_models(self):
        """Create ML models pipeline"""
        logger.info("🤖 Creating ML models...")
        
        models = {
            'random_forest': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(
                    n_estimators=self.config['models']['random_forest']['n_estimators'],
                    max_depth=self.config['models']['random_forest']['max_depth'],
                    random_state=42,
                    n_jobs=-1
                ))
            ]),
            
            'svm': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', SVC(
                    C=self.config['models']['svm']['C'],
                    kernel=self.config['models']['svm']['kernel'],
                    random_state=42,
                    probability=True
                ))
            ]),
            
            'logistic_regression': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(
                    C=self.config['models']['logistic_regression']['C'],
                    max_iter=self.config['models']['logistic_regression']['max_iter'],
                    random_state=42,
                    n_jobs=-1
                ))
            ])
        }
        
        return models
    
    def train_models(self, X_train, y_train):
        """Train all models với cross-validation"""
        logger.info("🚀 Training models với cross-validation...")
        
        models = self.create_models()
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        results = {}
        
        for model_name, model in models.items():
            logger.info(f"\n📊 Training {model_name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
            
            # Fit full model
            model.fit(X_train, y_train)
            
            # Store results
            results[model_name] = {
                'model': model,
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            logger.info(f"✅ {model_name}: CV Accuracy = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
            # Log to wandb
            if self.config.get('use_wandb', False):
                wandb.log({
                    f"{model_name}_cv_accuracy": cv_scores.mean(),
                    f"{model_name}_cv_std": cv_scores.std()
                })
        
        return results
    
    def evaluate_models(self, results, X_val, y_val, label_map):
        """Evaluate models trên validation set"""
        logger.info("\n📈 Evaluating models trên validation set...")
        
        best_model = None
        best_accuracy = 0
        best_model_name = ""
        
        evaluation_results = {}
        
        for model_name, result in results.items():
            model = result['model']
            
            # Predictions
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)
            
            # Metrics
            accuracy = accuracy_score(y_val, y_pred)
            
            logger.info(f"\n🎯 {model_name} Results:")
            logger.info(f"   Validation Accuracy: {accuracy:.4f}")
            
            # Detailed classification report
            class_names = list(label_map.keys())
            report = classification_report(y_val, y_pred, target_names=class_names, output_dict=True)
            
            logger.info("   Classification Report:")
            for class_name in class_names:
                metrics = report[class_name]
                logger.info(f"     {class_name}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
            
            # Store evaluation results
            evaluation_results[model_name] = {
                'accuracy': accuracy,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'classification_report': report,
                'confusion_matrix': confusion_matrix(y_val, y_pred)
            }
            
            # Track best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_model_name = model_name
            
            # Log to wandb
            if self.config.get('use_wandb', False):
                wandb.log({
                    f"{model_name}_val_accuracy": accuracy,
                    f"{model_name}_val_precision_macro": report['macro avg']['precision'],
                    f"{model_name}_val_recall_macro": report['macro avg']['recall'],
                    f"{model_name}_val_f1_macro": report['macro avg']['f1-score']
                })
        
        logger.info(f"\n🏆 Best Model: {best_model_name} (Accuracy: {best_accuracy:.4f})")
        
        return evaluation_results, best_model, best_model_name
    
    def analyze_feature_importance(self, best_model, best_model_name):
        """Analyze feature importance cho best model"""
        logger.info(f"\n🔍 Analyzing feature importance cho {best_model_name}...")
        
        try:
            if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
                # For Random Forest
                importances = best_model.named_steps['classifier'].feature_importances_
                
                # Get top 20 features
                top_indices = np.argsort(importances)[-20:][::-1]
                top_features = [self.feature_columns[i] for i in top_indices]
                top_importances = importances[top_indices]
                
                logger.info("   Top 20 Most Important Features:")
                for i, (feature, importance) in enumerate(zip(top_features, top_importances)):
                    logger.info(f"     {i+1:2d}. {feature}: {importance:.4f}")
                
                # Save feature importance plot
                plt.figure(figsize=(12, 8))
                plt.barh(range(len(top_features)), top_importances)
                plt.yticks(range(len(top_features)), top_features)
                plt.xlabel('Feature Importance')
                plt.title(f'Top 20 Feature Importances - {best_model_name}')
                plt.tight_layout()
                
                # Create results directory
                results_dir = Path("results/task2_classification")
                results_dir.mkdir(parents=True, exist_ok=True)
                
                plt.savefig(results_dir / f"feature_importance_{best_model_name}.png", dpi=300, bbox_inches='tight')
                plt.close()
                
                return top_features, top_importances
                
        except Exception as e:
            logger.warning(f"Could not analyze feature importance: {e}")
            return None, None
    
    def save_results(self, results, evaluation_results, best_model, best_model_name, label_map):
        """Save training results và models"""
        logger.info("💾 Saving results và models...")
        
        # Create output directories
        models_dir = Path("models/task2_classification")
        results_dir = Path("results/task2_classification")
        models_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save best model
        model_path = models_dir / f"best_model_{best_model_name}.pkl"
        joblib.dump(best_model, model_path)
        logger.info(f"✅ Saved best model: {model_path}")
        
        # Save all models
        for model_name, result in results.items():
            model_path = models_dir / f"model_{model_name}.pkl"
            joblib.dump(result['model'], model_path)
        
        # Save label mapping
        with open(models_dir / "label_mapping.json", 'w') as f:
            json.dump(label_map, f, indent=2)
        
        # Save detailed results
        summary_results = {
            'timestamp': datetime.now().isoformat(),
            'best_model': best_model_name,
            'best_accuracy': float(evaluation_results[best_model_name]['accuracy']),
            'config': self.config,
            'models_performance': {}
        }
        
        # Add performance cho từng model
        for model_name in results.keys():
            summary_results['models_performance'][model_name] = {
                'cv_accuracy_mean': float(results[model_name]['cv_mean']),
                'cv_accuracy_std': float(results[model_name]['cv_std']),
                'validation_accuracy': float(evaluation_results[model_name]['accuracy']),
                'classification_report': evaluation_results[model_name]['classification_report']
            }
        
        # Save summary
        with open(results_dir / "training_summary.json", 'w') as f:
            json.dump(summary_results, f, indent=2)
        
        logger.info(f"✅ Saved results summary: {results_dir / 'training_summary.json'}")
        
        return summary_results
    
    def create_confusion_matrices(self, evaluation_results, label_map):
        """Create confusion matrix visualizations"""
        logger.info("📊 Creating confusion matrix visualizations...")
        
        results_dir = Path("results/task2_classification")
        class_names = list(label_map.keys())
        
        fig, axes = plt.subplots(1, len(evaluation_results), figsize=(15, 4))
        if len(evaluation_results) == 1:
            axes = [axes]
        
        for idx, (model_name, results) in enumerate(evaluation_results.items()):
            cm = results['confusion_matrix']
            
            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Plot
            sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names,
                       ax=axes[idx])
            axes[idx].set_title(f'{model_name}\nAccuracy: {results["accuracy"]:.3f}')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(results_dir / "confusion_matrices.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("✅ Saved confusion matrices visualization")
    
    def run_training(self):
        """Main training pipeline"""
        logger.info("🚀 STARTING TASK 2 TRAINING: SBERT + ML CLASSIFICATION")
        logger.info("=" * 70)
        
        try:
            # Load data
            train_df, val_df, train_embeddings, val_embeddings = self.load_data()
            
            # Prepare features
            X_train = self.prepare_features(train_df, train_embeddings)
            X_val = self.prepare_features(val_df, val_embeddings)
            
            # Prepare labels
            y_train, label_map = self.prepare_labels(train_df)
            y_val, _ = self.prepare_labels(val_df)
            
            # Train models
            results = self.train_models(X_train, y_train)
            
            # Evaluate models
            evaluation_results, best_model, best_model_name = self.evaluate_models(
                results, X_val, y_val, label_map
            )
            
            # Feature importance analysis
            self.analyze_feature_importance(best_model, best_model_name)
            
            # Create visualizations
            self.create_confusion_matrices(evaluation_results, label_map)
            
            # Save results
            summary = self.save_results(results, evaluation_results, best_model, best_model_name, label_map)
            
            logger.info("\n" + "=" * 70)
            logger.info("🎉 TASK 2 TRAINING COMPLETED SUCCESSFULLY!")
            logger.info(f"🏆 Best Model: {best_model_name}")
            logger.info(f"🎯 Best Accuracy: {summary['best_accuracy']:.4f}")
            logger.info("=" * 70)
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise

def main():
    """Main function"""
    # Initialize classifier
    classifier = SpoilerClassifier()
    
    # Run training
    results = classifier.run_training()
    
    print(f"\n✅ Training completed successfully!")
    print(f"📊 Results saved in: results/task2_classification/")
    print(f"🤖 Models saved in: models/task2_classification/")

if __name__ == "__main__":
    main() 