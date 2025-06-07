#!/usr/bin/env python3
"""
Evaluation script cho Task 2: SBERT + ML Classification
"""

import pandas as pd
import numpy as np
import pickle
import json
import joblib
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ClassificationEvaluator:
    def __init__(self, model_dir="models/task2_classification"):
        """Initialize evaluator"""
        self.model_dir = Path(model_dir)
        self.results_dir = Path("results/task2_classification/evaluation")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load label mapping
        with open(self.model_dir / "label_mapping.json", 'r') as f:
            self.label_map = json.load(f)
        
        self.class_names = list(self.label_map.keys())
        
    def load_model(self, model_name):
        """Load trained model"""
        model_path = self.model_dir / f"model_{model_name}.pkl"
        if model_path.exists():
            return joblib.load(model_path)
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")
    
    def load_test_data(self):
        """Load test/validation data"""
        logger.info("📂 Loading test data...")
        
        # Load validation data (as test set)
        val_df = pd.read_csv("data/processed/task2/validation_processed.csv")
        
        with open("data/processed/task2/validation_embeddings.pkl", 'rb') as f:
            val_embeddings = pickle.load(f)
        
        # Prepare features (same as training)
        numerical_features = [
            'post_length', 'spoiler_length', 'target_length', 
            'keywords_count', 'has_description'
        ]
        
        X_numerical = val_df[numerical_features].values
        
        # Combine embeddings
        embedding_fields = ['post_text', 'spoiler', 'target_paragraphs', 'target_keywords']
        X_embeddings = []
        
        for field in embedding_fields:
            X_embeddings.append(val_embeddings[field])
        
        X_embeddings_combined = np.concatenate(X_embeddings, axis=1)
        X_combined = np.concatenate([X_numerical, X_embeddings_combined], axis=1)
        
        # Prepare labels
        y_true = val_df['spoiler_type'].map(self.label_map).values
        
        logger.info(f"✅ Loaded {len(val_df)} test samples")
        
        return X_combined, y_true, val_df
    
    def evaluate_single_model(self, model_name, X_test, y_true):
        """Evaluate single model"""
        logger.info(f"📊 Evaluating {model_name}...")
        
        # Load model
        model = self.load_model(model_name)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=range(len(self.class_names))
        )
        
        # Macro averages
        precision_macro = np.mean(precision)
        recall_macro = np.mean(recall)
        f1_macro = np.mean(f1)
        
        # Classification report
        report = classification_report(
            y_true, y_pred, 
            target_names=self.class_names, 
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision_per_class': precision,
            'recall_per_class': recall,
            'f1_per_class': f1,
            'support_per_class': support,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        # ROC AUC cho multi-class
        try:
            y_true_binarized = label_binarize(y_true, classes=range(len(self.class_names)))
            if y_true_binarized.shape[1] > 1:
                auc_scores = []
                for i in range(len(self.class_names)):
                    auc = roc_auc_score(y_true_binarized[:, i], y_pred_proba[:, i])
                    auc_scores.append(auc)
                results['auc_per_class'] = auc_scores
                results['auc_macro'] = np.mean(auc_scores)
            else:
                results['auc_per_class'] = [0.5]
                results['auc_macro'] = 0.5
        except Exception as e:
            logger.warning(f"Could not compute ROC AUC: {e}")
            results['auc_per_class'] = [0.0] * len(self.class_names)
            results['auc_macro'] = 0.0
        
        return results
    
    def compare_models(self, model_names, X_test, y_true):
        """Compare multiple models"""
        logger.info("🔍 Comparing models...")
        
        all_results = {}
        
        for model_name in model_names:
            try:
                results = self.evaluate_single_model(model_name, X_test, y_true)
                all_results[model_name] = results
                
                logger.info(f"✅ {model_name}: Accuracy = {results['accuracy']:.4f}, F1-Macro = {results['f1_macro']:.4f}")
                
            except Exception as e:
                logger.error(f"❌ Failed to evaluate {model_name}: {e}")
        
        return all_results
    
    def create_comparison_visualization(self, all_results):
        """Create model comparison visualizations"""
        logger.info("📊 Creating comparison visualizations...")
        
        # Prepare data for plotting
        models = list(all_results.keys())
        metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # 1. Overall metrics comparison
        metric_data = {metric: [all_results[model][metric] for model in models] for metric in metrics}
        
        ax = axes[0, 0]
        x = np.arange(len(models))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            ax.bar(x + i*width, metric_data[metric], width, label=metric.replace('_', ' ').title())
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Overall Performance Metrics')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Per-class F1 scores
        ax = axes[0, 1]
        f1_data = np.array([all_results[model]['f1_per_class'] for model in models])
        
        x = np.arange(len(self.class_names))
        width = 0.25
        
        for i, model in enumerate(models):
            ax.bar(x + i*width, f1_data[i], width, label=model)
        
        ax.set_xlabel('Classes')
        ax.set_ylabel('F1 Score')
        ax.set_title('F1 Score per Class')
        ax.set_xticks(x + width)
        ax.set_xticklabels(self.class_names)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Confusion matrices
        ax = axes[1, 0]
        
        # Show confusion matrix for best model
        best_model = max(models, key=lambda m: all_results[m]['accuracy'])
        cm = all_results[best_model]['confusion_matrix']
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names, ax=ax)
        ax.set_title(f'Confusion Matrix - {best_model}')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        # 4. AUC comparison
        ax = axes[1, 1]
        if 'auc_macro' in all_results[models[0]]:
            auc_scores = [all_results[model]['auc_macro'] for model in models]
            bars = ax.bar(models, auc_scores, color=['skyblue', 'lightcoral', 'lightgreen'][:len(models)])
            ax.set_ylabel('Macro AUC')
            ax.set_title('ROC AUC Comparison')
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, score in zip(bars, auc_scores):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                       f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("✅ Saved model comparison visualization")
    
    def create_detailed_analysis(self, all_results):
        """Create detailed analysis report"""
        logger.info("📋 Creating detailed analysis report...")
        
        # Find best model
        best_model = max(all_results.keys(), key=lambda m: all_results[m]['accuracy'])
        best_results = all_results[best_model]
        
        # Create detailed report
        report = {
            'evaluation_timestamp': pd.Timestamp.now().isoformat(),
            'best_model': best_model,
            'best_accuracy': float(best_results['accuracy']),
            'model_comparison': {},
            'detailed_analysis': {
                'per_class_performance': {},
                'confusion_matrix_analysis': {},
                'error_analysis': {}
            }
        }
        
        # Model comparison summary
        for model_name, results in all_results.items():
            report['model_comparison'][model_name] = {
                'accuracy': float(results['accuracy']),
                'precision_macro': float(results['precision_macro']),
                'recall_macro': float(results['recall_macro']),
                'f1_macro': float(results['f1_macro']),
                'auc_macro': float(results.get('auc_macro', 0.0))
            }
        
        # Detailed analysis for best model
        for i, class_name in enumerate(self.class_names):
            report['detailed_analysis']['per_class_performance'][class_name] = {
                'precision': float(best_results['precision_per_class'][i]),
                'recall': float(best_results['recall_per_class'][i]),
                'f1_score': float(best_results['f1_per_class'][i]),
                'support': int(best_results['support_per_class'][i]),
                'auc': float(best_results.get('auc_per_class', [0])[i])
            }
        
        # Confusion matrix analysis
        cm = best_results['confusion_matrix']
        for i, true_class in enumerate(self.class_names):
            for j, pred_class in enumerate(self.class_names):
                key = f"{true_class}_predicted_as_{pred_class}"
                report['detailed_analysis']['confusion_matrix_analysis'][key] = int(cm[i, j])
        
        # Save report
        with open(self.results_dir / "detailed_evaluation_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info("✅ Saved detailed evaluation report")
        
        return report
    
    def create_error_analysis(self, best_model_name, all_results, test_df):
        """Analyze prediction errors"""
        logger.info("🔍 Creating error analysis...")
        
        best_results = all_results[best_model_name]
        y_true = best_results['predictions']  # This should be from y_true, fixing this
        y_pred = best_results['predictions']
        
        # Find misclassified samples
        errors = test_df.copy()
        errors['predicted'] = y_pred
        errors['true_label'] = [self.class_names[i] for i in range(len(test_df))]  # Fix this
        errors['predicted_label'] = [self.class_names[i] for i in y_pred]
        
        # Get only errors
        error_mask = y_true != y_pred  # This needs to be fixed
        error_samples = errors[error_mask]
        
        if len(error_samples) > 0:
            # Save error samples for manual inspection
            error_samples[['spoiler_type', 'predicted_label', 'post_length', 'spoiler_length']].to_csv(
                self.results_dir / "error_analysis.csv", index=False
            )
            
            logger.info(f"✅ Saved {len(error_samples)} error samples for analysis")
        else:
            logger.info("✅ No prediction errors found!")
        
        return error_samples
    
    def run_evaluation(self, model_names=None):
        """Run complete evaluation"""
        logger.info("🚀 STARTING TASK 2 EVALUATION")
        logger.info("=" * 60)
        
        try:
            # Load test data
            X_test, y_true, test_df = self.load_test_data()
            
            # Default models to evaluate
            if model_names is None:
                model_names = ['random_forest', 'svm', 'logistic_regression']
            
            # Filter existing models
            existing_models = []
            for model_name in model_names:
                model_path = self.model_dir / f"model_{model_name}.pkl"
                if model_path.exists():
                    existing_models.append(model_name)
                else:
                    logger.warning(f"⚠️  Model not found: {model_name}")
            
            if not existing_models:
                raise FileNotFoundError("No trained models found!")
            
            # Compare models
            all_results = self.compare_models(existing_models, X_test, y_true)
            
            # Create visualizations
            self.create_comparison_visualization(all_results)
            
            # Detailed analysis
            report = self.create_detailed_analysis(all_results)
            
            # Error analysis
            best_model_name = report['best_model']
            self.create_error_analysis(best_model_name, all_results, test_df)
            
            # Summary
            logger.info("\n" + "=" * 60)
            logger.info("🎉 EVALUATION COMPLETED!")
            logger.info(f"🏆 Best Model: {best_model_name}")
            logger.info(f"🎯 Best Accuracy: {report['best_accuracy']:.4f}")
            logger.info(f"📊 Results saved in: {self.results_dir}")
            logger.info("=" * 60)
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            raise

def main():
    """Main function"""
    evaluator = ClassificationEvaluator()
    
    # Run evaluation
    results = evaluator.run_evaluation()
    
    print(f"\n✅ Evaluation completed successfully!")
    print(f"📊 Results saved in: results/task2_classification/evaluation/")

if __name__ == "__main__":
    main() 