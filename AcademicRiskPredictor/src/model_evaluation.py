import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import shap
import json
import joblib
import os

class StuRiskModelEvaluator:
    def __init__(self, models, plot_dir='plots'):
        self.models = models
        self.plot_dir = plot_dir
        os.makedirs(self.plot_dir, exist_ok=True)
        self.metrics_comparison = {
            'Model': [],
            'Accuracy': [],
            'F1 Score': [],
            'Precision': [],
            'Recall': [],
            'ROC AUC': []
        }

    def evaluate_models(self, X_train, X_test, y_train, y_test):
        """Evaluate models and generate plots."""
        # Binarize the output for multi-class ROC AUC
        n_classes = len(np.unique(y_test))
        y_test_bin = label_binarize(y_test, classes=np.unique(y_test)) if n_classes > 2 else y_test

        self.metrics_comparison = {'Model': [], 'Accuracy': [], 'F1 Score': [], 'Precision': [], 'Recall': [], 'ROC AUC': []}

        # Confusion Matrix figures
        plt.figure(figsize=(20, 15))
        for idx, (model_name, model) in enumerate(self.models.items(), 1):
            # Train the model
            model.fit(X_train, y_train)

            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            precision = precision_score(y_test, y_pred, average='macro')
            recall = recall_score(y_test, y_pred, average='macro')
            roc_auc = self._calculate_roc_auc(y_test, y_test_bin, y_pred_proba, n_classes)

            # Store metrics
            self._store_metrics(model_name, accuracy, f1, precision, recall, roc_auc)

            # Plot confusion matrix
            plt.subplot(2, 2, idx)
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'{model_name} Confusion Matrix')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')

        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Display Metrics Table
        self._display_metrics_table()

        # Feature Importance and SHAP
        self._plot_feature_importance(self.models, X_test)

        # Metrics Comparison Bar Chart
        self._plot_metrics_comparison()

        # ROC Curves
        self._plot_roc_curves(self.models, X_test, y_test, y_test_bin, n_classes)

        return pd.DataFrame(self.metrics_comparison)

    def _calculate_roc_auc(self, y_test, y_test_bin, y_pred_proba, n_classes):
        if y_pred_proba is not None and n_classes > 2:
            return roc_auc_score(y_test_bin, y_pred_proba, multi_class='ovr')
        elif y_pred_proba is not None:
            return roc_auc_score(y_test, y_pred_proba[:, 1])
        else:
            return None

    def _store_metrics(self, model_name, accuracy, f1, precision, recall, roc_auc):
        self.metrics_comparison['Model'].append(model_name)
        self.metrics_comparison['Accuracy'].append(accuracy)
        self.metrics_comparison['F1 Score'].append(f1)
        self.metrics_comparison['Precision'].append(precision)
        self.metrics_comparison['Recall'].append(recall)
        self.metrics_comparison['ROC AUC'].append(roc_auc)

    def _display_metrics_table(self):
        metrics_df = pd.DataFrame(self.metrics_comparison)
        print("Model Performance Metrics:")
        print(metrics_df)
        metrics_df.to_csv(os.path.join(self.plot_dir, 'model_metrics.csv'), index=False)



    def _plot_feature_importance(self, models, X_test):
        plt.figure(figsize=(20, 15))
        for idx, (model_name, model) in enumerate(models.items(), 1):
            if model_name in ['Random Forest']:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test)
                plt.subplot(2, 2, idx)
                shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
                plt.title(f'{model_name} SHAP Feature Importance')

        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_metrics_comparison(self):
        metrics_df = pd.DataFrame(self.metrics_comparison)
        metrics_to_plot = ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'ROC AUC']

        plt.figure(figsize=(12, 6))
        for i, metric in enumerate(metrics_to_plot, 1):
            plt.subplot(1, 5, i)
            plt.bar(metrics_df['Model'], metrics_df[metric], color='skyblue')
            plt.title(metric)
            plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_roc_curves(self, models, X_test, y_test, y_test_bin, n_classes):
        plt.figure(figsize=(12, 8))

        for model_name, model in models.items():
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)

                if n_classes > 2:
                    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_pred_proba.ravel())
                    roc_auc = roc_auc_score(y_test_bin, y_pred_proba, multi_class='ovr')
                else:
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
                    roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])

                plt.plot(fpr, tpr, label=f'{model_name} (AUC: {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves for All Models', fontsize=15)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def predict(self, model_path, input_row):
        """Make prediction on a single input row (with database encoding)."""
        if isinstance(input_row, dict):
            input_row = pd.DataFrame([input_row])
        elif not isinstance(input_row, pd.DataFrame):
            raise ValueError("Input row must be a dictionary or a pandas DataFrame.")

        model = joblib.load(model_path)
        predicted_encoded = model.predict(input_row)[0]

        # Reverse mapping for CGPA classes (standard for this dataset)
        cgpa_classes = ['Fail', 'Pass', 'Third Class', 'Second Class Lower',
                       'Second Class Upper', 'First Class']

        # Map numbers back to human readable (expected as integers now)
        if isinstance(predicted_encoded, int) and 0 <= predicted_encoded < len(cgpa_classes):
            predicted_human_readable = cgpa_classes[predicted_encoded]
        else:
            predicted_human_readable = f"CGPA_{predicted_encoded}"

        return predicted_human_readable

    def save_models(self, output_dir):
        """Save trained models to files."""
        os.makedirs(output_dir, exist_ok=True)
        for model_name, model in self.models.items():
            filename = f"{model_name.replace(' ', '_')}.pkl"
            joblib.dump(model, os.path.join(output_dir, filename))
        print(f"Models saved to {output_dir}")

    def load_model(self, model_path):
        """Load a single model from file."""
        return joblib.load(model_path)
