import pandas as pd
import numpy as np
import json
import re
import logging
from typing import Dict, List, Tuple
from pathlib import Path
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('wsd_model.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Data processing class responsible for loading and preprocessing data"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = None
        self.label_map = None
        self.inv_label_map = None
        
    def load_and_preprocess(self) -> pd.DataFrame:
        """Load and preprocess data"""
        logger.info(f"Loading data from {self.data_path}")
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.df = pd.DataFrame(data)
        
        # Extract context and meaning labels
        self.df['context'] = self.df['data'].apply(lambda x: x.strip('"'))
        self.df['meaning_label'] = self.df['meaning'].apply(lambda x: x.split(' [')[0])
        
        # Filter out classes with too few samples first
        class_counts = self.df['meaning_label'].value_counts()
        valid_classes = class_counts[class_counts >= 5].index
        self.df = self.df[self.df['meaning_label'].isin(valid_classes)].copy()
        
        # Create consistent label mapping after filtering
        unique_labels = sorted(self.df['meaning_label'].unique())
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
        self.inv_label_map = {idx: label for label, idx in self.label_map.items()}
        self.df['label_id'] = self.df['meaning_label'].map(self.label_map)
        
        # Verify all labels are properly mapped
        missing_labels = [label for label in self.df['meaning_label'] if label not in self.label_map]
        if missing_labels:
            logger.error(f"Missing labels in mapping: {missing_labels}")
            raise ValueError("Label mapping is incomplete after filtering")
            
        return self.df
    
    @staticmethod
    def preprocess_text(text: str) -> str:
        """Text preprocessing"""
        text = re.sub(r'[^\w\s]', ' ', text)
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def get_train_test_split(self, test_size: float = 0.2, random_state: int = 42):
        """Get train-test split"""
        if self.df is None:
            self.load_and_preprocess()
            
        # Preprocess text
        self.df['processed_context'] = self.df['context'].apply(self.preprocess_text)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.df['processed_context'], 
            self.df['meaning_label'], 
            test_size=test_size, 
            random_state=random_state
        )
        
        return (X_train, X_test, y_train, y_test)

class SVMModel:
    """SVM model class"""
    
    def __init__(self):
        self.model = None
        self.best_params = None
        
    def build_pipeline(self) -> Pipeline:
        """Build model pipeline"""
        return Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 3), max_features=5000)),
            ('svm', SVC(kernel='linear', probability=True, class_weight='balanced'))
        ])
    
    def train(self, X_train, y_train) -> Pipeline:
        """Train SVM model"""
        logger.info("Training SVM model...")
        
        pipeline = self.build_pipeline()
        
        param_grid = {
            'tfidf__max_df': [0.5, 0.75, 1.0],
            'tfidf__min_df': [1, 2, 3],
            'svm__C': [0.1, 1, 10]
        }
        
        grid_search = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=3, 
            n_jobs=-1, 
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        logger.info(f"SVM best parameters: {self.best_params}")
        return self.model
    
    def evaluate(self, X_test, y_test) -> Dict:
        """Evaluate model performance"""
        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        # Add custom handling for classes with no predictions
        for label in np.unique(y_test):
            if label not in y_pred:
                logger.warning(f"Class {label} had no predicted samples")
        
        logger.info("SVM Classification Report:")
        logger.info(classification_report(y_test, y_pred, zero_division=0))
        
        return report
    
    def cross_validate(self, X, y, cv: int = 5) -> np.ndarray:
        """Cross-validation"""
        logger.info("Performing cross-validation for SVM...")
        # Convert string labels to numeric for bincount
        unique_labels, y_numeric = np.unique(y, return_inverse=True)
        min_class_size = min(np.bincount(y_numeric))
        n_splits = min(5, min_class_size)
        
        if n_splits < 2:
            logger.warning("Not enough samples per class for cross-validation")
            return np.array([0.0])  # Return dummy score
            
        scores = cross_val_score(
            self.model, 
            X, 
            y, 
            cv=StratifiedKFold(n_splits=n_splits),
            scoring='accuracy'
        )
        
        logger.info(f"SVM CV scores: {scores}")
        logger.info(f"Mean accuracy: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")
        return scores
    
    def predict(self, text: str) -> Dict:
        """Predict new sample"""
        processed = DataProcessor.preprocess_text(text)
        prediction = self.model.predict([processed])[0]
        probabilities = self.model.predict_proba([processed])[0]
        best_prob = max(probabilities)
        
        return {
            'predicted_sense': prediction,
            'confidence': best_prob,
            'probabilities': {cls: prob for cls, prob in zip(self.model.classes_, probabilities)}
        }
    
    def get_feature_importance(self, class_idx: int = 0) -> List[Tuple[str, float]]:
        """Get feature importance"""
        tfidf = self.model.named_steps['tfidf']
        svm = self.model.named_steps['svm']
        feature_names = tfidf.get_feature_names_out()
        coef = svm.coef_[class_idx].toarray().flatten()
        
        indices = np.argsort(coef)[-10:]  # Get top 10 most important features
        return [(feature_names[i], float(coef[i])) for i in indices]
    
    def plot_feature_importance(self, save_path: str = None):
        """Visualize feature importance"""
        plt.rcParams['font.sans-serif'] = ['SimHei'] 
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(10,6))
        for i, sense in enumerate(self.model.classes_):
            features = self.get_feature_importance(i)
            feature_names = [str(f[0]) for f in features]
            coef_values = [float(f[1]) for f in features]
            plt.barh(feature_names, coef_values, label=str(sense))
        
        plt.xlabel("Feature importance coefficient")
        plt.title("SVM Model Feature Importance Analysis")
        plt.legend(fontsize='small', title='Senses', title_fontsize='small') 
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved feature importance plot to {save_path}")
        plt.close()

class WSDSystem:
    """Word Sense Disambiguation system main class"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data_processor = DataProcessor(data_path)
        self.svm_model = SVMModel()
        self.results = {}
        
    def run_experiment(self):
        """Run complete experiment workflow"""
        # 1. Data loading and preprocessing
        df = self.data_processor.load_and_preprocess()
        
        # 2. Data splitting
        X_train, X_test, y_train, y_test = self.data_processor.get_train_test_split()
        
        # 3. SVM model training and evaluation
        self.svm_model.train(X_train, y_train)
        svm_report = self.svm_model.evaluate(X_test, y_test)
        svm_scores = self.svm_model.cross_validate(
            self.data_processor.df['processed_context'],
            self.data_processor.df['meaning_label']
        )
        
        # Visualize feature importance
        self.svm_model.plot_feature_importance('svm_feature_importance.png')
        
        # Save results
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'svm': {
                'best_params': self.svm_model.best_params,
                'classification_report': svm_report,
                'cv_scores': svm_scores.tolist()
            }
        }
        
        with open('results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info("Experiment completed. Results saved to results.json")

    def predict(self, text: str) -> Dict:
        """Predict sense for new sample"""
        return self.svm_model.predict(text)

if __name__ == "__main__":
    # Initialize system
    wsd_system = WSDSystem('dataset.json')
    # Run experiment
    wsd_system.run_experiment()

    # Example predictions
    examples = [
        "The glass cracked when I dropped it.",
        "He cracked the code after hours of work.",
        "Her voice cracked with emotion."
    ]

    for example in examples:
        print(f"\nPredicting sense for: '{example}'")
        result = wsd_system.predict(example)
        print(f"Predicted sense: {result['predicted_sense']} (confidence: {result['confidence']:.2f})")