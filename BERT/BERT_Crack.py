import pandas as pd
import numpy as np
import json
import torch
import logging
from typing import Dict, List, Tuple
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bert_model.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Data processing class specialized for BERT"""
    
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
        
        # Extract context and labels
        self.df['context'] = self.df['data'].apply(lambda x: x.strip('"'))
        self.df['meaning_label'] = self.df['meaning'].apply(lambda x: x.split(' [')[0])
        
        # Filter out low-frequency classes
        class_counts = self.df['meaning_label'].value_counts()
        valid_classes = class_counts[class_counts >= 5].index
        self.df = self.df[self.df['meaning_label'].isin(valid_classes)].copy()
        
        # Create label mapping
        unique_labels = sorted(self.df['meaning_label'].unique())
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
        self.inv_label_map = {idx: label for label, idx in self.label_map.items()}
        self.df['label_id'] = self.df['meaning_label'].map(self.label_map)
        
        return self.df
    
    def get_train_test_split(self, test_size: float = 0.2, random_state: int = 42):
        """Get train-test split"""
        if self.df is None:
            self.load_and_preprocess()
            
        X_train, X_test, y_train, y_test = train_test_split(
            self.df['context'].tolist(),
            self.df['label_id'].tolist(),
            test_size=test_size,
            random_state=random_state
        )
        
        return X_train, X_test, y_train, y_test

class BERTModel:
    """Complete BERT model implementation"""
    
    def __init__(self, num_labels: int, model_name: str = 'bert-base-uncased'):
        self.num_labels = num_labels
        self.model_name = model_name
        self.label_map = None
        self.inv_label_map = None
        
        logger.info("Initializing BERT model...")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        logger.info(f"Model loaded on {self.device}")

    class CrackWSDDataset(Dataset):
        """Custom dataset class for word sense disambiguation"""
        
        def __init__(self, contexts: List[str], labels: List[int], tokenizer: BertTokenizer, max_length: int = 128):
            self.contexts = contexts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length
            
        def __len__(self) -> int:
            return len(self.contexts)
        
        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            context = self.contexts[idx]
            
            encoding = self.tokenizer(
                context,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'label': torch.tensor(self.labels[idx], dtype=torch.long)
            }
    
    def train(self, train_texts: List[str], train_labels: List[int], 
              val_texts: List[str], val_labels: List[int],
              batch_size: int = 8, num_epochs: int = 5, learning_rate: float = 3e-5):
        """Complete training pipeline"""
        logger.info("Starting BERT training...") 
        
        # Create datasets
        train_dataset = self.CrackWSDDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = self.CrackWSDDataset(val_texts, val_labels, self.tokenizer)
        
        # Class-balanced sampling
        class_counts = np.bincount(train_labels)
        class_weights = (1. / class_counts) * (len(class_counts) / sum(1. / class_counts))
        sampler = WeightedRandomSampler(
            weights=class_weights[train_labels],
            num_samples=len(train_labels),
            replacement=True
        )
        
        # Create dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Optimizer setup
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            
            for batch in train_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
            
            # Validation evaluation
            avg_loss = total_loss / len(train_dataloader)
            val_preds, val_labels = self.evaluate(val_dataloader)
            report = classification_report(val_labels, val_preds)
            
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            logger.info(f"Train Loss: {avg_loss:.4f}")
            logger.info("Validation Report:\n" + report)
    
    def evaluate(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Complete evaluation pipeline"""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        return np.array(all_preds), np.array(all_labels)
    
    def predict(self, text: str) -> Dict:
        """Prediction interface"""
        encoding = self.tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            confidence, predicted_class = torch.max(probabilities, dim=1)
        
        return {
            'predicted_sense': self.inv_label_map[predicted_class.item()],
            'confidence': confidence.item(),
            'probabilities': {self.inv_label_map[i]: prob.item() for i, prob in enumerate(probabilities.squeeze())}
        }
    
    def save_model(self, output_dir: str):
        """Model saving"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Model saved to {output_dir}")

class WSDSystem:
    """Pure BERT-based Word Sense Disambiguation system"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data_processor = DataProcessor(data_path)
        self.bert_model = None
        
    def run_experiment(self):
        """Complete experiment workflow"""
        # Data preparation
        self.data_processor.load_and_preprocess()
        X_train, X_test, y_train, y_test = self.data_processor.get_train_test_split()
        
        # Model initialization
        self.bert_model = BERTModel(len(self.data_processor.label_map))
        self.bert_model.label_map = self.data_processor.label_map
        self.bert_model.inv_label_map = self.data_processor.inv_label_map
        
        # Training and validation
        self.bert_model.train(X_train, y_train, X_test, y_test)
        
        # Final evaluation
        test_dataset = self.bert_model.CrackWSDDataset(X_test, y_test, self.bert_model.tokenizer)
        test_dataloader = DataLoader(test_dataset, batch_size=4)
        final_preds, final_labels = self.bert_model.evaluate(test_dataloader)
        
        # Generate report
        final_report = classification_report(
            [self.data_processor.inv_label_map[l] for l in final_labels],
            [self.data_processor.inv_label_map[p] for p in final_preds]
        )
        logger.info("Final Test Report:\n" + final_report)
        
        # Save results
        results = {
            'timestamp': datetime.now().isoformat(),
            'label_map': self.data_processor.label_map,
            'test_report': final_report,
            'model_info': {
                'model_name': 'bert-base-uncased',
                'num_labels': len(self.data_processor.label_map)
            }
        }
        
        with open('bert_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("Experiment completed. Results saved to bert_results.json")

    def predict(self, text: str) -> Dict:
        """Prediction interface"""
        return self.bert_model.predict(text)

if __name__ == "__main__":
    # Initialize system
    wsd_system = WSDSystem('dataset.json')
    
    # Run complete experiment
    wsd_system.run_experiment()
    
    # Example predictions
    examples = [
        "The phone screen cracked after falling on the floor.",  # Phone screen cracked
        "The ice on the lake is starting to crack.",           # Ice cracking
        "He finally cracked under the pressure of exams.",     # Mental breakdown
    ]

    for example in examples:
        result = wsd_system.predict(example)
        print(f"\nInput: {example}")
        print(f"Predicted Sense: {result['predicted_sense']}")
        print(f"Confidence: {result['confidence']:.2f}")
