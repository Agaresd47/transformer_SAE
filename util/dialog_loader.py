"""
Dialog data loader for DailyDialog dataset
Handles downloading, tokenization, and dataset preparation
"""

from daily_dialog import DailyDialog
from transformers import AutoTokenizer
from datasets import DatasetDict

class DialogLoader:
    def __init__(self, model_name='bert-base-uncased', max_length=128):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        print(f"Initialized DialogLoader with {model_name} tokenizer")

    def load_data(self):
        """
        Download and prepare the DailyDialog dataset
        """
        print("Loading DailyDialog dataset...")
        builder = DailyDialog()
        builder.download_and_prepare()
        dataset = builder.as_dataset()
        print("Dataset loaded successfully")
        return dataset

    def tokenize_function(self, example):
        """
        Tokenize dialog entries, handling both single strings and lists
        """
        if isinstance(example['dialog'], list):
            example['dialog'] = [' '.join(dialog) if isinstance(dialog, list) else dialog 
                               for dialog in example['dialog']]
        
        return self.tokenizer(
            example['dialog'],
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        )

    def prepare_dataset(self):
        """
        Load, tokenize, and format the dataset for PyTorch
        """
        # Load raw dataset
        dataset = self.load_data()
        
        # Tokenize the dataset
        print("Tokenizing dataset...")
        tokenized_datasets = dataset.map(
            self.tokenize_function,
            batched=True
        )
        
        # Format for PyTorch
        print("Formatting dataset for PyTorch...")
        tokenized_datasets.set_format(
            type='torch',
            columns=['input_ids', 'attention_mask', 'act', 'emotion']
        )
        
        print("Dataset preparation complete")
        return tokenized_datasets

def get_dialog_data(model_name='bert-base-uncased', max_length=128):
    """
    Convenience function to get the prepared dataset
    """
    loader = DialogLoader(model_name, max_length)
    return loader.prepare_dataset()

if __name__ == "__main__":
    # Example usage
    dataset = get_dialog_data()
    print("\nDataset splits available:", dataset.keys())
    print("Features available:", dataset['train'].features)
    print("Number of training examples:", len(dataset['train'])) 