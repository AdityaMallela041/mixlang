"""
MixLang+ - Train on ALL LinCE Language Pairs
Language pairs: Hindi-English, Spanish-English, Nepali-English, MSA-Egyptian Arabic
Tasks: Language Identification (LID)
"""
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
import torch
import os
import json
from datetime import datetime

def compute_metrics(pred):
    """Compute comprehensive evaluation metrics"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    # Calculate all metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0
    )
    acc = accuracy_score(labels, preds)
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0
    )
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'f1_macro': f1_score(labels, preds, average='macro', zero_division=0),
        'precision_macro': precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)[0],
        'recall_macro': precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)[1],
    }

def load_lince_language_pair(base_path, language_pair, task='lid'):
    """
    Load specific language pair from LinCE
    
    Args:
        base_path: Path to LinCE directory
        language_pair: 'hineng', 'spaeng', 'nepeng', 'msaea'
        task: 'lid', 'ner', 'pos', 'sa'
    """
    print(f"\n📂 Loading {language_pair.upper()} - {task.upper()}...")
    
    try:
        train_file = f'{base_path}/{task}_{language_pair}_train.csv'
        val_file = f'{base_path}/{task}_{language_pair}_validation.csv'
        test_file = f'{base_path}/{task}_{language_pair}_test.csv'
        
        train_df = pd.read_csv(train_file)
        val_df = pd.read_csv(val_file)
        test_df = pd.read_csv(test_file)
        
        print(f"   ✅ Train: {len(train_df):,} samples")
        print(f"   ✅ Val: {len(val_df):,} samples")
        print(f"   ✅ Test: {len(test_df):,} samples")
        
        return train_df, val_df, test_df
    except FileNotFoundError as e:
        print(f"   ⚠️ Files not found for {language_pair}-{task}")
        return None, None, None

def prepare_lince_for_classification(df):
    """
    Convert LinCE DataFrame to binary classification format
    0 = Monolingual (predominantly one language)
    1 = Code-switched (mixed languages)
    """
    texts = []
    labels = []
    
    for idx, row in df.iterrows():
        # Get text from 'words' column
        if 'words' in df.columns:
            text = str(row['words'])
        else:
            # Concatenate all text columns
            text = ' '.join([str(val) for val in row.values if isinstance(val, str)])
        
        texts.append(text)
        
        # Binary label: detect if text has code-switching
        # Simple heuristic: check if language tags indicate multiple languages
        if 'lid' in df.columns or 'lang' in df.columns:
            # If we have language tags, check for mixing
            lid_col = 'lid' if 'lid' in df.columns else 'lang'
            lang_tags = str(row[lid_col])
            # Code-switched if multiple language tags present
            unique_langs = len(set(lang_tags.split()))
            labels.append(1 if unique_langs > 1 else 0)
        else:
            # Default to code-switched (conservative)
            labels.append(1)
    
    return Dataset.from_dict({
        'text': texts,
        'labels': labels
    })

def train_on_language_pair(language_pair, base_path, model_name="xlm-roberta-base"):
    """Train model on a specific language pair"""
    
    print("\n" + "="*70)
    print(f"🚀 Training on {language_pair.upper()}")
    print("="*70)
    
    # Load data
    train_df, val_df, test_df = load_lince_language_pair(base_path, language_pair, 'lid')
    
    if train_df is None:
        print(f"⚠️ Skipping {language_pair} - data not found")
        return None
    
    # Prepare datasets
    print("\n🔄 Preparing datasets...")
    train_dataset = prepare_lince_for_classification(train_df)
    val_dataset = prepare_lince_for_classification(val_df)
    test_dataset = prepare_lince_for_classification(test_df)
    
    # Load tokenizer and model
    print(f"\n🤖 Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2
    )
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples['text'], 
            padding='max_length', 
            truncation=True, 
            max_length=128
        )
    
    print("🔄 Tokenizing...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    # Training arguments
    output_dir = f'./models/checkpoints/{language_pair}'
    os.makedirs(output_dir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir=f'./logs/{language_pair}',
        logging_steps=50,
        save_strategy='epoch',
        load_best_model_at_end=True,
        fp16=True,
        dataloader_num_workers=2,
        report_to="none",
        metric_for_best_model='f1',
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train
    print("\n🏋️ Training...")
    print(f"   Samples: {len(train_dataset):,}")
    print(f"   Epochs: 3")
    print(f"   Batch size: 16")
    
    trainer.train()
    
    # Evaluate on test set
    print("\n📊 Evaluating on test set...")
    results = trainer.evaluate(test_dataset)
    
    # Get detailed predictions
    predictions = trainer.predict(test_dataset)
    preds = predictions.predictions.argmax(-1)
    labels = predictions.label_ids
    
    # Detailed metrics
    print("\n" + "="*70)
    print(f"✅ RESULTS - {language_pair.upper()}")
    print("="*70)
    print(f"Accuracy:  {results['eval_accuracy']:.4f}")
    print(f"F1 Score:  {results['eval_f1']:.4f}")
    print(f"Precision: {results['eval_precision']:.4f}")
    print(f"Recall:    {results['eval_recall']:.4f}")
    
    print("\n📊 Classification Report:")
    print(classification_report(
        labels, preds, 
        target_names=['Monolingual', 'Code-Switched'],
        digits=4
    ))
    
    # Save model
    print(f"\n💾 Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Return results for summary
    return {
        'language_pair': language_pair,
        'train_samples': len(train_dataset),
        'test_samples': len(test_dataset),
        'accuracy': results['eval_accuracy'],
        'f1': results['eval_f1'],
        'precision': results['eval_precision'],
        'recall': results['eval_recall'],
        'detailed_results': results
    }

def main():
    """Main training function for all language pairs"""
    
    print("="*70)
    print("🚀 MixLang+ - Training on ALL LinCE Language Pairs")
    print("="*70)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"\n✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("\n⚠️ No GPU detected - training will be slower")
    
    # Language pairs to train
    language_pairs = [
        'hineng',   # Hindi-English
        'spaeng',   # Spanish-English
        'nepeng',   # Nepali-English
        'msaea',    # MSA-Egyptian Arabic
    ]
    
    base_path = './data/raw/LinCE'
    all_results = []
    
    start_time = datetime.now()
    
    # Train on each language pair
    for lang_pair in language_pairs:
        result = train_on_language_pair(lang_pair, base_path)
        if result:
            all_results.append(result)
    
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds() / 60
    
    # Summary
    print("\n" + "="*70)
    print("📊 FINAL SUMMARY - ALL LANGUAGE PAIRS")
    print("="*70)
    
    print(f"\n{'Language Pair':<20} {'Accuracy':<12} {'F1':<12} {'Precision':<12} {'Recall':<12}")
    print("-" * 70)
    
    for result in all_results:
        print(f"{result['language_pair'].upper():<20} "
              f"{result['accuracy']:<12.4f} "
              f"{result['f1']:<12.4f} "
              f"{result['precision']:<12.4f} "
              f"{result['recall']:<12.4f}")
    
    # Calculate averages
    if all_results:
        avg_acc = np.mean([r['accuracy'] for r in all_results])
        avg_f1 = np.mean([r['f1'] for r in all_results])
        avg_prec = np.mean([r['precision'] for r in all_results])
        avg_rec = np.mean([r['recall'] for r in all_results])
        
        print("-" * 70)
        print(f"{'AVERAGE':<20} "
              f"{avg_acc:<12.4f} "
              f"{avg_f1:<12.4f} "
              f"{avg_prec:<12.4f} "
              f"{avg_rec:<12.4f}")
    
    print("\n" + "="*70)
    print(f"⏱️ Total training time: {total_time:.1f} minutes")
    print("="*70)
    
    # Save summary
    os.makedirs('./outputs', exist_ok=True)
    summary = {
        'training_date': datetime.now().isoformat(),
        'total_time_minutes': total_time,
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'model': 'xlm-roberta-base',
        'results': all_results,
        'averages': {
            'accuracy': avg_acc,
            'f1': avg_f1,
            'precision': avg_prec,
            'recall': avg_rec
        } if all_results else None
    }
    
    with open('./outputs/all_languages_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n✅ Summary saved to: ./outputs/all_languages_summary.json")
    print("✅ Models saved to: ./models/checkpoints/[language_pair]/")
    
    print("\n🎉 ALL TRAINING COMPLETE!")

if __name__ == "__main__":
    main()
