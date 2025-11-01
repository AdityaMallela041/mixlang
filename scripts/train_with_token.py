"""
TOKEN-LEVEL LID - FINAL - Uses validation as test (test data is empty)
"""
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification
)
from datasets import Dataset
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import classification_report, accuracy_score
import os
import re

def parse_lince_csv_direct(csv_path):
    """Load and parse LinCE CSV directly"""
    df = pd.read_csv(csv_path, encoding='utf-8', encoding_errors='ignore')
    
    samples = []
    for idx, row in df.iterrows():
        words = re.findall(r"'([^']*)'", str(row['words']))
        langs = re.findall(r"'([^']*)'", str(row['lid']))
        
        words = [w.strip() for w in words if w.strip()]
        langs = [l.strip() for l in langs if l.strip()]
        
        if len(words) == len(langs) and 0 < len(words) <= 100:
            samples.append({'words': words, 'labels': langs})
    
    return samples

def train_language_pair(lang_pair, base_path='./data/raw/LinCE'):
    """Train on one language pair"""
    
    print("="*70)
    print(f"🚀 {lang_pair.upper()}")
    print("="*70)
    
    # Load
    print("\n📥 Loading...")
    train_data = parse_lince_csv_direct(f'{base_path}/lid_{lang_pair}_train.csv')
    val_data = parse_lince_csv_direct(f'{base_path}/lid_{lang_pair}_validation.csv')
    
    print(f"   Train: {len(train_data):,}")
    print(f"   Val/Test: {len(val_data):,}")
    
    if len(train_data) == 0 or len(val_data) == 0:
        print("   ⚠️ No data!")
        return None
    
    # Labels
    all_labels = set()
    for sample in train_data:
        all_labels.update(sample['labels'])
    
    label2id = {label: idx for idx, label in enumerate(sorted(all_labels))}
    id2label = {idx: label for label, idx in label2id.items()}
    num_labels = len(label2id)
    
    print(f"   Labels: {num_labels} ({', '.join(sorted(all_labels))})")
    
    # Split validation in half for val and test
    split_point = len(val_data) // 2
    val_split = val_data[:split_point]
    test_split = val_data[split_point:]
    
    print(f"   Split val into: {len(val_split)} val, {len(test_split)} test")
    
    # Datasets
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_split)
    test_dataset = Dataset.from_list(test_split)
    
    # Model
    print("\n🤖 Loading XLM-RoBERTa...")
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    model = AutoModelForTokenClassification.from_pretrained(
        "xlm-roberta-base",
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    
    # Tokenize
    def tokenize_and_align(examples):
        tokenized_inputs = tokenizer(
            examples["words"],
            truncation=True,
            is_split_into_words=True,
            max_length=128,
            padding='max_length'
        )
        
        labels = []
        for i, label in enumerate(examples["labels"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            previous_word_idx = None
            
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label2id[label[word_idx]])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            
            labels.append(label_ids)
        
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    print("🔄 Tokenizing...")
    train_dataset = train_dataset.map(tokenize_and_align, batched=True, remove_columns=['words', 'labels'])
    val_dataset = val_dataset.map(tokenize_and_align, batched=True, remove_columns=['words', 'labels'])
    test_dataset = test_dataset.map(tokenize_and_align, batched=True, remove_columns=['words', 'labels'])
    
    # Training
    output_dir = f'./models/final_{lang_pair}'
    os.makedirs(output_dir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=100,
        save_strategy='epoch',
        load_best_model_at_end=True,
        fp16=True,
        report_to="none",
    )
    
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        
        true_labels = []
        true_preds = []
        
        for pred, lab in zip(predictions, labels):
            for p, l in zip(pred, lab):
                if l != -100:
                    true_labels.append(id2label[l])
                    true_preds.append(id2label[p])
        
        acc = accuracy_score(true_labels, true_preds)
        return {'accuracy': acc}
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )
    
    print("\n🏋️ Training...")
    trainer.train()
    
    # Test
    print("\n📊 Testing...")
    pred_out = trainer.predict(test_dataset)
    predictions = np.argmax(pred_out.predictions, axis=2)
    labels = pred_out.label_ids
    
    true_labels = []
    true_preds = []
    
    for pred, lab in zip(predictions, labels):
        for p, l in zip(pred, lab):
            if l != -100:
                true_labels.append(id2label[l])
                true_preds.append(id2label[p])
    
    acc = accuracy_score(true_labels, true_preds)
    
    print("\n" + "="*70)
    print(f"✅ {lang_pair.upper()} RESULTS")
    print("="*70)
    print(f"Accuracy: {acc:.4f} | Tokens: {len(true_labels):,}")
    print("\n" + classification_report(true_labels, true_preds, digits=4, zero_division=0))
    
    # Save
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return {'pair': lang_pair, 'acc': acc, 'tokens': len(true_labels)}

def main():
    print("="*70)
    print("🚀 TOKEN-LEVEL LID - LinCE Benchmark")
    print("="*70)
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    
    results = []
    for lang in ['hineng', 'spaeng', 'nepeng']:
        try:
            result = train_language_pair(lang)
            if result:
                results.append(result)
        except Exception as e:
            print(f"\n⚠️ {lang} failed: {str(e)[:200]}")
            import traceback
            traceback.print_exc()
    
    if results:
        print("\n" + "="*70)
        print("📊 FINAL RESULTS")
        print("="*70)
        for r in results:
            print(f"{r['pair']:10s}: {r['acc']:.4f} ({r['tokens']:,} tokens)")
        print("-"*70)
        avg = sum(r['acc'] for r in results) / len(results)
        print(f"{'AVERAGE':10s}: {avg:.4f}")
        print("\n✅ Complete! Models in: ./models/final_*/")

if __name__ == "__main__":
    main()
