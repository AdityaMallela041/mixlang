"""
POS Tagging for Code-Switched Text
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
from sklearn.metrics import accuracy_score, f1_score, classification_report
import re
import os

def parse_pos_csv(csv_path):
    """Parse LinCE POS tagging CSV"""
    print(f"   Loading: {csv_path}")
    
    if not os.path.exists(csv_path):
        print(f"   ⚠️  File not found, skipping")
        return []
    
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except Exception as e:
        print(f"   ⚠️  Error reading file: {e}")
        return []
    
    if len(df) == 0:
        print(f"   ⚠️  Empty file")
        return []
    
    samples = []
    
    for idx, row in df.iterrows():
        try:
            # Parse words and POS tags from string representation
            words = re.findall(r"'([^']*)'", str(row['words']))
            pos_tags = re.findall(r"'([^']*)'", str(row['pos']))
            
            words = [w.strip() for w in words if w.strip()]
            pos_tags = [p.strip() for p in pos_tags if p.strip()]
            
            if len(words) == len(pos_tags) and len(words) > 0:
                samples.append({'words': words, 'pos_tags': pos_tags})
        except:
            continue
    
    return samples

def train_pos_tagging(lang_code='spaeng', lang_name='Spanish-English'):
    """Train POS tagging model"""
    
    print("="*70)
    print(f"🚀 POS Tagging: {lang_name}")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    # Load data
    base_path = './data/raw/LinCE'
    print("📥 Loading datasets...")
    train_data = parse_pos_csv(f'{base_path}/pos_{lang_code}_train.csv')
    val_data = parse_pos_csv(f'{base_path}/pos_{lang_code}_validation.csv')
    test_data = parse_pos_csv(f'{base_path}/pos_{lang_code}_test.csv')
    
    # If test is empty, use part of validation
    if len(test_data) == 0:
        print("   ⚠️  Test set empty, using validation data for testing")
        test_data = val_data
    
    print(f"\n📊 Dataset:")
    print(f"   Train: {len(train_data)} samples")
    print(f"   Val:   {len(val_data)} samples")
    print(f"   Test:  {len(test_data)} samples")
    
    if len(train_data) == 0:
        print("❌ No training data available!")
        return None
    
    # Get unique POS tags
    all_tags = set()
    for sample in train_data + val_data:
        all_tags.update(sample['pos_tags'])
    
    tag2id = {tag: idx for idx, tag in enumerate(sorted(all_tags))}
    id2tag = {idx: tag for tag, idx in tag2id.items()}
    num_labels = len(tag2id)
    
    print(f"\n📋 POS Tags: {num_labels} classes")
    tags_list = sorted(list(all_tags))
    print(f"   {', '.join(tags_list[:15])}{'...' if len(tags_list) > 15 else ''}")
    
    # Create datasets
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    test_dataset = Dataset.from_list(test_data)
    
    # Load model
    print("\n🤖 Loading XLM-RoBERTa...")
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    model = AutoModelForTokenClassification.from_pretrained(
        "xlm-roberta-base",
        num_labels=num_labels,
        id2label=id2tag,
        label2id=tag2id
    )
    model.to(device)
    
    # Tokenize function
    def tokenize_and_align(examples):
        tokenized_inputs = tokenizer(
            examples["words"],
            truncation=True,
            is_split_into_words=True,
            max_length=128,
            padding='max_length'
        )
        
        labels = []
        for i, pos_tags in enumerate(examples["pos_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            previous_word_idx = None
            
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(tag2id[pos_tags[word_idx]])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            
            labels.append(label_ids)
        
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    print("🔄 Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_and_align, batched=True, 
                                     remove_columns=['words', 'pos_tags'])
    val_dataset = val_dataset.map(tokenize_and_align, batched=True,
                                  remove_columns=['words', 'pos_tags'])
    test_dataset = test_dataset.map(tokenize_and_align, batched=True,
                                    remove_columns=['words', 'pos_tags'])
    
    # Training configuration
    output_dir = f'./models/pos_{lang_code}'
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
        save_total_limit=1,  # Only keep best model
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
                    true_labels.append(id2tag[l])
                    true_preds.append(id2tag[p])
        
        acc = accuracy_score(true_labels, true_preds)
        f1 = f1_score(true_labels, true_preds, average='weighted', zero_division=0)
        return {'accuracy': acc, 'f1': f1}
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )
    
    print("\n🏋️ Training (3 epochs)...")
    trainer.train()
    
    # Evaluate on test set
    print("\n📊 Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    
    # Detailed predictions for classification report
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=2)
    labels = predictions.label_ids
    
    true_labels = []
    true_preds = []
    
    for pred, lab in zip(preds, labels):
        for p, l in zip(pred, lab):
            if l != -100:
                true_labels.append(id2tag[l])
                true_preds.append(id2tag[p])
    
    print("\n" + "="*70)
    print(f"✅ RESULTS - POS Tagging ({lang_name})")
    print("="*70)
    print(f"Accuracy: {test_results['eval_accuracy']:.4f}")
    print(f"F1 Score: {test_results['eval_f1']:.4f}")
    print(f"Tokens:   {len(true_labels):,}")
    
    print("\n📋 Per-tag Performance (top 10):")
    # Show per-tag metrics
    from sklearn.metrics import precision_recall_fscore_support
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        true_labels, true_preds, average=None, labels=sorted(list(set(true_labels))), zero_division=0
    )
    
    tag_metrics = []
    for tag, p, r, f, s in zip(sorted(list(set(true_labels))), precisions, recalls, f1s, supports):
        tag_metrics.append((tag, f, s))
    
    # Sort by frequency
    tag_metrics.sort(key=lambda x: x[2], reverse=True)
    
    for tag, f1_score, support in tag_metrics[:10]:
        print(f"   {tag:12s}: F1={f1_score:.4f}  (support: {int(support):,})")
    
    # Save model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n💾 Model saved: {output_dir}")
    
    # Clean up checkpoints to save space
    import shutil
    for item in os.listdir(output_dir):
        if item.startswith('checkpoint-'):
            shutil.rmtree(os.path.join(output_dir, item))
    
    return {
        'accuracy': test_results['eval_accuracy'],
        'f1': test_results['eval_f1'],
        'tokens': len(true_labels),
        'language': lang_name,
        'num_tags': num_labels
    }

if __name__ == "__main__":
    print("\n🌍 Training POS Tagging Models")
    print("="*70)
    
    # Train both Spanish-English and Hindi-English
    results = {}
    
    # Spanish-English
    print("\n")
    result_spa = train_pos_tagging('spaeng', 'Spanish-English')
    if result_spa:
        results['spaeng'] = result_spa
    
    # Hindi-English
    print("\n")
    result_hin = train_pos_tagging('hineng', 'Hindi-English')
    if result_hin:
        results['hineng'] = result_hin
    
    # Summary
    if results:
        print("\n" + "="*70)
        print("📊 POS TAGGING - FINAL SUMMARY")
        print("="*70)
        for lang_code, res in results.items():
            print(f"\n{res['language']}:")
            print(f"  Accuracy:  {res['accuracy']:.4f}")
            print(f"  F1 Score:  {res['f1']:.4f}")
            print(f"  Tokens:    {res['tokens']:,}")
            print(f"  POS Tags:  {res['num_tags']}")
        
        avg_acc = np.mean([r['accuracy'] for r in results.values()])
        avg_f1 = np.mean([r['f1'] for r in results.values()])
        
        print(f"\nAverage Across Languages:")
        print(f"  Accuracy: {avg_acc:.4f}")
        print(f"  F1 Score: {avg_f1:.4f}")
        
        print("\n✅ POS Tagging training complete!")
    else:
        print("\n❌ No models trained successfully")
