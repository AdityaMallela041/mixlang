"""
Preprocess LinCE - ULTRA ROBUST for different CSV formats
"""
import pandas as pd
import re
import json
import os
import ast

def parse_lince_row_v1(words_str, lid_str):
    """Method 1: Regex extraction from string"""
    try:
        words = re.findall(r"'([^']*)'", str(words_str))
        langs = re.findall(r"'([^']*)'", str(lid_str))
        
        words = [w.strip() for w in words if w.strip()]
        langs = [l.strip() for l in langs if l.strip()]
        
        if len(words) == len(langs) and 0 < len(words) <= 100:
            return words, langs
    except:
        pass
    return None, None

def parse_lince_row_v2(words_str, lid_str):
    """Method 2: Literal eval (for actual Python lists)"""
    try:
        # Try to evaluate as Python literal
        words = ast.literal_eval(str(words_str))
        langs = ast.literal_eval(str(lid_str))
        
        if isinstance(words, (list, tuple)) and isinstance(langs, (list, tuple)):
            words = [str(w).strip() for w in words if str(w).strip()]
            langs = [str(l).strip() for l in langs if str(l).strip()]
            
            if len(words) == len(langs) and 0 < len(words) <= 100:
                return words, langs
    except:
        pass
    return None, None

def parse_lince_row_v3(words_str, lid_str):
    """Method 3: Space-separated (backup format)"""
    try:
        words = str(words_str).strip().split()
        langs = str(lid_str).strip().split()
        
        if len(words) == len(langs) and 0 < len(words) <= 100:
            return words, langs
    except:
        pass
    return None, None

def parse_lince_row_multi_method(words_str, lid_str):
    """Try all methods"""
    for method in [parse_lince_row_v1, parse_lince_row_v2, parse_lince_row_v3]:
        result = method(words_str, lid_str)
        if result[0] is not None:
            return result
    return None, None

def preprocess_language_pair(lang_pair, base_path='./data/raw/LinCE'):
    """Preprocess one language pair"""
    print(f"\n{'='*60}")
    print(f"Processing {lang_pair.upper()}")
    print(f"{'='*60}")
    
    output_dir = f'./data/processed/{lang_pair}'
    os.makedirs(output_dir, exist_ok=True)
    
    for split in ['train', 'validation', 'test']:
        print(f"\n📄 {split}...")
        
        # Load CSV
        try:
            df = pd.read_csv(
                f'{base_path}/lid_{lang_pair}_{split}.csv',
                encoding='utf-8',
                encoding_errors='ignore'
            )
        except:
            df = pd.read_csv(
                f'{base_path}/lid_{lang_pair}_{split}.csv',
                encoding='latin-1',
                encoding_errors='ignore'
            )
        
        print(f"   Loaded: {len(df)} rows")
        
        # Check columns
        if 'lid' not in df.columns:
            print(f"   ⚠️ No 'lid' column! Columns: {df.columns.tolist()}")
            # Try to find alternative column names
            possible_lid_cols = [c for c in df.columns if 'label' in c.lower() or 'tag' in c.lower()]
            if possible_lid_cols:
                print(f"   Found possible label column: {possible_lid_cols[0]}")
                df['lid'] = df[possible_lid_cols[0]]
            else:
                print(f"   ⚠️ Cannot find language labels!")
                continue
        
        # Parse all rows with multiple methods
        valid_samples = []
        method_stats = {'v1': 0, 'v2': 0, 'v3': 0, 'failed': 0}
        
        for idx, row in df.iterrows():
            # Try each method
            words, langs = parse_lince_row_v1(row['words'], row['lid'])
            if words:
                method_stats['v1'] += 1
            else:
                words, langs = parse_lince_row_v2(row['words'], row['lid'])
                if words:
                    method_stats['v2'] += 1
                else:
                    words, langs = parse_lince_row_v3(row['words'], row['lid'])
                    if words:
                        method_stats['v3'] += 1
                    else:
                        method_stats['failed'] += 1
            
            if words is not None:
                valid_samples.append({
                    'words': words,
                    'labels': langs
                })
        
        print(f"   Valid: {len(valid_samples)} samples")
        print(f"   Methods used: regex={method_stats['v1']}, literal={method_stats['v2']}, space={method_stats['v3']}, failed={method_stats['failed']}")
        
        # Save
        output_file = f'{output_dir}/{split}.json'
        with open(output_file, 'w', encoding='utf-8', errors='ignore') as f:
            json.dump(valid_samples, f, ensure_ascii=False, indent=2)
        
        print(f"   ✅ Saved: {output_file}")
    
    # Labels
    train_file = f'{output_dir}/train.json'
    with open(train_file, encoding='utf-8') as f:
        train_data = json.load(f)
    
    if len(train_data) == 0:
        return 0
    
    all_labels = set()
    for sample in train_data:
        all_labels.update(sample['labels'])
    
    label2id = {label: idx for idx, label in enumerate(sorted(all_labels))}
    id2label = {idx: label for label, idx in label2id.items()}
    
    with open(f'{output_dir}/labels.json', 'w', encoding='utf-8') as f:
        json.dump({'label2id': label2id, 'id2label': id2label}, f, indent=2)
    
    print(f"\n📋 Labels: {len(label2id)}")
    for label, idx in sorted(label2id.items(), key=lambda x: x[1]):
        print(f"   {idx}: {label}")
    
    return len(label2id)

def main():
    print("="*60)
    print("🔄 Preprocessing LinCE Data (Multi-Method)")
    print("="*60)
    
    language_pairs = ['hineng', 'spaeng', 'nepeng']
    summary = []
    
    for lang_pair in language_pairs:
        try:
            num_labels = preprocess_language_pair(lang_pair)
            
            # Load stats
            stats = {}
            for split in ['train', 'validation', 'test']:
                file_path = f'./data/processed/{lang_pair}/{split}.json'
                with open(file_path, encoding='utf-8') as f:
                    stats[split] = len(json.load(f))
            
            summary.append({
                'pair': lang_pair,
                'train': stats['train'],
                'val': stats['validation'],
                'test': stats['test'],
                'labels': num_labels
            })
            
            print(f"\n✅ {lang_pair.upper()}: {stats['train']:,} train | {stats['validation']:,} val | {stats['test']:,} test")
        
        except Exception as e:
            print(f"\n⚠️ {lang_pair}: {str(e)[:150]}")
    
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    print(f"{'Language':<15} {'Train':>8} {'Val':>8} {'Test':>8} {'Labels':>8}")
    print("-"*60)
    
    for s in summary:
        print(f"{s['pair'].upper():<15} {s['train']:>8,} {s['val']:>8,} {s['test']:>8,} {s['labels']:>8}")
    
    # Check if ready
    has_test = all(s['test'] > 0 for s in summary)
    
    if has_test:
        print("\n🚀 Ready to train! Run: python scripts\\train_with_preprocessed.py")
    else:
        print("\n⚠️ Test sets are empty! Using validation as test for now...")
        print("You can still train, but will need to use validation set for evaluation.")

if __name__ == "__main__":
    main()
