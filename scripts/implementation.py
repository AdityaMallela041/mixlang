"""
MixLang+ Live Code-Switching Demo
Token-level Language Identification with Out-of-Domain Detection
Author: Aditya Mallela | VBIT
Model: XLM-RoBERTa | Accuracy: 97.39%
"""
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch.nn.functional as F
import sys
import numpy as np

# Language label mappings - CORRECTED ORDER
LABEL_MAPPINGS = {
    'Hindi-English': {
        'lang1': 'English',
        'lang2': 'Hindi',
        'other': 'Other',
        'ne': 'Named Entity',
        'mixed': 'Mixed',
        'ambiguous': 'Ambiguous',
        'fw': 'Foreign Word',
        'unk': 'Unknown'
    },
    'Spanish-English': {
        'lang1': 'English',
        'lang2': 'Spanish',
        'other': 'Other',
        'ne': 'Named Entity',
        'mixed': 'Mixed',
        'ambiguous': 'Ambiguous',
        'fw': 'Foreign Word',
        'unk': 'Unknown'
    },
    'Nepali-English': {
        'lang1': 'English',
        'lang2': 'Nepali',
        'other': 'Other',
        'ne': 'Named Entity',
        'mixed': 'Mixed',
        'ambiguous': 'Ambiguous',
        'fw': 'Foreign Word',
        'unk': 'Unknown'
    }
}

def load_model(model_path):
    """Load pre-trained model and tokenizer"""
    print(f"   Loading model... ", end='', flush=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        model.eval()
        print("✓")
        return tokenizer, model
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        sys.exit(1)

def get_readable_label(label, lang_pair):
    """Convert model label to human-readable language name"""
    mapping = LABEL_MAPPINGS.get(lang_pair, {})
    return mapping.get(label, label)

def detect_out_of_domain(results, all_logits, label2id):
    """
    Detect out-of-domain input using model probability analysis
    Pure model-based reasoning - no external word lists
    """
    modified_results = []
    warnings = []
    
    # Analyze pattern
    lang1_count = sum(1 for r in results if r['model_label'] == 'lang1')
    lang2_count = sum(1 for r in results if r['model_label'] == 'lang2')
    total = len(results)
    avg_conf = np.mean([r['confidence'] for r in results])
    
    # Suspicious pattern: overwhelming non-English with high confidence
    suspicious = (lang2_count / total > 0.7) and (avg_conf > 0.95)
    
    relabeled_count = 0
    
    for i, r in enumerate(results):
        new_result = r.copy()
        new_result['relabeled'] = False
        
        # Only relabel lang2 (non-English) predictions in suspicious contexts
        if suspicious and r['model_label'] == 'lang2':
            # Check if English probability is also viable
            logits_tensor = torch.tensor(all_logits[i])
            probs = F.softmax(logits_tensor, dim=-1)
            
            english_idx = label2id.get('lang1', -1)
            english_prob = probs[english_idx].item() if english_idx != -1 else 0.0
            
            # If English probability is very low, mark as "Other"
            if english_prob < 0.05:
                new_result['label'] = 'Other'
                new_result['original_label'] = r['label']
                new_result['relabeled'] = True
                relabeled_count += 1
        
        modified_results.append(new_result)
    
    # Generate warnings
    if suspicious:
        warnings.append("⚠️  PATTERN DETECTED: Anomalous language distribution.")
        warnings.append("   Model analysis suggests out-of-domain input.")
    
    if relabeled_count > 0:
        pct = (relabeled_count / len(results)) * 100
        warnings.append(f"ℹ️  {relabeled_count} token(s) ({pct:.1f}%) marked as 'Other' (unknown language).")
    
    return modified_results, warnings

def predict(text, tokenizer, model, lang_pair):
    """Make predictions with out-of-domain detection"""
    # Clean input: remove quotes and extra whitespace
    text = text.strip()
    # Remove surrounding quotes (all types)
    for quote_type in ['"', "'", '"', '"', '\'', '\'']:
        if text.startswith(quote_type) and text.endswith(quote_type) and len(text) > 1:
            text = text[1:-1].strip()
            break
    
    words = text.split()
    if not words:
        return [], []
    
    inputs = tokenizer(words, is_split_into_words=True, return_tensors='pt')
    word_ids = inputs.word_ids(batch_index=0)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0]
        probs = F.softmax(logits, dim=-1)
        pred_ids = torch.argmax(probs, dim=-1).tolist()
        confs = torch.max(probs, dim=-1).values.tolist()
    
    id2label = model.config.id2label
    label2id = model.config.label2id
    previous_word_idx = None
    results = []
    all_logits = []
    
    for idx, (word_idx, pred_id, prob) in enumerate(zip(word_ids, pred_ids, confs)):
        if word_idx is not None and word_idx != previous_word_idx:
            model_label = id2label[pred_id]
            readable_label = get_readable_label(model_label, lang_pair)
            results.append({
                'word': words[word_idx],
                'label': readable_label,
                'model_label': model_label,
                'confidence': prob
            })
            all_logits.append(logits[idx].tolist())
            previous_word_idx = word_idx
    
    # Apply out-of-domain detection
    results, warnings = detect_out_of_domain(results, all_logits, label2id)
    return results, warnings

def is_codemixed(results):
    """Check if text contains multiple main languages"""
    main_langs = [r['label'] for r in results 
                  if r['label'] not in ['Other', 'Named Entity', 'Mixed', 'Ambiguous', 'Unknown', 'Foreign Word']]
    return len(set(main_langs)) > 1

def print_results(results, is_mixed, warnings):
    """Print formatted analysis results"""
    print("\n" + "="*70)
    print("📊 TOKEN-LEVEL ANALYSIS")
    print("="*70)
    print(f"{'Token':<20} {'Language':<20} {'Confidence':>12}")
    print("-"*70)
    
    for r in results:
        confidence_bar = "█" * int(r['confidence'] * 10)
        label_display = r['label']
        if r.get('relabeled', False):
            label_display = f"{r['label']} ⚠️"
        print(f"{r['word']:<20} {label_display:<20} {r['confidence']*100:>6.2f}%  {confidence_bar}")
    
    print("="*70)
    
    # Statistics
    avg_conf = np.mean([r['confidence'] for r in results])
    lang_counts = {}
    for r in results:
        lang_counts[r['label']] = lang_counts.get(r['label'], 0) + 1
    
    print(f"\n{'Code-Mixing Status:':<25} {'YES ✓' if is_mixed else 'NO ✗'}")
    print(f"{'Average Confidence:':<25} {avg_conf*100:.2f}%")
    print(f"{'Total Tokens:':<25} {len(results)}")
    
    print(f"\n📈 Language Distribution:")
    for lang, count in sorted(lang_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(results)) * 100
        print(f"   {lang:<20}: {count:>3} tokens ({percentage:>5.1f}%)")
    
    if warnings:
        print("\n🔔 DETECTION ALERTS:")
        for warning in warnings:
            print(f"   {warning}")
    
    print("="*70 + "\n")

def select_model():
    """Display model selection menu"""
    model_map = {
        '1': ('Hindi-English', './models/final_hineng'),
        '2': ('Spanish-English', './models/final_spaeng'),
        '3': ('Nepali-English', './models/final_nepeng')
    }
    
    print("\n" + "="*70)
    print("🌍 SELECT LANGUAGE PAIR")
    print("="*70)
    print("  1. Hindi-English")
    print("  2. Spanish-English")
    print("  3. Nepali-English")
    print("-"*70)
    
    while True:
        choice = input("Enter choice [1-3]: ").strip()
        if choice in model_map:
            return model_map[choice]
        print("❌ Invalid choice. Please enter 1, 2, or 3.")

def main():
    """Main interactive loop"""
    print("\n" + "="*70)
    print("🚀 MixLang+ LIVE CODE-SWITCHING DEMO")
    print("="*70)
    print("Token-level Language Identification using XLM-RoBERTa")
    print("Trained on LinCE Benchmark | Average Accuracy: 97.39%")
    print("="*70)
    
    # Initial model selection
    lang_pair, model_path = select_model()
    print(f"\n✓ Selected: {lang_pair}")
    tokenizer, model = load_model(model_path)
    
    print("\n💡 Commands:")
    print("   - Type your sentence to analyze (no quotes needed)")
    print("   - Type 'change' to switch language models")
    print("   - Type 'exit' to quit")
    print("\n💡 TIP: Works best with natural code-switched sentences!\n")
    
    while True:
        print("-"*70)
        user_input = input("Enter sentence: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() == 'exit':
            print("\n" + "="*70)
            print("👋 Thank you for using MixLang+!")
            print("Project by: Aditya Mallela | VBIT")
            print("GitHub: github.com/adityamallela")
            print("="*70 + "\n")
            break
        
        if user_input.lower() == 'change':
            lang_pair, model_path = select_model()
            print(f"\n✓ Switched to: {lang_pair}")
            tokenizer, model = load_model(model_path)
            print()
            continue
        
        try:
            results, warnings = predict(user_input, tokenizer, model, lang_pair)
            if not results:
                print("⚠️  No valid tokens found. Please try a different sentence.")
                continue
            
            is_mixed = is_codemixed(results)
            print_results(results, is_mixed, warnings)
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again with a different sentence.\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("👋 Interrupted. Thank you for using MixLang+!")
        print("="*70 + "\n")
        sys.exit(0)
