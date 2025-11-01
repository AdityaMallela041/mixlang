"""
Check what LinCE files we actually have
"""
import os

lince_path = './data/raw/LinCE'

print("="*70)
print("📁 LinCE Directory Contents")
print("="*70)

if os.path.exists(lince_path):
    files = sorted(os.listdir(lince_path))
    
    # Group by task
    lid_files = [f for f in files if f.startswith('lid_')]
    pos_files = [f for f in files if f.startswith('pos_')]
    ner_files = [f for f in files if f.startswith('ner_')]
    sa_files = [f for f in files if f.startswith('sa_') or f.startswith('sentiment_')]
    other_files = [f for f in files if not any(f.startswith(x) for x in ['lid_', 'pos_', 'ner_', 'sa_', 'sentiment_'])]
    
    print(f"\n📊 Language Identification (LID): {len(lid_files)} files")
    for f in lid_files:
        print(f"   - {f}")
    
    print(f"\n🏷️  POS Tagging: {len(pos_files)} files")
    if pos_files:
        for f in pos_files:
            print(f"   - {f}")
    else:
        print("   ⚠️  No POS files found")
    
    print(f"\n🏷️  Named Entity Recognition (NER): {len(ner_files)} files")
    if ner_files:
        for f in ner_files:
            print(f"   - {f}")
    else:
        print("   ⚠️  No NER files found")
    
    print(f"\n😊 Sentiment Analysis: {len(sa_files)} files")
    if sa_files:
        for f in sa_files:
            print(f"   - {f}")
    else:
        print("   ⚠️  No SA files found")
    
    if other_files:
        print(f"\n📄 Other files: {len(other_files)}")
        for f in other_files:
            print(f"   - {f}")
    
    print("\n" + "="*70)
    print("📋 Summary")
    print("="*70)
    print(f"Total files: {len(files)}")
    print(f"\nAvailable tasks:")
    if lid_files:
        print("  ✅ Language Identification (LID)")
    if pos_files:
        print("  ✅ POS Tagging")
    if ner_files:
        print("  ✅ Named Entity Recognition")
    if sa_files:
        print("  ✅ Sentiment Analysis")
    
    if not (pos_files or ner_files or sa_files):
        print("\n⚠️  WARNING: Only LID data is available!")
        print("   LinCE benchmark has separate downloads for each task.")
        print("   You may need to download additional datasets.")
        
else:
    print(f"❌ Directory not found: {lince_path}")
