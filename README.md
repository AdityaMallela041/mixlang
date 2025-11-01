# 🌐 MixLang: Transformer-Based Code-Switching Detection in Multilingual Text

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

MixLang is a state-of-the-art token-level language identification system for code-switched multilingual text. Leveraging XLM-RoBERTa transformers, it achieves **97.15% average accuracy** across Hindi-English, Spanish-English, and Nepali-English language pairs.

---

## 🎯 **Key Features**

- **Token-Level Classification**: Identifies language (Hindi/English/Spanish/Nepali/Other) for each word in a sentence
- **Multi-Language Support**: Unified framework for 3 language pairs without language-specific rules
- **State-of-the-Art Performance**: 97.15% average accuracy on LinCE benchmark
- **Interactive CLI**: Real-time code-switching detection with confidence scores
- **Reproducible Research**: Standardized evaluation on LinCE datasets
- **Production-Ready**: Modular architecture for easy API integration

---

## 📊 **Results**

| Language Pair      | Accuracy | Precision | Recall | F1-Score |
|-------------------|----------|-----------|--------|----------|
| Hindi-English     | 96.41%   | 96.18%    | 96.41% | 96.29%   |
| Spanish-English   | 98.27%   | 98.15%    | 98.27% | 98.21%   |
| Nepali-English    | 96.77%   | 96.64%    | 96.77% | 96.70%   |
| **Average**       | **97.15%** | **96.99%** | **97.15%** | **97.07%** |

**Baseline Comparison**: +16.42 percentage points improvement over previous state-of-the-art (80.73%)

---

## 🚀 **Quick Start**

### **Prerequisites**

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- 16GB RAM minimum

### **Installation**

Clone the repository
git clone https://github.com/AdityaMallela041/mixlang.git
cd mixlang

Create virtual environment
python -m venv venv

Activate virtual environment
On Windows:
venv\Scripts\activate

On macOS/Linux:
source venv/bin/activate

Install dependencies
pip install -r requirements.txt

text

### **Download LinCE Dataset**

Create data directories
mkdir -p data/raw/LinCE
mkdir -p data/processed

Download LinCE benchmark datasets
Visit: https://ritual.uh.edu/lince/
Download lid_hineng, lid_spaeng, lid_nepeng datasets
Place CSV files in data/raw/LinCE/
text

---

## 💻 **Usage**

### **1. Data Preprocessing**

python scripts/preprocess.py
--lang_pair hineng
--input_dir data/raw/LinCE
--output_dir data/processed

text

Supported language pairs: `hineng`, `spaeng`, `nepeng`

### **2. Model Training**

python scripts/train.py
--lang_pair hineng
--data_dir data/processed/hineng
--output_dir models/hineng
--epochs 3
--learning_rate 2e-5
--batch_size 16

text

**Training Parameters:**
- **Model**: XLM-RoBERTa base (270M parameters)
- **Epochs**: 3
- **Learning Rate**: 2e-5
- **Batch Size**: 16
- **Optimizer**: AdamW (weight decay 0.01)
- **Max Sequence Length**: 128 tokens

### **3. Evaluation**

python scripts/eval.py
--model_path models/hineng
--test_data data/processed/hineng/test.csv
--output_dir results/hineng

text

Generates:
- Classification metrics (accuracy, precision, recall, F1)
- Confusion matrices
- Learning curves
- Token-level error analysis

### **4. Interactive CLI Demo**

python scripts/implementation.py

text

**Example Session:**
======================================================================
🚀 MixLang+ LIVE CODE-SWITCHING DETECTION
Select language pair:

Hindi-English

Spanish-English

Nepali-English

Choice: 1

Enter sentence (or 'quit' to exit): Kal I went to market aur shopping ki

📊 TOKEN-LEVEL ANALYSIS
Token Language Confidence
"Kal" Hindi 0.98
"I" English 0.99
"went" English 0.99
"to" English 0.99
"market" English 0.97
"aur" Hindi 0.99
"shopping" English 0.98
"ki" Hindi 0.99
text

---

```
mixlang/
├── configs/
│   └── config.yaml              # Configuration settings
│
├── data/
│   ├── raw/                     # Raw datasets (not in repo - see setup)
│   │   └── LinCE/               # LinCE benchmark data
│   └── processed/               # Preprocessed data ready for training
│       ├── hineng/
│       ├── spaeng/
│       └── nepeng/
│
├── models/                      # Saved model checkpoints (not in repo)
│   ├── hineng/
│   ├── spaeng/
│   └── nepeng/
│
├── scripts/
│   ├── preprocess_lince.py      # Data preprocessing
│   ├── train_with_token.py      # Token-level training
│   ├── train_pos.py             # POS tagging training
│   ├── eval.py                  # Comprehensive evaluation
│   ├── eval1.py                 # Alternative evaluation
│   ├── implementation.py        # Interactive CLI demo
│   └── inspect_lince_files.py   # Dataset inspection utility
│
├── src/
│   ├── data/
│   │   ├── dataset_loader.py    # Dataset loading utilities
│   │   └── preprocessor.py      # Text preprocessing
│   ├── models/
│   │   ├── xlm_roberta_model.py # XLM-RoBERTa implementation
│   │   └── indicbert_model.py   # IndicBERT implementation
│   ├── evaluation/
│   │   ├── evaluator.py         # Model evaluation
│   │   └── metrics.py           # Performance metrics
│   └── utils/
│       └── helpers.py           # Utility functions
│
├── results/
│   ├── confusion_matrix_*.png   # Confusion matrices
│   ├── baseline_comparison.png  # Performance comparison
│   ├── training_curves_*.png    # Learning curves
│   └── classification_report.csv # Detailed metrics
│
├── .gitignore                   # Git exclusions
├── .gitattributes               # Git LFS configuration
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
└── setup.py                     # Package setup
```

---

## 🛠️ **Technical Details**

### **Architecture**

Input Text
↓
XLM-RoBERTa Tokenizer (SentencePiece)
↓
XLM-RoBERTa Base Model (12 layers, 270M params)
↓
Token Classification Head (Linear + Softmax)
↓
Language Labels + Confidence Scores

text

### **Model Specifications**

- **Base Model**: `xlm-roberta-base` (Hugging Face)
- **Architecture**: 12-layer Transformer
- **Hidden Size**: 768
- **Attention Heads**: 12
- **Parameters**: 270 million
- **Pre-training Languages**: 100 languages
- **Vocabulary Size**: 250,000 tokens

### **Hyperparameters**

| Parameter          | Value    | Justification                      |
|--------------------|----------|------------------------------------|
| Learning Rate      | 2e-5     | Optimal for fine-tuning large LMs  |
| Epochs             | 3        | Prevents overfitting               |
| Batch Size         | 16       | Fits in 16GB GPU memory            |
| Weight Decay       | 0.01     | L2 regularization                  |
| Max Seq Length     | 128      | Covers 95% of sentences            |
| Optimizer          | AdamW    | Handles sparse gradients           |

---

## 📈 **Evaluation Metrics**

### **Token-Level Metrics**

- **Accuracy**: Overall correctness
- **Precision**: Per-class precision (Hindi/English/Other)
- **Recall**: Per-class recall
- **F1-Score**: Harmonic mean of precision and recall

### **Error Analysis**

Common error patterns:
1. **Named Entities** (40%): "WhatsApp", "Mumbai" - ambiguous language assignment
2. **Loanwords** (35%): "computer", "mobile" - used in both languages
3. **Romanized Text** (20%): Inconsistent transliteration ("thik" vs "theek")
4. **Code-Mixed Compounds** (5%): Hybrid constructions

---

## 🎓 **Citation**

If you use MixLang in your research, please cite:

@inproceedings{mallela2025mixlang,
title={MixLang: Transformer-Based Code-Switching Detection in Multilingual Text},
author={Mallela, Aditya and Liharini, M. and Abhijeet, M.},
booktitle={Proceedings of VBIT Research Symposium},
year={2025},
institution={Vignana Bharathi Institute of Technology}
}

text

---

## 🔬 **Research Paper**

Read the full paper: [MixLang: Transformer-Based Code-Switching Detection](docs/paper.pdf)

**Abstract**: MixLang addresses the challenge of code-switching detection in multilingual text, where speakers blend words from multiple languages within a single sentence. Our system leverages the XLM-RoBERTa transformer model, fine-tuned for token-level language identification across Hindi-English, Spanish-English, and Nepali-English pairs...

---

## 🤝 **Contributing**

We welcome contributions! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

**Areas for Contribution:**
- Support for additional language pairs (Tamil-English, Bengali-English)
- Model optimization (quantization, distillation)
- Web API deployment
- Mobile app integration
- Improved error handling

---

## 🐛 **Known Issues**

- Named entity classification accuracy (95%) lower than average (97%)
- Romanized Hindi handling less accurate than Devanagari
- Inference time increases linearly with sentence length (>128 tokens)
- GPU memory requirement (16GB) limits accessibility

**Roadmap for fixes**: See [Issues](https://github.com/AdityaMallela041/mixlang/issues)

---

## 📜 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 **Authors**

- **M. Liharini** - [GitHub](https://github.com/liharini) | Roll No: 22P61A66A3
- **MSV. Aditya Phani Kumar** - [GitHub](https://github.com/AdityaMallela041) | Roll No: 22P61A6697
- **M. Abhijeet** - [GitHub](https://github.com/abhijeet) | Roll No: 22P61A6699

**Supervisor**: Mrs. P. Laxmi  
**Institution**: Vignana Bharathi Institute of Technology (VBIT), Hyderabad  
**Department**: Computer Science & Engineering (AI & ML)

---

## 🙏 **Acknowledgments**

- **LinCE Benchmark**: Aguilar et al. (2020) for standardized evaluation datasets
- **Hugging Face**: For XLM-RoBERTa model and Transformers library
- **PyTorch Team**: For deep learning framework
- **VBIT Research Committee**: For funding and support

---

## 📧 **Contact**

For questions or collaboration:
- **Email**: adityamallela041@gmail.com
- **LinkedIn**: [Aditya Mallela](https://www.linkedin.com/in/adityamallela/)
- **Issues**: [GitHub Issues](https://github.com/AdityaMallela041/mixlang/issues)

---

## 📊 **Project Status**

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Code Coverage](https://img.shields.io/badge/coverage-85%25-yellow)
![Maintained](https://img.shields.io/badge/maintained-yes-blue)

**Current Version**: 1.0.0  
**Last Updated**: November 2025

---

**Made with ❤️ by Team MixLang @ VBIT**