# AI-Generated Peer Review Detection

## Overview
This project introduces a **novel PSR (Part-of-Speech Shift Ratio) model** to detect AI-generated peer reviews in academic research. The model analyzes linguistic patterns in peer review text to distinguish between human-written and AI-generated reviews.

## Research Paper
📄 **"Detecting AI-Generated Peer Reviews Using POS Shift Ratio and Lexical Diversity Features"**
- **Paper Link**: [\[Paper URL\]](https://drive.google.com/file/d/1LaLg_4-SBgZkjXOM9i0GAFshrMOS6r2C/view?usp=sharing)

## Problem Statement
AI-generated peer reviews pose a significant threat to academic integrity. Our PSR model addresses this by analyzing sentence structure variations, but it's **vulnerable to paraphrasing attacks**. To improve robustness, we combine PSR with TTR (Type-Token Ratio) features.

## Dataset
- **Source**: 1480 research papers from **ISLR** and **NeurIPS** conferences
- **Human Reviews**: Authentic peer reviews from these papers
- **AI Reviews**: Generated using **ChatGPT** by providing the actual research papers
- **Paraphrased Dataset**: Both human and AI reviews paraphrased using **Gemini** to test model robustness

## Key Features

### 1. PSR (Part-of-Speech Shift Ratio) - Novel Approach
- **What it measures**: Sentence structure variation between consecutive sentences
- **Why it works**: Human reviewers naturally vary sentence structure more than AI
- **Novelty**: First application to peer review detection

### 2. TTR (Type-Token Ratio) - Robustness Feature
- **What it measures**: Vocabulary diversity (unique words / total words)
- **Why it's added**: PSR is vulnerable to paraphrasing attacks
- **Combined approach**: PSR + TTR for better attack resistance

## Performance Results

### Normal Dataset (Original Reviews)
- **PSR Model**: 87.16% accuracy
- **PSR+TTR Model**: 88.85% accuracy

### Paraphrased Dataset (Gemini Attack)
- **PSR Model**: 63.67% accuracy (significant drop)
- **PSR+TTR Model**: 74.36% accuracy (improved robustness)

## Key Findings

1. **PSR is effective** for detecting AI-generated peer reviews (87% accuracy)
2. **Paraphrasing attacks significantly reduce** PSR model performance
3. **Adding TTR improves robustness** against paraphrasing (74% vs 64%)
4. **Feature combination** always outperforms single-feature models

## Files

- `psr_model.py`: Novel PSR-only detection model
- `psr+ttr_model.py`: Combined PSR+TTR model for attack resistance
- `dataset.csv`: Original human/AI peer reviews
- `para_dataset.csv`: Gemini-paraphrased reviews for attack testing

## Usage

```bash
# Install dependencies
pip install pandas numpy scikit-learn spacy matplotlib seaborn scipy
python -m spacy download en_core_web_sm

# Run models
python psr_model.py          # PSR only
python psr+ttr_model.py      # PSR + TTR
```

## Technical Details

- **Algorithm**: Logistic Regression
- **Features**: PSR (sentence structure) + TTR (vocabulary diversity)
- **Data Split**: 80% training, 20% testing
- **NLP Pipeline**: spaCy for part-of-speech analysis

## Applications

- **Academic Integrity**: Detect AI-generated peer reviews
- **Conference Management**: Maintain review quality
- **Research Validation**: Ensure authentic peer feedback

## Limitations

- **Paraphrasing Vulnerability**: PSR alone can be evaded
- **Text Length**: Requires sufficient sentence count
- **Language Dependence**: Currently English-only

## Future Work

- **Advanced Attack Resistance**: Adversarial training
- **Multi-language Support**: Extend beyond English
- **Deep Learning Integration**: BERT/GPT analysis

---

*This work addresses the critical challenge of AI-generated peer review detection in academic publishing.*
