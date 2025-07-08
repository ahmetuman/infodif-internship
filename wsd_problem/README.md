1. **Find Common Words**: Compare words between input and candidate sentences
2. **Smart Method Selection**:
   - **Meaningful words** (homophones like "bank", "play", "light") → **GlossBERT** for word sense disambiguation
   - **Basic and stop words** (like "think", "people", "the") → **MiniLM** for semantic similarity
   - **No common words** → **MiniLM** for general similarity

## Files

- `mixed_wic_large_scale_evaluation.py` → Tests on both basic similarity + WiC homophone datasets (realistic results)
- `semantic_similarity_evaluation.py` → Tests on basic similarity only (with GlossBERT on/off flag)
- `glossbert_wsd_ranker.py` → Core hybrid system logic

## Usage

### Mixed Evaluation (Recommended)
```bash
python3 mixed_wic_large_scale_evaluation.py
```
Shows intelligent method selection with true/false examples.

### Pure Semantic Similarity
```bash
python3 semantic_similarity_evaluation.py
```
Compares Pure MiniLM vs Hybrid approaches on semantic similarity tasks.

**Option A: From Original Repository**
```bash
git clone https://github.com/HSLCY/GlossBERT.git
```

**Option B: Direct Download Links**
- GlossBERT Models: [Download from HuggingFace](https://huggingface.co/models?search=glossbert)
- WordNet Data: [Download from Princeton](https://wordnet.princeton.edu/download)