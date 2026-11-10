# E-Commerce Search Learning-to-Rank Pipeline (`CatBoost YetiRank` + Cross-Encoder Reranking)

Two-stage product search ranking pipeline optimizing **`nDCG@10`** across query groups (`~49k` labeled training pairs, `~21k` test pairs, relevance grades `{0, 1, 2, 3}`), combining `GroupKFold` **`CatBoost` (`YetiRank`)** over lexical, dense-embedding, and cached cross-encoder features with optional fine-tuned **`MiniLM`** top-K reranking.

---

## 1. Pipeline Architecture

```text
Raw Query + Product Catalog (title, description, bullet_point, brand, color)
   |
   +--> Text Normalization (`src/preprocessing.py`: HTML stripping, ftfy, +/-160 char query-centric windowing)
   |
   +--> Multi-Signal Feature Engineering (`src/features.py`, disk-cached in `cache/`)
   |       |-- Lexical: BM25 (title / description / bullets / full), Jaccard, Levenshtein, RapidFuzz, numeric overlap
   |       |-- Dense Bi-Encoder: `intfloat/multilingual-e5-large-instruct` cosine similarities across fields
   |       |-- Cross-Encoder Feature: cached `BAAI/bge-reranker-v2-m3` query-document relevance logits
   |       '-- Out-of-Fold Target Encoding (`product_id`, `product_title`, `query`) + categorical brand/color
   |
   +--> Stage 1 Ranker (`src/train.py`): GroupKFold (`query_id`) CatBoost `YetiRank` Ensemble
   |
   '--> Stage 2 Blending & Optional Top-K Reranking (`src/ce_train.py`, `src/rerank.py`, `src/inference.py`)
           '-- Score fusion with cached CE features + fine-tuned `MiniLM` cross-encoder over top-K candidates
```

---

## 2. Core Modules

- **`src/preprocessing.py`**: Cleans broken HTML/entities (`BeautifulSoup`, `ftfy`), normalizes masked/null brand and color tokens, drops constant columns (`product_locale`), and extracts `+/-160`-character context windows centered on query matches inside long product descriptions.
- **`src/features.py`**: Computes multi-field `BM25`, token coverage, edit-distance ratios (`Levenshtein`, `RapidFuzz`), `intfloat/multilingual-e5-large-instruct` dense field similarities, cached `BAAI/bge-reranker-v2-m3` cross-encoder scores, and leakage-free out-of-fold target encodings.
- **`src/train.py`**: Trains a `GroupKFold` (`query_id`, `seed = 993`) `CatBoost` ensemble under the listwise/pairwise `YetiRank` objective (`catboost_fold*.cbm` + `metadata.json`).
- **`src/ce_train.py` & `src/rerank.py`**: Fine-tunes a lightweight `MiniLM` cross-encoder (`max_len = 160`) on query-product pairs and reranks the top-`K` candidates produced by Stage 1.
- **`src/inference.py` & `main.py`**: Unified CLI supporting standalone training, cross-encoder fine-tuning, score blending (`--ce-blend-weight`), top-K reranking (`--ce-rerank-top-k`), and full end-to-end submission generation.

---

## 3. Repository Structure

```text
AMML-ranking-competition/
|-- main.py                     # CLI entrypoint (train / ce-train / infer / rerank / full)
|-- src/
|   |-- preprocessing.py        # HTML stripping, normalization & query-window extraction
|   |-- features.py             # BM25, E5 dense, BGE-reranker-v2-m3 & OOF target encoding
|   |-- train.py                # GroupKFold CatBoost YetiRank training
|   |-- ce_train.py             # Cross-encoder fine-tuning
|   |-- rerank.py               # Top-K cross-encoder score fusion
|   '-- inference.py            # Ensemble prediction & submission writer
|-- notebooks/eda.ipynb         # Corpus & query-group exploratory analysis
|-- Dockerfile                  # Containerized reproducible run
'-- requirements.txt            # Pinned dependencies
```

---

## 4. Reproducible Usage

### Setup

```bash
git clone https://github.com/maxlyara1/AMML-ranking-competition.git
cd AMML-ranking-competition

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### End-to-End Training & Inference (`main.py`)

```bash
# Full pipeline (CatBoost YetiRank + optional CE stage + test inference):
python main.py --mode full \
  --train-path data/train.csv \
  --test-path data/test.csv \
  --models-dir models \
  --submission-path results/submission.csv

# Stage-1 CatBoost YetiRank training with explicit hyperparameters:
python main.py --mode train \
  --train-path data/train.csv \
  --models-dir models \
  --iterations 2000 \
  --learning-rate 0.03 \
  --depth 8

# Inference with top-K cross-encoder reranking:
python main.py --mode infer \
  --test-path data/test.csv \
  --models-dir models \
  --submission-path results/submission.csv \
  --ce-model-dir models_ce_minilm \
  --ce-rerank-top-k 200 \
  --ce-rerank-weight 0.4
```

### Docker

```bash
docker build -t ltr-search-ranking .
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/results:/app/results" \
  ltr-search-ranking
```
