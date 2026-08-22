# IR Assignment 2 Demo

This folder contains a complete end-to-end Streamlit implementation for the Information Retrieval assignment.

## What is included

- Streamlit dashboard, crawler, index manager, ranked search, recommendation panel, evaluation dashboard, analytics, and discussion answers
- Generated demo corpus with local HTML seed pages so the crawler can run offline
- Query set and synthetic interaction log for evaluation and collaborative recommendation

## Dataset

The generated dataset lives in `data/`:

- `demo_corpus.csv` - document collection with metadata and content
- `demo_queries.csv` - evaluation queries with topic labels
- `demo_interactions.csv` - synthetic user-document interactions for collaborative recommendation
- `demo_seeds.csv` - local file URLs for the demo crawler
- `demo_site/` - local HTML pages used by the crawler

## Install

```bash
pip install -r requirements.txt
```

## Generate the demo data

```bash
python generate_demo_data.py
```

## Run the app

```bash
streamlit run app.py
```

## Notes

- The crawler supports both `http(s)` URLs and local `file://` seed pages.
- Ranking supports lexical, PageRank-aware, and HITS-aware scoring.
- Recommendation supports content-based, collaborative, and hybrid modes.
- Evaluation reports Precision, Recall, F1, Precision@K, Recall@K, MAP, MRR, and NDCG.
