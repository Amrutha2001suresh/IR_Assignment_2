from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from difflib import SequenceMatcher
from hashlib import md5
from pathlib import Path
from typing import Iterable, Literal, Sequence
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.request import url2pathname
import html
import math
import re
import time

import networkx as nx
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

try:
    from nltk.stem import PorterStemmer
except Exception:  # pragma: no cover
    PorterStemmer = None


TOKEN_RE = re.compile(r"[a-zA-Z0-9']+")


@dataclass
class CrawlResult:
    documents: pd.DataFrame
    graph: nx.DiGraph
    stats: dict[str, int | float]


@dataclass
class IndexBundle:
    corpus: pd.DataFrame
    cleaned_texts: list[str]
    vectorizer: TfidfVectorizer
    matrix: any
    pagerank: dict[str, float]
    hits_authority: dict[str, float]
    hits_hub: dict[str, float]
    build_seconds: float
    strategy: str


def normalize_url(raw_url: str, base_url: str | None = None) -> str:
    absolute = urljoin(base_url, raw_url) if base_url else raw_url
    parsed = urlparse(absolute)
    if parsed.scheme in {"http", "https"}:
        netloc = parsed.netloc.lower()
        path = parsed.path.rstrip("/") or "/"
        cleaned = parsed._replace(scheme=parsed.scheme.lower(), netloc=netloc, path=path, fragment="")
        return urlunparse(cleaned)
    if parsed.scheme == "file":
        path = Path(url2pathname(parsed.path)).resolve()
        return path.as_uri()
    path = Path(absolute)
    return path.resolve().as_uri() if path.exists() else absolute


def fetch_html(url: str, timeout: int = 12) -> tuple[str, str]:
    parsed = urlparse(url)
    if parsed.scheme in {"http", "https"}:
        response = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 IR Demo"})
        response.raise_for_status()
        return response.text, response.url
    if parsed.scheme == "file":
        path = Path(url2pathname(parsed.path))
        return path.read_text(encoding="utf-8", errors="ignore"), path.as_uri()
    path = Path(url)
    if path.exists():
        return path.read_text(encoding="utf-8", errors="ignore"), path.resolve().as_uri()
    raise ValueError(f"Unsupported URL scheme: {url}")


def extract_text_and_links(html_text: str, base_url: str) -> tuple[str, list[str], str]:
    soup = BeautifulSoup(html_text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    parts: list[str] = []
    for element in soup.find_all(["p", "li", "article", "section", "h1", "h2", "h3", "h4", "title", "blockquote"]):
        text = element.get_text(" ", strip=True)
        if text:
            parts.append(text)

    links: list[str] = []
    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href")
        if not href or href.startswith("mailto:") or href.startswith("javascript:"):
            continue
        links.append(normalize_url(href, base_url=base_url))

    visible_text = html.unescape(re.sub(r"\s+", " ", " ".join(parts))).strip()
    return visible_text, links, title


def canonical_content(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def fingerprint(text: str) -> str:
    return md5(canonical_content(text).encode("utf-8")).hexdigest()


def maybe_duplicate(candidate: str, seen_texts: list[str], threshold: float = 0.95) -> bool:
    candidate_norm = canonical_content(candidate)
    for text in seen_texts:
        if SequenceMatcher(None, candidate_norm, text).ratio() >= threshold:
            return True
    return False


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def stem_tokens(tokens: Iterable[str]) -> list[str]:
    if PorterStemmer is None:
        return list(tokens)
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]


def preprocess_text(text: str, strategy: str = "basic") -> str:
    tokens = tokenize(text)
    if strategy in {"stopwords", "stemming", "full"}:
        tokens = [token for token in tokens if token not in ENGLISH_STOP_WORDS and len(token) > 1]
    if strategy in {"stemming", "full"}:
        tokens = stem_tokens(tokens)
    if strategy == "full":
        tokens = [token for token in tokens if not token.isdigit()]
    return " ".join(tokens)


def crawl_sources(
    seed_urls: Sequence[str],
    max_depth: int = 2,
    max_pages: int = 60,
    same_domain_only: bool = False,
) -> CrawlResult:
    start = time.perf_counter()
    queue = deque((normalize_url(url), 0) for url in seed_urls if str(url).strip())
    visited: set[str] = set()
    seen_hashes: set[str] = set()
    seen_texts: list[str] = []
    rows: list[dict] = []
    graph = nx.DiGraph()
    duplicate_urls = 0
    duplicate_docs = 0
    source_domains = {urlparse(normalize_url(url)).netloc for url in seed_urls if urlparse(normalize_url(url)).netloc}

    while queue and len(rows) < max_pages:
        current_url, depth = queue.popleft()
        if current_url in visited:
            duplicate_urls += 1
            continue
        visited.add(current_url)

        try:
            html_text, resolved_url = fetch_html(current_url)
        except Exception:
            continue

        current_url = normalize_url(resolved_url)
        if same_domain_only and source_domains:
            current_domain = urlparse(current_url).netloc
            if current_domain and current_domain not in source_domains:
                continue

        text, links, title = extract_text_and_links(html_text, current_url)
        if not text:
            continue

        topic_match = re.search(r"Topic:\s*([^|]+)", text, flags=re.IGNORECASE)
        topic = topic_match.group(1).strip() if topic_match else ""

        content_hash = fingerprint(text)
        if content_hash in seen_hashes or maybe_duplicate(text, seen_texts):
            duplicate_docs += 1
            continue

        seen_hashes.add(content_hash)
        seen_texts.append(canonical_content(text))
        graph.add_node(current_url)
        for link in links:
            graph.add_edge(current_url, link)
            if link not in visited and depth < max_depth:
                queue.append((link, depth + 1))

        parsed = urlparse(current_url)
        source = parsed.netloc or (Path(url2pathname(parsed.path)).parent.name if parsed.scheme == "file" else "local")
        rows.append(
            {
                "doc_id": f"doc_{len(rows) + 1:03d}",
                "title": title or Path(parsed.path).stem.replace("-", " ").title() or current_url,
                "url": current_url,
                "source_domain": parsed.netloc or parsed.path.split("/")[1] if parsed.scheme == "file" else parsed.netloc,
                "source": source,
                "topic": topic,
                "depth": depth,
                "content": text,
                "content_hash": content_hash,
                "word_count": len(tokenize(text)),
                "char_count": len(text),
                "outlinks": len(links),
                "outlink_urls": " | ".join(links),
                "fetched_at": pd.Timestamp.utcnow().isoformat(),
            }
        )

    documents = pd.DataFrame(rows)
    if not documents.empty:
        documents["doc_id"] = documents["doc_id"].astype(str)

    stats = {
        "seed_count": len(seed_urls),
        "visited_pages": len(visited),
        "unique_documents": len(documents),
        "duplicate_urls": duplicate_urls,
        "duplicate_documents": duplicate_docs,
        "crawl_seconds": round(time.perf_counter() - start, 3),
    }
    return CrawlResult(documents=documents, graph=graph, stats=stats)


def build_graph_scores(graph: nx.DiGraph) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        return {}, {}, {}
    pagerank = nx.pagerank(graph, alpha=0.85)
    try:
        hubs, authorities = nx.hits(graph, max_iter=1000, normalized=True)
    except Exception:
        return pagerank, {node: 0.0 for node in graph.nodes()}, {node: 0.0 for node in graph.nodes()}
    return pagerank, authorities, hubs


def normalize_scores(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    minimum = float(series.min())
    maximum = float(series.max())
    if math.isclose(minimum, maximum):
        return pd.Series(np.ones(len(series)), index=series.index, dtype=float)
    return (series - minimum) / (maximum - minimum)


def build_index(corpus: pd.DataFrame, strategy: str = "full") -> IndexBundle:
    start = time.perf_counter()
    working = corpus.copy().reset_index(drop=True)
    working["clean_text"] = working["content"].fillna("").map(lambda text: preprocess_text(str(text), strategy=strategy))
    vectorizer = TfidfVectorizer(
        lowercase=False,
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        ngram_range=(1, 2),
        min_df=1,
    )
    matrix = vectorizer.fit_transform(working["clean_text"])
    graph = nx.DiGraph()
    for _, row in working.iterrows():
        graph.add_node(row["url"])
    if "outlink_urls" in working.columns:
        for _, row in working.iterrows():
            source = row["url"]
            if pd.notna(row.get("outlink_urls")) and str(row["outlink_urls"]).strip():
                for target in str(row["outlink_urls"]).split(" | "):
                    if target:
                        graph.add_edge(source, target)
    pagerank, authorities, hubs = build_graph_scores(graph)
    return IndexBundle(
        corpus=working,
        cleaned_texts=working["clean_text"].tolist(),
        vectorizer=vectorizer,
        matrix=matrix,
        pagerank=pagerank,
        hits_authority=authorities,
        hits_hub=hubs,
        build_seconds=round(time.perf_counter() - start, 3),
        strategy=strategy,
    )


def extract_query_terms(query: str, strategy: str = "full") -> str:
    return preprocess_text(query, strategy=strategy)


def build_snippet(text: str, query: str, width: int = 220) -> str:
    lowered = text.lower()
    terms = [term for term in tokenize(query) if len(term) > 2]
    index = min([lowered.find(term) for term in terms if lowered.find(term) != -1], default=0)
    start = max(0, index - 80)
    snippet = text[start : start + width].strip()
    return re.sub(r"\s+", " ", snippet)


def score_documents(bundle: IndexBundle, query: str, ranking_mode: str = "Hybrid", top_k: int = 10) -> pd.DataFrame:
    query_clean = extract_query_terms(query, strategy=bundle.strategy)
    if not query_clean.strip():
        return pd.DataFrame(columns=["doc_id", "title", "topic", "url", "lexical_score", "pagerank", "authority", "hub", "final_score", "snippet"])

    query_vector = bundle.vectorizer.transform([query_clean])
    lexical = (bundle.matrix @ query_vector.T).toarray().ravel()
    lexical = pd.Series(lexical, index=bundle.corpus.index, dtype=float)
    lexical_norm = normalize_scores(lexical)

    pagerank = pd.Series([bundle.pagerank.get(url, 0.0) for url in bundle.corpus["url"]], index=bundle.corpus.index, dtype=float)
    authority = pd.Series([bundle.hits_authority.get(url, 0.0) for url in bundle.corpus["url"]], index=bundle.corpus.index, dtype=float)
    hub = pd.Series([bundle.hits_hub.get(url, 0.0) for url in bundle.corpus["url"]], index=bundle.corpus.index, dtype=float)
    pagerank_norm = normalize_scores(pagerank)
    authority_norm = normalize_scores(authority)
    hub_norm = normalize_scores(hub)

    if ranking_mode == "Lexical only":
        final = lexical_norm
    elif ranking_mode == "PageRank aware":
        final = 0.75 * lexical_norm + 0.25 * pagerank_norm
    elif ranking_mode == "HITS aware":
        final = 0.7 * lexical_norm + 0.2 * authority_norm + 0.1 * hub_norm
    else:
        final = 0.68 * lexical_norm + 0.18 * pagerank_norm + 0.14 * authority_norm

    topic_series = bundle.corpus["topic"] if "topic" in bundle.corpus.columns else pd.Series([""] * len(bundle.corpus), index=bundle.corpus.index)
    ranked = bundle.corpus.copy().assign(
        lexical_score=lexical_norm,
        pagerank=pagerank_norm,
        authority=authority_norm,
        hub=hub_norm,
        final_score=final,
    ).sort_values("final_score", ascending=False)
    ranked["snippet"] = ranked["content"].map(lambda text: build_snippet(text, query))
    ranked["topic"] = topic_series
    return ranked.head(top_k)[["doc_id", "title", "topic", "url", "lexical_score", "pagerank", "authority", "hub", "final_score", "snippet"]]


def build_similarity_matrix(bundle: IndexBundle) -> np.ndarray:
    dense = bundle.matrix @ bundle.matrix.T
    return dense.toarray() if hasattr(dense, "toarray") else np.asarray(dense)


def recommend_documents(
    bundle: IndexBundle,
    interactions: pd.DataFrame,
    doc_id: str,
    method: Literal["content-based", "collaborative", "hybrid"] = "hybrid",
    top_k: int = 5,
) -> pd.DataFrame:
    corpus = bundle.corpus.reset_index(drop=True)
    if doc_id not in set(corpus["doc_id"]):
        raise ValueError(f"Unknown document id: {doc_id}")

    doc_index = corpus.index[corpus["doc_id"] == doc_id][0]
    content_similarity = build_similarity_matrix(bundle)[doc_index]
    content_scores = normalize_scores(pd.Series(content_similarity, index=corpus.index, dtype=float))

    pivot = interactions.pivot_table(index="user_id", columns="doc_id", values="rating", fill_value=0.0)
    if doc_id not in pivot.columns:
        pivot[doc_id] = 0.0
    item_matrix = pivot.T
    collaborative_similarity = item_matrix @ item_matrix.T
    collaborative_scores = pd.Series(collaborative_similarity.loc[doc_id], dtype=float)
    collaborative_scores = collaborative_scores.reindex(corpus["doc_id"])
    collaborative_scores.index = corpus.index
    collaborative_scores = normalize_scores(collaborative_scores)

    if method == "content-based":
        scores = content_scores
    elif method == "collaborative":
        scores = collaborative_scores
    else:
        scores = 0.6 * content_scores + 0.4 * collaborative_scores

    scored = corpus.copy().assign(similarity_score=scores, rank_reason=method)
    scored = scored[scored["doc_id"] != doc_id].sort_values("similarity_score", ascending=False)
    if "topic" not in scored.columns:
        scored["topic"] = ""
    scored["explanation"] = scored.apply(
        lambda row: f"Topic: {row['topic']} | Source: {row['source']} | Score: {row['similarity_score']:.3f}",
        axis=1,
    )
    return scored.head(top_k)[["doc_id", "title", "topic", "similarity_score", "url", "explanation"]]


def build_relevance_sets(queries: pd.DataFrame, corpus: pd.DataFrame) -> dict[str, set[str]]:
    relevance: dict[str, set[str]] = {}
    for _, row in queries.iterrows():
        relevance[row["query"]] = set(corpus.loc[corpus["topic"] == row["topic"], "doc_id"].tolist())
    return relevance


def average_precision(retrieved: Sequence[str], relevant: set[str]) -> float:
    hits = 0
    precisions = []
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            hits += 1
            precisions.append(hits / rank)
    return float(np.mean(precisions)) if precisions else 0.0


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    gains = [1 if doc_id in relevant else 0 for doc_id in retrieved[:k]]
    dcg = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(gains))
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


def evaluate_ranker(bundle: IndexBundle, queries: pd.DataFrame, ranking_mode: str = "Hybrid", k: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    relevance_sets = build_relevance_sets(queries, bundle.corpus)
    for _, qrow in queries.iterrows():
        query = qrow["query"]
        relevant = relevance_sets[query]
        ranked = score_documents(bundle, query, ranking_mode=ranking_mode, top_k=max(k, len(bundle.corpus)))
        retrieved = ranked["doc_id"].tolist()
        retrieved_k = retrieved[:k]
        hits = [doc_id for doc_id in retrieved_k if doc_id in relevant]
        precision = len(hits) / k if k else 0.0
        recall = len(hits) / len(relevant) if relevant else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "query": query,
                "topic": qrow["topic"],
                "precision@k": precision,
                "recall@k": recall,
                "f1@k": f1,
                "MAP": average_precision(retrieved, relevant),
                "MRR": reciprocal_rank(retrieved, relevant),
                "NDCG@k": ndcg_at_k(retrieved, relevant, k),
                "relevant_docs": len(relevant),
                "retrieved_relevant_docs": len(hits),
            }
        )
    metrics = pd.DataFrame(rows)
    summary = pd.DataFrame(
        {
            "metric": ["precision@k", "recall@k", "f1@k", "MAP", "MRR", "NDCG@k"],
            "score": [metrics[c].mean() for c in ["precision@k", "recall@k", "f1@k", "MAP", "MRR", "NDCG@k"]],
        }
    )
    return metrics, summary


def topic_keywords(corpus: pd.DataFrame, strategy: str = "full", top_n: int = 10) -> pd.DataFrame:
    df = corpus.copy()
    df["clean_text"] = df["content"].map(lambda text: preprocess_text(str(text), strategy=strategy))
    rows = []
    for topic, group in df.groupby("topic"):
        counter = Counter()
        for text in group["clean_text"]:
            counter.update(text.split())
        for keyword, score in counter.most_common(top_n):
            rows.append({"topic": topic, "keyword": keyword, "frequency": score})
    return pd.DataFrame(rows)


def enrich_document_stats(corpus: pd.DataFrame) -> pd.DataFrame:
    profile = corpus.copy()
    if "word_count" not in profile.columns:
        profile["word_count"] = profile["content"].fillna("").map(lambda text: len(tokenize(str(text))))
    if "char_count" not in profile.columns:
        profile["char_count"] = profile["content"].fillna("").map(lambda text: len(str(text)))
    return profile


def corpus_profile(corpus: pd.DataFrame) -> pd.DataFrame:
    profile = enrich_document_stats(corpus)
    profile["sentence_count"] = profile["content"].fillna("").map(lambda text: max(1, len(re.split(r"[.!?]+", text))))
    profile["avg_token_length"] = profile["content"].fillna("").map(lambda text: np.mean([len(token) for token in tokenize(text)]) if tokenize(text) else 0.0)
    return profile[["doc_id", "title", "topic", "word_count", "char_count", "sentence_count", "avg_token_length"]]


def train_document_classifier(corpus: pd.DataFrame, strategy: str = "full", model_name: str = "logreg") -> dict[str, object]:
    df = corpus.copy().reset_index(drop=True)
    df["clean_text"] = df["content"].map(lambda text: preprocess_text(str(text), strategy=strategy))
    X_train, X_test, y_train, y_test = train_test_split(df["clean_text"], df["topic"], test_size=0.3, random_state=42, stratify=df["topic"])
    model = MultinomialNB() if model_name == "naive_bayes" else LogisticRegression(max_iter=2000)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=False, tokenizer=str.split, preprocessor=None, token_pattern=None, ngram_range=(1, 2))),
        ("model", model),
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    confusion = confusion_matrix(y_test, predictions, labels=sorted(df["topic"].unique()))
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "macro_f1": f1_score(y_test, predictions, average="macro", zero_division=0),
        "report": report,
        "confusion": confusion,
        "labels": sorted(df["topic"].unique()),
        "pipeline": pipeline,
        "sample_size": len(df),
    }


def comparison_table(corpus: pd.DataFrame) -> pd.DataFrame:
    strategies = ["basic", "stopwords", "stemming", "full"]
    rows = []
    for strategy in strategies:
        bundle = build_index(corpus, strategy=strategy)
        query = "information retrieval ranking search recommender"
        ranked = score_documents(bundle, query, ranking_mode="Hybrid", top_k=len(corpus))
        relevant = set(corpus.loc[corpus["topic"].str.contains("Information|AI", case=False, regex=True), "doc_id"])
        rows.append(
            {
                "strategy": strategy,
                "avg_query_score": float(ranked["final_score"].mean()) if not ranked.empty else 0.0,
                "MAP_proxy": average_precision(ranked["doc_id"].tolist(), relevant),
                "build_seconds": bundle.build_seconds,
                "vocabulary_size": len(bundle.vectorizer.vocabulary_),
            }
        )
    return pd.DataFrame(rows)
