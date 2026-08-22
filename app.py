from __future__ import annotations

from pathlib import Path
import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ir_core import (
    build_index,
    comparison_table,
    corpus_profile,
    crawl_sources,
    enrich_document_stats,
    evaluate_ranker,
    recommend_documents,
    score_documents,
    topic_keywords,
    train_document_classifier,
)


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CORPUS_PATH = DATA / "demo_corpus.csv"
QUERIES_PATH = DATA / "demo_queries.csv"
INTERACTIONS_PATH = DATA / "demo_interactions.csv"
SEEDS_PATH = DATA / "demo_seeds.csv"


st.set_page_config(page_title="IR End-to-End Demo", page_icon="search", layout="wide", initial_sidebar_state="expanded")


st.markdown(
    """
    <style>
    .stApp {
        background:
          radial-gradient(circle at top left, rgba(125, 211, 252, 0.14), transparent 30%),
          radial-gradient(circle at top right, rgba(251, 191, 36, 0.10), transparent 25%),
          linear-gradient(180deg, #09111f 0%, #0d1728 45%, #0b1220 100%);
        color: #e8eefc;
    }
    .hero {
        background: linear-gradient(135deg, rgba(22,34,58,0.95), rgba(11,18,32,0.92));
        border: 1px solid rgba(125, 211, 252, 0.22);
        border-radius: 24px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.25);
        margin-bottom: 1rem;
    }
    .hero h1 { color: #f8fbff; margin-bottom: 0.2rem; font-size: 2.1rem; }
    .hero p { color: #9fb0d0; margin-bottom: 0; }
    .section-title { color: #f8fbff; font-size: 1.2rem; margin-top: 0.2rem; margin-bottom: 0.75rem; }
    div[data-testid="stMetric"] { background: rgba(17,26,46,0.78); border: 1px solid rgba(125, 211, 252, 0.14); border-radius: 16px; padding: 0.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return load_csv(CORPUS_PATH), load_csv(QUERIES_PATH), load_csv(INTERACTIONS_PATH), load_csv(SEEDS_PATH)


@st.cache_resource(show_spinner=False)
def build_bundle(corpus: pd.DataFrame, strategy: str):
    return build_index(corpus, strategy=strategy)


def ensure_session_defaults() -> None:
    defaults = {"corpus": pd.DataFrame(), "bundle": None, "crawl_result": None, "metrics_history": [], "last_search": pd.DataFrame(), "last_recs": pd.DataFrame(), "last_eval": pd.DataFrame(), "last_summary": pd.DataFrame()}
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_demo_corpus() -> None:
    corpus, _, interactions, seeds = load_inputs()
    st.session_state.corpus = enrich_document_stats(corpus.copy())
    st.session_state.interactions = interactions.copy()
    st.session_state.seed_urls = seeds["seed_url"].tolist() if not seeds.empty else []
    st.session_state.bundle = build_bundle(st.session_state.corpus, strategy="full")


def human_seconds(seconds: float) -> str:
    return f"{seconds:.2f}s"


def render_overview(corpus: pd.DataFrame) -> None:
    st.markdown('<div class="hero"><h1>End-to-End Information Retrieval System</h1><p>Streamlit demo with crawling, indexing, ranked search, recommendations, preprocessing, classification, and evaluation.</p></div>', unsafe_allow_html=True)
    if corpus.empty:
        st.info("Load the demo corpus or run the crawler to begin.")
        return
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Documents", len(corpus))
    col2.metric("Topics", corpus["topic"].nunique())
    col3.metric("Sources", corpus["source"].nunique() if "source" in corpus else 0)
    avg_words = round(corpus["word_count"].mean(), 1) if "word_count" in corpus else round(corpus["content"].fillna("").map(lambda text: len(str(text).split())).mean(), 1)
    col4.metric("Avg words", avg_words)
    col5.metric("Duplicates removed", st.session_state.crawl_result.stats["duplicate_documents"] if st.session_state.crawl_result else 0)


def render_dashboard(corpus: pd.DataFrame) -> None:
    render_overview(corpus)
    if corpus.empty:
        return
    left, right = st.columns([1.3, 1])
    with left:
        st.markdown('<div class="section-title">Corpus composition</div>', unsafe_allow_html=True)
        topic_counts = corpus.groupby("topic").size().reset_index(name="documents")
        fig = px.bar(topic_counts, x="topic", y="documents", color="topic", title="Documents per topic")
        fig.update_layout(template="plotly_dark", showlegend=False, height=380, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown('<div class="section-title">Corpus profile</div>', unsafe_allow_html=True)
        st.dataframe(corpus_profile(corpus).head(10), use_container_width=True, hide_index=True)


def render_crawler() -> None:
    st.markdown('<div class="section-title">Crawling interface</div>', unsafe_allow_html=True)
    seeds_default = "\n".join(st.session_state.get("seed_urls", []))
    seeds_text = st.text_area("Seed URLs", value=seeds_default, height=120, help="Use local file URIs from the generated demo site or supply public URLs.")
    col1, col2, col3 = st.columns(3)
    with col1:
        depth = st.slider("Crawl depth", 0, 4, 2)
    with col2:
        max_pages = st.slider("Max pages", 10, 120, 60, step=5)
    with col3:
        same_domain = st.checkbox("Restrict to same seed domains", value=False)
    if st.button("Run crawler", type="primary"):
        seed_urls = [line.strip() for line in seeds_text.splitlines() if line.strip()]
        if not seed_urls:
            st.warning("Provide at least one seed URL.")
            return
        with st.spinner("Crawling and deduplicating pages..."):
            result = crawl_sources(seed_urls, max_depth=depth, max_pages=max_pages, same_domain_only=same_domain)
        st.session_state.crawl_result = result
        if not result.documents.empty:
            st.session_state.corpus = result.documents.copy()
            st.session_state.bundle = build_bundle(st.session_state.corpus, strategy="full")
        st.success("Crawl complete.")
        st.json(result.stats)
        if not result.documents.empty:
            st.dataframe(result.documents.head(15), use_container_width=True, hide_index=True)


def render_index_management() -> None:
    st.markdown('<div class="section-title">Index management</div>', unsafe_allow_html=True)
    if st.session_state.corpus.empty:
        st.info("No corpus loaded yet.")
        return
    strategy = st.selectbox("Preprocessing strategy", ["basic", "stopwords", "stemming", "full"], index=3)
    if st.button("Build / rebuild index"):
        with st.spinner("Building the TF-IDF index and graph scores..."):
            st.session_state.bundle = build_bundle(st.session_state.corpus, strategy=strategy)
        st.success(f"Index built in {st.session_state.bundle.build_seconds:.3f}s")
    bundle = st.session_state.bundle
    if bundle is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Vocabulary size", len(bundle.vectorizer.vocabulary_))
        c2.metric("Build time", human_seconds(bundle.build_seconds))
        c3.metric("Strategy", bundle.strategy)
        st.dataframe(bundle.corpus[["doc_id", "title", "topic", "source", "word_count"]].head(12), use_container_width=True, hide_index=True)


def render_search() -> None:
    st.markdown('<div class="section-title">Search interface</div>', unsafe_allow_html=True)
    if st.session_state.bundle is None:
        st.info("Build the index first.")
        return
    query = st.text_input("Query", value="ranking and relevance feedback for search")
    col1, col2 = st.columns([1, 1])
    with col1:
        ranking_mode = st.selectbox("Ranking mode", ["Hybrid", "Lexical only", "PageRank aware", "HITS aware"])
    with col2:
        top_k = st.slider("Top-K results", 3, 20, 8)
    if st.button("Search now", type="primary"):
        start = time.perf_counter()
        results = score_documents(st.session_state.bundle, query, ranking_mode=ranking_mode, top_k=top_k)
        latency = time.perf_counter() - start
        st.session_state.metrics_history.append({"action": "search", "seconds": latency})
        st.session_state.last_search = results.copy()
        st.success(f"Retrieved {len(results)} results in {latency:.3f}s")
        st.dataframe(results, use_container_width=True, hide_index=True)
        if not results.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=results["title"], y=results["lexical_score"], name="Lexical"))
            fig.add_trace(go.Bar(x=results["title"], y=results["pagerank"], name="PageRank"))
            fig.add_trace(go.Bar(x=results["title"], y=results["final_score"], name="Final"))
            fig.update_layout(barmode="group", template="plotly_dark", height=420, title="Ranking score decomposition")
            st.plotly_chart(fig, use_container_width=True)


def render_recommendations() -> None:
    st.markdown('<div class="section-title">Recommendation panel</div>', unsafe_allow_html=True)
    if st.session_state.bundle is None or st.session_state.corpus.empty:
        st.info("Load data and build the index first.")
        return
    interactions = st.session_state.get("interactions")
    if interactions is None or interactions.empty:
        interactions = load_csv(INTERACTIONS_PATH)
    doc_options = st.session_state.corpus[["doc_id", "title"]].apply(lambda row: f"{row['doc_id']} - {row['title']}", axis=1).tolist()
    chosen = st.selectbox("Choose a document", doc_options)
    doc_id = chosen.split(" - ", 1)[0]
    method = st.selectbox("Recommendation method", ["hybrid", "content-based", "collaborative"])
    top_k = st.slider("Recommendation K", 3, 10, 5)
    if st.button("Recommend now", type="primary"):
        recs = recommend_documents(st.session_state.bundle, interactions, doc_id=doc_id, method=method, top_k=top_k)
        st.session_state.last_recs = recs.copy()
        st.dataframe(recs, use_container_width=True, hide_index=True)
        if not recs.empty:
            fig = px.bar(recs, x="title", y="similarity_score", color="topic", title="Top-K recommendation scores")
            fig.update_layout(template="plotly_dark", height=380, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)


def render_preprocessing_and_mining() -> None:
    st.markdown('<div class="section-title">Text preprocessing and mining</div>', unsafe_allow_html=True)
    if st.session_state.corpus.empty:
        st.info("Load a corpus first.")
        return
    strategy = st.selectbox("Feature strategy", ["basic", "stopwords", "stemming", "full"], index=3, key="profile_strategy")
    keywords = topic_keywords(st.session_state.corpus, strategy=strategy, top_n=8)
    st.subheader("Top keywords by topic")
    fig = px.bar(keywords, x="keyword", y="frequency", color="topic", facet_col="topic", facet_col_wrap=2, title="Keyword distribution")
    fig.update_layout(template="plotly_dark", height=700, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Document classification")
    if st.button("Train document classifier"):
        with st.spinner("Training classifier and measuring hold-out performance..."):
            score = train_document_classifier(st.session_state.corpus, strategy=strategy, model_name="logreg")
        st.metric("Accuracy", f"{score['accuracy']:.3f}")
        st.metric("Macro F1", f"{score['macro_f1']:.3f}")
        st.json(score["report"])
        confusion = pd.DataFrame(score["confusion"], index=score["labels"], columns=score["labels"])
        st.dataframe(confusion, use_container_width=True)


def render_evaluation() -> None:
    st.markdown('<div class="section-title">Evaluation dashboard</div>', unsafe_allow_html=True)
    if st.session_state.bundle is None or st.session_state.corpus.empty:
        st.info("Build the index before evaluating.")
        return
    queries = load_csv(QUERIES_PATH)
    ranking_mode = st.selectbox("Evaluation ranking mode", ["Hybrid", "Lexical only", "PageRank aware", "HITS aware"], key="eval_mode")
    k = st.slider("Evaluation K", 3, 10, 5, key="eval_k")
    if st.button("Run evaluation"):
        metrics, summary = evaluate_ranker(st.session_state.bundle, queries, ranking_mode=ranking_mode, k=k)
        st.session_state.last_eval = metrics.copy()
        st.session_state.last_summary = summary.copy()
        st.dataframe(metrics, use_container_width=True, hide_index=True)
        st.dataframe(summary, use_container_width=True, hide_index=True)
        fig = px.bar(summary, x="metric", y="score", color="metric", title="Mean IR metrics")
        fig.update_layout(template="plotly_dark", showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)


def render_analytics() -> None:
    st.markdown('<div class="section-title">Performance analytics</div>', unsafe_allow_html=True)
    if st.session_state.corpus.empty:
        st.info("No corpus available.")
        return
    corpus = st.session_state.corpus
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Average word count", round(corpus["word_count"].mean(), 1))
        st.metric("Average character count", round(corpus["char_count"].mean(), 1))
    with col2:
        actions = pd.DataFrame(st.session_state.metrics_history) if st.session_state.metrics_history else pd.DataFrame(columns=["action", "seconds"])
        if not actions.empty:
            st.dataframe(actions, use_container_width=True, hide_index=True)
        else:
            st.caption("No runtime measurements captured yet.")
    fig = px.box(corpus, x="topic", y="word_count", color="topic", title="Word-count distribution by topic")
    fig.update_layout(template="plotly_dark", height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def render_comparative_analysis() -> None:
    st.markdown('<div class="section-title">Comparative preprocessing analysis</div>', unsafe_allow_html=True)
    if st.session_state.corpus.empty:
        st.info("Load a corpus first.")
        return
    if st.button("Compare preprocessing strategies"):
        comparison = comparison_table(st.session_state.corpus)
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        fig = px.bar(comparison, x="strategy", y=["MAP_proxy", "avg_query_score"], barmode="group", title="Strategy comparison")
        fig.update_layout(template="plotly_dark", height=420)
        st.plotly_chart(fig, use_container_width=True)


def render_discussion() -> None:
    st.markdown('<div class="section-title">Inference and discussion</div>', unsafe_allow_html=True)
    answers = [
        "1. Poor ranking with highly relevant documents usually comes from weak term weighting, insufficient query expansion, poor authority signals, or noisy duplicates. Improve by tuning lexical weights, using PageRank/HITS, adding query expansion, and validating with MAP/NDCG instead of only precision.",
        "2. Duplicate or near-duplicate documents inflate index size, distort term frequencies, create redundant recommendations, and can artificially boost evaluation metrics. Mitigate with URL canonicalisation, content hashing, similarity-based deduplication, and duplicate-aware evaluation splits.",
        "3. Content-based recommendation works best when document text is rich and user history is sparse. Collaborative recommendation is preferable when interaction logs are available and hidden preference patterns matter. Hybrid systems are stronger when both signals exist.",
        "4. Crawling supplies the corpus, preprocessing turns raw text into features, indexing enables fast retrieval, ranking orders the results, and recommendation extends retrieval into discovery. Together they create a closed-loop IR system that is measurable and adaptable.",
        "5. The main learning is that retrieval quality depends on the full pipeline, not only the ranker. Data quality, duplicate control, feature engineering, and evaluation choices materially change the final results.",
    ]
    for answer in answers:
        st.write(answer)


def sidebar_controls() -> None:
    with st.sidebar:
        st.title("IR Demo Controls")
        if st.button("Load generated demo corpus"):
            load_demo_corpus()
            st.success("Demo corpus loaded.")
        st.caption("Use the buttons in each tab to run the pipeline end-to-end.")
        if st.session_state.corpus.empty:
            st.warning("No corpus loaded yet.")
        else:
            st.success(f"Loaded {len(st.session_state.corpus)} documents")
            st.caption(f"Topics: {st.session_state.corpus['topic'].nunique()}")


def main() -> None:
    ensure_session_defaults()
    sidebar_controls()
    tabs = st.tabs(["Dashboard", "Crawling", "Index management", "Search", "Recommendations", "Text mining", "Evaluation", "Analytics", "Comparative analysis", "Discussion"])
    with tabs[0]:
        render_dashboard(st.session_state.corpus)
    with tabs[1]:
        render_crawler()
    with tabs[2]:
        render_index_management()
    with tabs[3]:
        render_search()
    with tabs[4]:
        render_recommendations()
    with tabs[5]:
        render_preprocessing_and_mining()
    with tabs[6]:
        render_evaluation()
    with tabs[7]:
        render_analytics()
    with tabs[8]:
        render_comparative_analysis()
    with tabs[9]:
        render_discussion()


if __name__ == "__main__":
    main()
