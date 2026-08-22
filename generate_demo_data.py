from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import csv
import json
import random
import textwrap


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SITE = DATA / "demo_site"


TOPICS = {
    "Information Retrieval": ["indexing", "ranking", "query processing", "relevance feedback", "BM25", "PageRank", "HITS", "search evaluation"],
    "Artificial Intelligence": ["machine learning", "transformer", "classification", "features", "inference", "optimization", "prediction", "knowledge graph"],
    "Healthcare Analytics": ["clinical notes", "patient journey", "triage", "diagnosis", "risk prediction", "medical search", "symptoms", "care pathway"],
    "Agritech and Farming": ["soil health", "crop yield", "weather signals", "drip irrigation", "sensors", "precision farming", "farm advisory", "harvest"],
    "Renewable Energy": ["solar forecasting", "wind farms", "grid balancing", "battery storage", "carbon reduction", "smart meters", "efficiency", "clean power"],
    "Finance Intelligence": ["portfolio", "risk management", "fraud detection", "market signals", "credit scoring", "compliance", "trading", "alternative data"],
}

SOURCES = [
    ("research-lab", "Research Lab"),
    ("industry-news", "Industry News"),
    ("knowledge-hub", "Knowledge Hub"),
]

QUERY_ROWS = [
    ("search ranking relevance feedback", "Information Retrieval"),
    ("machine learning classification pipeline", "Artificial Intelligence"),
    ("patient triage risk prediction", "Healthcare Analytics"),
    ("precision farming soil health sensors", "Agritech and Farming"),
    ("solar wind battery storage grid", "Renewable Energy"),
    ("fraud detection credit scoring market signals", "Finance Intelligence"),
]


def make_paragraph(topic: str, keywords: list[str], variant: int) -> str:
    base = [
        f"This {topic.lower()} briefing connects {keywords[0]} with {keywords[1]} in a practical workflow.",
        f"The article explains how {keywords[2]} influences {keywords[3]} and why careful indexing improves search quality.",
        f"A repeated emphasis on {keywords[4]} helps the system surface documents that are both relevant and comparable.",
        f"The example also shows how {keywords[5]} can be measured, ranked, and used for recommendation.",
        f"In the demo, {keywords[6]} and {keywords[7]} appear across titles, metadata, and body text to support retrieval experiments.",
    ]
    random.Random(variant).shuffle(base)
    return " ".join(base)


def make_doc(topic: str, topic_keywords: list[str], source_slug: str, source_name: str, index: int) -> dict:
    seed = f"{topic}-{source_slug}-{index}"
    rng = random.Random(seed)
    key_pool = topic_keywords[:]
    rng.shuffle(key_pool)
    title = f"{topic}: {key_pool[0].title()} for {source_name}"
    paragraphs = [make_paragraph(topic, key_pool, index * 3 + 1), make_paragraph(topic, key_pool[::-1], index * 3 + 2)]
    summary = f"A practical overview of {key_pool[0]}, {key_pool[1]} and {key_pool[2]} for {topic.lower()}."
    return {
        "topic": topic,
        "source_slug": source_slug,
        "source_name": source_name,
        "title": title,
        "summary": summary,
        "paragraphs": paragraphs,
        "keywords": ", ".join(topic_keywords[:6]),
    }


def build_dataset() -> list[dict]:
    rows: list[dict] = []
    day = date(2025, 1, 1)
    doc_counter = 1
    for topic, keywords in TOPICS.items():
        for source_index, (source_slug, source_name) in enumerate(SOURCES):
            for variant in range(2):
                doc = make_doc(topic, keywords, source_slug, source_name, variant + source_index)
                doc_id = f"DOC{doc_counter:03d}"
                slug = f"{topic.lower().replace(' ', '-')}-{source_slug}-{variant + 1}"
                url = f"{source_slug}/{slug}.html"
                rows.append(
                    {
                        "doc_id": doc_id,
                        "title": doc["title"],
                        "topic": topic,
                        "source": source_name,
                        "source_slug": source_slug,
                        "url": url,
                        "published_at": (day + timedelta(days=doc_counter)).isoformat(),
                        "author": f"Editorial Desk {source_name}",
                        "keywords": doc["keywords"],
                        "summary": doc["summary"],
                        "content": "\n\n".join(doc["paragraphs"]),
                        "outlink_urls": "",
                    }
                )
                doc_counter += 1
    all_urls = [row["url"] for row in rows]
    for row in rows:
        same_topic = [other["url"] for other in rows if other["topic"] == row["topic"] and other["doc_id"] != row["doc_id"]]
        cross_topic = random.Random(row["doc_id"]).sample(all_urls, k=3)
        row["outlink_urls"] = " | ".join((same_topic[:3] + cross_topic)[:6])
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_html_pages(rows: list[dict]) -> None:
    SITE.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict]] = {slug: [] for slug, _ in SOURCES}
    for row in rows:
        grouped[row["source_slug"]].append(row)

    all_urls = [row["url"] for row in rows]
    for slug, source_name in SOURCES:
        site_dir = SITE / slug
        site_dir.mkdir(parents=True, exist_ok=True)
        index_links = []
        for row in grouped[slug]:
            index_links.append(f'<li><a href="{Path(row["url"]).name}">{row["title"]}</a> - {row["topic"]}</li>')
        index_html = f"""
        <html>
          <head>
            <title>{source_name} Collection</title>
          </head>
          <body>
            <h1>{source_name}</h1>
            <p>This seed page links to the local demo corpus for the assignment crawler.</p>
            <ul>
              {''.join(index_links)}
            </ul>
          </body>
        </html>
        """
        (site_dir / "index.html").write_text(textwrap.dedent(index_html).strip(), encoding="utf-8")

    for row in rows:
        page_path = SITE / row["source_slug"] / Path(row["url"]).name
        topic_pages = [other for other in rows if other["topic"] == row["topic"] and other["doc_id"] != row["doc_id"]]
        related_links = "".join(f'<li><a href="../{Path(other["url"]).name}">{other["title"]}</a></li>' for other in topic_pages[:3])
        cross_topic = random.Random(row["doc_id"]).sample(all_urls, k=3)
        cross_links = "".join(f'<li><a href="../../{Path(link).name}">{Path(link).stem.replace("-", " ").title()}</a></li>' for link in cross_topic)
        html_page = f"""
        <html>
          <head>
            <title>{row['title']}</title>
          </head>
          <body>
            <article>
              <h1>{row['title']}</h1>
              <p class="meta">Source: {row['source']} | Topic: {row['topic']} | Published: {row['published_at']}</p>
              <p class="summary">{row['summary']}</p>
              <p>{row['content'].splitlines()[0]}</p>
              <p>{row['content'].splitlines()[-1]}</p>
              <p>Keywords: {row['keywords']}</p>
              <section>
                <h2>Related articles in the same topic</h2>
                <ul>{related_links}</ul>
              </section>
              <section>
                <h2>Cross-topic links</h2>
                <ul>{cross_links}</ul>
              </section>
            </article>
          </body>
        </html>
        """
        page_path.write_text(textwrap.dedent(html_page).strip(), encoding="utf-8")


def build_interactions(rows: list[dict]) -> list[dict]:
    interactions: list[dict] = []
    rng = random.Random(42)
    users = [f"U{idx:02d}" for idx in range(1, 13)]
    topics = list(TOPICS.keys())
    doc_by_topic: dict[str, list[dict]] = {}
    for row in rows:
        doc_by_topic.setdefault(row["topic"], []).append(row)
    for user in users:
        favorite = rng.choice(topics)
        for topic in topics:
            choices = doc_by_topic[topic]
            selected = rng.sample(choices, k=min(2, len(choices))) if topic == favorite else rng.sample(choices, k=1)
            for item in selected:
                interactions.append({"user_id": user, "doc_id": item["doc_id"], "rating": 1 if topic == favorite else 0.5, "event": "view"})
    return interactions


def main() -> None:
    rows = build_dataset()
    DATA.mkdir(parents=True, exist_ok=True)
    write_csv(DATA / "demo_corpus.csv", rows, ["doc_id", "title", "topic", "source", "source_slug", "url", "published_at", "author", "keywords", "summary", "content", "outlink_urls"])

    queries = [{"query": query, "topic": topic} for query, topic in QUERY_ROWS]
    write_csv(DATA / "demo_queries.csv", queries, ["query", "topic"])

    interactions = build_interactions(rows)
    write_csv(DATA / "demo_interactions.csv", interactions, ["user_id", "doc_id", "rating", "event"])

    seeds = [{"seed_url": (SITE / slug / "index.html").resolve().as_uri(), "label": slug} for slug, _ in SOURCES]
    write_csv(DATA / "demo_seeds.csv", seeds, ["seed_url", "label"])

    write_html_pages(rows)
    (DATA / "demo_manifest.json").write_text(json.dumps({"document_count": len(rows), "topics": list(TOPICS.keys()), "sources": [source_name for _, source_name in SOURCES], "seeds": seeds}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
