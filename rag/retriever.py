"""rag/retriever.py — tiny TF-IDF retriever over the knowledge base."""
from __future__ import annotations

import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


_KB_DIR = Path(__file__).parent / "knowledge_base"
_HEADING_RE = re.compile(r"^#{1,6}\s+(.*)$")
_SUPERSCRIPT_MAP = str.maketrans({"²": "2", "³": "3", "¹": "1"})
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for",
    "how", "i", "in", "is", "it", "of", "on", "or", "tell", "that", "the",
    "this", "to", "us", "what", "when", "where", "which", "who", "why", "with",
}
# Words too generic to trust for the heading-match boost on their own — they
# show up in headings across many unrelated topics on this dashboard.
_GENERIC_BOOST_EXCLUDE = {
    "run", "dashboard", "model", "models", "data", "season", "seasons",
    "tab", "page", "show", "use", "used", "does",
}


def _normalize(text: str) -> str:
    """Fold Unicode superscripts (R² → R2) so "R2" and "R²" retrieve the same
    chunk regardless of which one the person happens to type."""
    return text.translate(_SUPERSCRIPT_MAP)


class KnowledgeBase:
    """Load every .md file under knowledge_base/ and index them for retrieval."""

    def __init__(self, kb_dir: Path | None = None):
        self.kb_dir = Path(kb_dir) if kb_dir else _KB_DIR
        self.documents: list[dict] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None
        self._load()

    def _load(self) -> None:
        self.documents = []
        if not self.kb_dir.exists():
            return
        for path in sorted(self.kb_dir.glob("*.md")):
            text = path.read_text(encoding="utf-8").strip()
            if not text:
                continue
            for heading, chunk in self._split_chunks(text):
                if len(chunk) < 40:
                    continue
                self.documents.append({
                    "source": path.name,
                    "title": path.stem.replace("_", " ").title(),
                    "heading": heading,
                    "text": chunk,
                })
        if self.documents:
            self._vectorizer = TfidfVectorizer(
                lowercase=True, stop_words="english", ngram_range=(1, 2),
            )
            self._matrix = self._vectorizer.fit_transform(
                [_normalize(d["text"]) for d in self.documents]
            )

    @staticmethod
    def _split_chunks(text: str) -> list[tuple[str, str]]:
        """One chunk per paragraph, each paired with its nearest heading.

        A doc with a single top-level heading and several unrelated
        paragraphs beneath it (a metrics glossary, an FAQ, a troubleshooting
        list) used to collapse into one giant chunk, which diluted term
        weights and made retrieval imprecise — "explain KGE" could easily
        lose to an unrelated chunk that just happened to share more common
        words. Handles both markdown styles: a heading on its own line
        followed by a blank line before its body, and a heading immediately
        followed by its body on the very next line (no blank line) — either
        way, each heading's own paragraph(s) become one chunk per paragraph,
        tagged with that heading. Multi-line markdown lists stay intact as
        one chunk since their items are separated by single newlines, not
        blank lines, so they still split exactly where a person would expect.
        Returns a list of (heading_text, chunk_text) tuples.
        """
        raw = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: list[tuple[str, str]] = []
        current_heading = ""
        for para in raw:
            first_line, _, rest = para.partition("\n")
            match = _HEADING_RE.match(first_line)
            if match:
                current_heading = match.group(1).strip()
                rest = rest.strip()
                if rest:
                    chunks.append((current_heading, f"{current_heading}\n\n{rest}"))
                continue
            chunk_text = f"{current_heading}\n\n{para}" if current_heading else para
            chunks.append((current_heading, chunk_text))
        return chunks

    def search(self, query: str, top_k: int = 4,
               min_score: float = 0.10) -> list[dict]:
        """Return the top-k most relevant chunks with cosine similarity
        above `min_score`. Returns [] when nothing is close enough.

        Cosine similarity alone struggles when two chunks share heavy
        vocabulary (e.g. RMSE and MAE both say "mean", "error", "lower is
        better"). A small, transparent boost is added when the query's
        significant words appear directly in a chunk's own heading — a
        query literally asking about "RMSE" should win against the chunk
        titled "RMSE" over one titled "MAE" that merely mentions RMSE in
        passing.
        """
        if not self.documents or self._vectorizer is None or self._matrix is None:
            return []
        q_vec = self._vectorizer.transform([_normalize(query)])
        sims = cosine_similarity(q_vec, self._matrix).ravel()

        query_tokens = {
            token for token in re.findall(r"[a-z0-9]+", _normalize(query).lower())
            if token not in _STOPWORDS and len(token) > 1
        }
        if query_tokens:
            for i, doc in enumerate(self.documents):
                heading_tokens = set(re.findall(r"[a-z0-9]+", _normalize(doc["heading"]).lower()))
                heading_tokens -= _GENERIC_BOOST_EXCLUDE
                if heading_tokens and query_tokens & heading_tokens:
                    overlap = len(query_tokens & heading_tokens) / len(heading_tokens)
                    sims[i] += 0.5 * overlap

        order = sims.argsort()[::-1][:top_k]
        out: list[dict] = []
        for i in order:
            if sims[i] < min_score:
                continue
            doc = dict(self.documents[i])
            doc["score"] = float(sims[i])
            out.append(doc)
        return out
