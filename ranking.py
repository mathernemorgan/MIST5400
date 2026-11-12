"""Relevance ranking for AIntern.

Scores every task against the student's stated skill goals, then orders the
open tasks. Two scorers:

  * Voyage AI embeddings (cosine similarity) when VOYAGE_API_KEY is set.
  * A TF-IDF keyword fallback with no extra dependency or key, used when Voyage
    is not configured or the call fails. The API response says which one ran.

Re-ranking is coverage-aware: every completed task "uses up" the skill goals it
covers, so goals the student has not practised yet get more weight.
"""

import logging
import math
import os
import re
from collections import Counter

import httpx

log = logging.getLogger("aintern.ranking")

VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"
VOYAGE_MODEL = os.getenv("VOYAGE_MODEL", "voyage-3.5")

# Mild nudge so foundations come before shipping work when relevance is close.
STAGE_FACTOR = {"Foundations": 1.15, "Apply": 1.0, "Ship": 0.9}

_STOP = set(
    "a an the and or of to in on for with by at from as is are be this that it its "
    "into your you our their".split()
)


def split_goals(skills: str) -> list[str]:
    goals = [g.strip() for g in re.split(r"[,;\n]+", skills) if g.strip()]
    return goals[:8]


def _tokens(text: str) -> list[str]:
    out = []
    for w in re.findall(r"[a-z0-9][a-z0-9+#.\-]*", text.lower()):
        w = w.strip(".-")
        if not w or w in _STOP:
            continue
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return out


def _tfidf_vectors(texts: list[str]) -> list[dict[str, float]]:
    docs = [Counter(_tokens(t)) for t in texts]
    n = len(docs)
    df: Counter = Counter()
    for d in docs:
        df.update(d.keys())
    idf = {w: math.log((1 + n) / (1 + c)) + 1 for w, c in df.items()}
    vecs = []
    for d in docs:
        v = {w: (1 + math.log(c)) * idf[w] for w, c in d.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        vecs.append({w: x / norm for w, x in v.items()})
    return vecs


def _cos_sparse(a: dict, b: dict) -> float:
    return sum(x * b.get(w, 0.0) for w, x in a.items())


def _cos_dense(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _voyage_embed(texts: list[str], input_type: str) -> list[list[float]] | None:
    key = os.getenv("VOYAGE_API_KEY")
    if not key:
        return None
    try:
        r = httpx.post(
            VOYAGE_URL,
            headers={"Authorization": f"Bearer {key}"},
            json={"input": texts, "model": VOYAGE_MODEL, "input_type": input_type},
            timeout=20,
        )
        r.raise_for_status()
        data = sorted(r.json()["data"], key=lambda d: d["index"])
        return [d["embedding"] for d in data]
    except Exception as exc:  # fall back rather than break the demo
        log.warning("Voyage embedding failed, using keyword fallback: %s", exc)
        return None


def similarity_matrix(task_texts: list[str], goals: list[str]) -> tuple[list[list[float]], str]:
    """Return sims[task][goal] and the scorer that produced them."""
    task_vecs = _voyage_embed(task_texts, "document")
    goal_vecs = _voyage_embed(goals, "query") if task_vecs else None
    if task_vecs and goal_vecs:
        return (
            [[_cos_dense(t, g) for g in goal_vecs] for t in task_vecs],
            f"embeddings ({VOYAGE_MODEL})",
        )
    vecs = _tfidf_vectors(task_texts + goals)
    tv, gv = vecs[: len(task_texts)], vecs[len(task_texts):]
    return [[_cos_sparse(t, g) for g in gv] for t in tv], "keyword fallback (TF-IDF)"


def task_text(t: dict) -> str:
    return f"{t.get('title', '')}. {t.get('objective', '')} {' '.join(t.get('skills', []))}"


def rank_tasks(tasks: list[dict], goals: list[str]) -> dict:
    """Score and prioritise tasks. A task dict may carry done=True."""
    if not goals:
        goals = ["general professional skills"]
    sims, mode = similarity_matrix([task_text(t) for t in tasks], goals)

    # Scale so the strongest task/goal pair is 1.0 ("relative match").
    peak = max((max(row) for row in sims if row), default=0.0) or 1.0
    sims = [[max(0.0, s) / peak for s in row] for row in sims]

    done_idx = [i for i, t in enumerate(tasks) if t.get("done")]
    cover = [sum(sims[i][g] for i in done_idx) for g in range(len(goals))]
    weight = [1.0 / (1.0 + c) for c in cover]
    totals = [sum(sims[i][g] for i in range(len(tasks))) or 1.0 for g in range(len(goals))]

    ranked = []
    for i, t in enumerate(tasks):
        best = max(range(len(goals)), key=lambda g: sims[i][g])
        priority = (
            sum(sims[i][g] * weight[g] for g in range(len(goals))) / len(goals)
        ) * STAGE_FACTOR.get(t.get("stage", ""), 1.0)
        ranked.append(
            {
                "id": t["id"],
                "relevance": round(sims[i][best] * 100),
                "best_goal": goals[best],
                "priority": round(priority, 4),
            }
        )
    coverage = [
        {"goal": goals[g], "pct": round(100 * cover[g] / totals[g])}
        for g in range(len(goals))
    ]
    return {"ranked": ranked, "mode": mode, "goal_coverage": coverage}
