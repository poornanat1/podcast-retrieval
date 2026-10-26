"""Word error rate and named-entity error rate.

Both sides are normalized the same way before comparison: Unicode
compatibility form with accents removed, lower case, punctuation to
spaces. So "The Glenn Beck Program." against "the glenn beck program" is
0 errors, while "Glenn Back" is one substitution. The normalization is
deliberately simple and the same for every recognizer; it is not the
Whisper normalizer, which also spells out numbers and expands
contractions, so a WER here is comparable within this project only.

The named-entity error rate is the share of annotated entities (show,
host and publisher names, see ``ml.speech.entities``) that do not appear
word for word in the transcript. A near miss ("glenn back" for "glenn
beck") counts as an error, as it does for lexical search, and its
character error rate is reported so near misses can be told from
omissions.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


def normalize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).split()


def edit_ops(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    """Substitutions, deletions and insertions of the minimal edit path."""
    n, m = len(ref), len(hyp)
    # cost and the operation counts that reached it, per cell
    best = [[(j, (0, 0, j)) for j in range(m + 1)]]
    for i in range(1, n + 1):
        row = [(i, (0, i, 0))]
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                cost, ops = best[i - 1][j - 1]
                row.append((cost, ops))
                continue
            sub_cost, (s, d, ins) = best[i - 1][j - 1]
            candidates = [(sub_cost + 1, (s + 1, d, ins))]
            del_cost, (s, d, ins) = best[i - 1][j]
            candidates.append((del_cost + 1, (s, d + 1, ins)))
            ins_cost, (s, d, ins) = row[j - 1]
            candidates.append((ins_cost + 1, (s, d, ins + 1)))
            row.append(min(candidates))
        best.append(row)
    return best[n][m][1]


@dataclass(frozen=True)
class WordErrors:
    words: int
    substitutions: int
    deletions: int
    insertions: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> float:
        return self.errors / self.words if self.words else (1.0 if self.errors else 0.0)


def word_errors(reference: str, hypothesis: str) -> WordErrors:
    ref, hyp = normalize(reference), normalize(hypothesis)
    s, d, i = edit_ops(ref, hyp)
    return WordErrors(words=len(ref), substitutions=s, deletions=d, insertions=i)


def character_error_rate(reference: str, hypothesis: str) -> float:
    ref, hyp = list(" ".join(normalize(reference))), list(" ".join(normalize(hypothesis)))
    s, d, i = edit_ops(ref, hyp)
    return (s + d + i) / len(ref) if ref else (1.0 if hyp else 0.0)


@dataclass(frozen=True)
class EntityMatch:
    entity: str
    found: bool
    closest: str  # the transcript span nearest to the entity
    cer: float    # character error rate of that span against the entity


def entity_matches(entities: list[str], hypothesis: str) -> list[EntityMatch]:
    hyp = normalize(hypothesis)
    out = []
    for entity in entities:
        ent = normalize(entity)
        n = len(ent)
        spans = [" ".join(hyp[i:i + n]) for i in range(max(1, len(hyp) - n + 1))] or [""]
        target = " ".join(ent)
        if target in spans:
            out.append(EntityMatch(entity=target, found=True, closest=target, cer=0.0))
            continue
        closest = min(spans, key=lambda s: character_error_rate(target, s))
        out.append(EntityMatch(entity=target, found=False, closest=closest,
                               cer=character_error_rate(target, closest)))
    return out


def entity_error_rate(matches: list[EntityMatch]) -> float | None:
    if not matches:
        return None
    return sum(not m.found for m in matches) / len(matches)
