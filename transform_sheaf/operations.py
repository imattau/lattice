"""Fixed linguistic operation library.

Each operation is a deterministic function from a token sequence to
an OpResult. The outcome is APPLIES, BLOCKS, or CONTRADICTS.

These operations are HEURISTIC. They use a small verb lexicon and
structural rules, not a full grammar. A production system would use
a formal grammar (Displacement Calculus, Merge algebra). The
heuristics here are sufficient to demonstrate the pattern and are
documented so their limits are visible.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Outcome(str, Enum):
    APPLIES = "applies"
    BLOCKS = "blocks"
    CONTRADICTS = "contradicts"


@dataclass
class OpResult:
    operation: str
    outcome: Outcome
    detail: str = ""
    transformed: Optional[str] = None


# --- Lexicons (fixed before any input) ------------------------------------

VERBS = {
    "is", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "do", "does", "did",
    "need", "needs", "needed", "want", "wants", "wanted",
    "can", "could", "will", "would", "should", "must", "may", "might",
    "charge", "charged", "refund", "refunded",
    "work", "works", "worked", "crash", "crashes", "crashed",
    "log", "logged", "sign", "signed", "login", "logout",
    "pay", "paid", "update", "updated", "change", "changed",
    "delete", "deleted", "access", "accessed", "use", "used",
    "find", "found", "see", "saw", "get", "got", "give", "gave",
}

AUXILIARIES = {"is", "are", "was", "were", "has", "have", "had",
               "can", "could", "will", "would", "should", "must",
               "do", "does", "did", "may", "might"}

DETERMINERS = {"a", "an", "the", "my", "your", "his", "her",
               "our", "their", "its", "this", "that", "these",
               "those", "some", "any", "every", "all", "no"}

PRONOUNS = {"i", "me", "my", "we", "us", "our", "you", "your",
            "he", "him", "his", "she", "her", "it", "its",
            "they", "them", "their"}

WH_WORDS = {"what", "where", "when", "why", "how", "who", "which"}


def tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    out = []
    for tok in text.lower().replace("?", " ? ").replace(".", " . ").split():
        if tok.strip():
            out.append(tok)
    return out


def _first_verb_index(tokens: list[str]) -> int:
    for i, t in enumerate(tokens):
        if t in VERBS or t in AUXILIARIES:
            return i
    return -1


# --- The six operations ---------------------------------------------------

def op_negate(tokens: list[str]) -> OpResult:
    """Insert 'not' before the first auxiliary/verb.

    Applies if there is a verb. Blocks if not.
    Contradicts if the input already contains explicit negation.
    """
    if "not" in tokens or "n't" in tokens:
        return OpResult("negate", Outcome.CONTRADICTS,
                        detail="already negated")
    i = _first_verb_index(tokens)
    if i < 0:
        return OpResult("negate", Outcome.BLOCKS, detail="no verb")
    new = tokens[:i] + ["not"] + tokens[i:]
    return OpResult("negate", Outcome.APPLIES,
                    transformed=" ".join(new))


def op_passivize(tokens: list[str]) -> OpResult:
    """Rewrite NP V NP -> NP was Ved by NP.

    Requires a transitive structure. Blocks if there is no object.
    Contradicts if the sentence is already passive.
    """
    if "by" in tokens and any(t in AUXILIARIES for t in tokens):
        return OpResult("passivize", Outcome.CONTRADICTS,
                        detail="already passive")
    verbs = [i for i, t in enumerate(tokens) if t in VERBS]
    if not verbs:
        return OpResult("passivize", Outcome.BLOCKS, detail="no verb")
    v = verbs[0]
    if v >= len(tokens) - 2:
        return OpResult("passivize", Outcome.BLOCKS, detail="no object")
    return OpResult("passivize", Outcome.APPLIES,
                    transformed=" ".join(tokens) + " [passive]")


def op_embed(tokens: list[str]) -> OpResult:
    """Embed as a subordinate clause: 'the fact that X'.

    Applies for any declarative. Blocks for questions.
    """
    if tokens and tokens[-1] == "?":
        return OpResult("embed", Outcome.BLOCKS, detail="interrogative")
    return OpResult("embed", Outcome.APPLIES,
                    transformed="the fact that " + " ".join(tokens))


def op_quantify(tokens: list[str]) -> OpResult:
    """Replace the first determiner with a quantifier.

    Applies if there is a determiner. Blocks if not.
    """
    for i, t in enumerate(tokens):
        if t in DETERMINERS:
            new = tokens[:i] + ["every"] + tokens[i + 1:]
            return OpResult("quantify", Outcome.APPLIES,
                            transformed=" ".join(new))
    return OpResult("quantify", Outcome.BLOCKS, detail="no determiner")


def op_cleft(tokens: list[str]) -> OpResult:
    """Cleft: 'it is X that ...'.

    Requires a subject NP. Blocks if the sentence is interrogative.
    """
    if tokens and tokens[-1] == "?":
        return OpResult("cleft", Outcome.BLOCKS, detail="interrogative")
    if not tokens:
        return OpResult("cleft", Outcome.BLOCKS, detail="empty")
    return OpResult("cleft", Outcome.APPLIES,
                    transformed="it is " + tokens[0] + " that " +
                                " ".join(tokens[1:]))


def op_question(tokens: list[str]) -> OpResult:
    """Form a yes/no question by fronting the auxiliary.

    Applies if there is an auxiliary in first position.
    Contradicts if the input is already interrogative.
    """
    if tokens and tokens[-1] == "?":
        return OpResult("question", Outcome.CONTRADICTS,
                        detail="already interrogative")
    for i, t in enumerate(tokens[:3]):
        if t in AUXILIARIES:
            new = tokens[i:i + 1] + tokens[:i] + tokens[i + 1:]
            return OpResult("question", Outcome.APPLIES,
                            transformed=" ".join(new) + " ?")
    return OpResult("question", Outcome.BLOCKS, detail="no frontable aux")


OPERATIONS = [
    op_negate,
    op_passivize,
    op_embed,
    op_quantify,
    op_cleft,
    op_question,
]

OPERATION_NAMES = [op.__name__.replace("op_", "") for op in OPERATIONS]
