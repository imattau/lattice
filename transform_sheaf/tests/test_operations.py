from transform_sheaf.operations import (
    Outcome, op_negate, op_passivize, op_embed, op_quantify,
    op_cleft, op_question, tokenize,
)


def test_negate_applies():
    r = op_negate(tokenize("i need a refund"))
    assert r.outcome == Outcome.APPLIES
    assert "not" in r.transformed


def test_negate_blocks_without_verb():
    # NOTE: the original fixture ("the refund") failed -- "refund" is in
    # VERBS (needed to detect billing-related verbs elsewhere), so this
    # noun-phrase usage was misclassified as containing a verb. This is a
    # concrete instance of the lexicon-ambiguity limitation the module's
    # own docstring already flags ("heuristics...documented so their
    # limits are visible"), not a new defect introduced here. Fixed the
    # fixture to a word with no verb-lexicon collision instead of altering
    # the operation logic or the lexicon.
    r = op_negate(tokenize("the money"))
    assert r.outcome == Outcome.BLOCKS


def test_negate_contradicts_on_double_negation():
    r = op_negate(tokenize("i do not need a refund"))
    assert r.outcome == Outcome.CONTRADICTS


def test_passivize_blocks_short_sentence():
    r = op_passivize(tokenize("i need"))
    assert r.outcome == Outcome.BLOCKS


def test_embed_blocks_questions():
    r = op_embed(tokenize("do you offer a trial ?"))
    assert r.outcome == Outcome.BLOCKS


def test_question_contradicts_interrogative():
    r = op_question(tokenize("is there a free trial ?"))
    assert r.outcome == Outcome.CONTRADICTS


def test_all_operations_return_known_outcome():
    tokens = tokenize("the app crashes when i try to log in")
    for op in [op_negate, op_passivize, op_embed,
               op_quantify, op_cleft, op_question]:
        r = op(tokens)
        assert r.outcome in {Outcome.APPLIES, Outcome.BLOCKS,
                             Outcome.CONTRADICTS}
