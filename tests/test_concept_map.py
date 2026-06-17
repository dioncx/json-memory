import pytest
from json_memory.concept_map import CONCEPT_MAP, expand_query_semantic, get_concept_category

def test_expand_query_semantic_known_tokens():
    tokens = {"who", "name"}
    expanded = expand_query_semantic(tokens)
    assert "who" in expanded
    assert "name" in expanded
    assert "identity" in expanded
    assert "user" in expanded

def test_expand_query_semantic_unknown_tokens():
    tokens = {"unknown", "words"}
    expanded = expand_query_semantic(tokens)
    assert expanded == {"unknown", "words"}

def test_expand_query_semantic_empty_set():
    assert expand_query_semantic(set()) == set()

def test_expand_query_semantic_mixed_tokens():
    tokens = {"who", "unknown"}
    expanded = expand_query_semantic(tokens)
    assert "unknown" in expanded
    assert "identity" in expanded
    assert "who" in expanded

def test_get_concept_category_known():
    assert get_concept_category("who") == "identity"
    assert get_concept_category("time") == "time"
    assert get_concept_category("where") == "location"
    assert get_concept_category("restart") == "action"
    assert get_concept_category("bot") == "trading"
    assert get_concept_category("repo") == "project"
    assert get_concept_category("error") == "system"
    assert get_concept_category("message") == "communication"

def test_get_concept_category_unknown():
    assert get_concept_category("unknown") is None

def test_get_concept_category_empty():
    assert get_concept_category("") is None

def test_concept_map_structure():
    # Make sure all keys are strings and values are lists of strings
    for k, v in CONCEPT_MAP.items():
        assert isinstance(k, str)
        assert isinstance(v, list)
        for item in v:
            assert isinstance(item, str)
