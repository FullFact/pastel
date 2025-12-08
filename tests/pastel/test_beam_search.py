from unittest.mock import Mock, patch

from pastel.models import BiasType
from pastel.pastel import Pastel
from training.beam_search import add_one, run_beam_search


def test_add_one():
    current_set = frozenset(["a", "b"])
    all_features = ["a", "b", "c", "d"]
    new_sets = add_one(current_set, all_features)
    print(new_sets)
    assert len(new_sets) == 2
    assert all("a" in s and "b" in s for s in new_sets)
    assert sum("c" in s for s in new_sets) == 1
    assert sum("d" in s for s in new_sets) == 1


def test_add_one_empty_set():
    current_set = frozenset()
    all_features = ["a", "b"]
    new_sets = add_one(current_set, all_features)
    assert len(new_sets) == 2
    assert any("a" in s for s in new_sets)
    assert any("b" in s for s in new_sets)


def test_beam_search():
    # Very weak test! But confirms it returns the right shaped response
    all_features = ["a", "b", "c", "d"]
    with patch(
        "training.beam_search.evaluate_pastel_set", new_callable=Mock
    ) as mock_eval:
        pastel_model = Pastel(
            {
                BiasType.BIAS: 1.0,
                "a": -3.0,
                "b": 2.0,
            }
        )
        mock_eval.return_value = ({"f1": 4.0}, pastel_model)
        best_model, best_f1 = run_beam_search(all_features, beta=3, max_iter=3)
        assert best_model == pastel_model
        assert best_f1 == 4.0
