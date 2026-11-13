import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools as make_data_tools
from tools_dtm import make_tools as make_dtm_tools


def _load_fixture_session():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )
    return session


def test_tokenize_tool_sample_fixture():
    session = _load_fixture_session()
    data_tools = make_data_tools(session)
    tokenize_tool = next(t for t in data_tools if t.name == "tokenize_tool")

    res = tokenize_tool.invoke({})
    assert "error" not in res
    assert res["n_documents"] == 8

    # Known Answer #10 exact token lists
    expected_tokens = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        [],
    ]

    tokens_in_df = session.dataframe["unigrams"].tolist()
    assert tokens_in_df == expected_tokens
    assert session.artifacts["unigrams"] == expected_tokens

    # Verify result storage
    res_id = res["result_id"]
    assert res_id in session.results_store
    assert session.results_store[res_id]["record"].args == {}


def test_list_files_tool():
    session = SessionState()
    data_tools = make_data_tools(session)
    list_files_tool = next(t for t in data_tools if t.name == "list_files_tool")

    # Known Answer #11 on "newdataset"
    res = list_files_tool.invoke({"subdirectory": "newdataset"})
    assert "error" not in res
    assert res["directories"] == []
    assert res["files"] == ["Reddit-stock-sentiment.csv"]

    res_id = res["result_id"]
    assert res_id in session.results_store
    assert session.results_store[res_id]["record"].args == {"subdirectory": "newdataset"}


def test_check_missing_tool_sample_fixture():
    session = _load_fixture_session()
    data_tools = make_data_tools(session)
    check_missing_tool = next(t for t in data_tools if t.name == "check_missing_tool")

    # Known Answer #2
    res = check_missing_tool.invoke({})
    assert "error" not in res
    assert res["missing_count"] == 1
    assert res["missing_indices"] == [7]

    res_id = res["result_id"]
    assert res_id in session.results_store


def test_check_duplicates_tool_sample_fixture():
    session = _load_fixture_session()
    data_tools = make_data_tools(session)
    check_duplicates_tool = next(t for t in data_tools if t.name == "check_duplicates_tool")
    tokenize_tool = next(t for t in data_tools if t.name == "tokenize_tool")

    # Build DTM to populate downstream state
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Tokenize first (adds list-valued 'unigrams' column)
    tokenize_tool.invoke({})

    # Known Answer #3 - duplicate detection count=1, index=[6]
    res_check = check_duplicates_tool.invoke({"drop": False})
    assert "error" not in res_check
    assert res_check["duplicate_count"] == 1
    assert res_check["dropped_count"] == 0
    assert res_check["duplicate_indices"] == [6]
    assert len(session.dataframe) == 8

    # Seed row-dependent artifacts & pending_figure prior to drop
    session.artifacts["patterns_test"] = [{"pattern": ["alpha"], "support": 3}]
    session.pending_figure = plt.figure()

    # Save existing results_store entries to verify preservation
    prior_stored_entries = dict(session.results_store)

    # Known Answer #3 - drop=True removes ALL copies (keep=False), leaving 6 rows (dropped 2)
    res_drop = check_duplicates_tool.invoke({"drop": True})
    assert "error" not in res_drop
    assert res_drop["duplicate_count"] == 1
    assert res_drop["dropped_count"] == 2
    assert res_drop["remaining_rows"] == 6
    assert len(session.dataframe) == 6

    # Verify downstream state invalidation
    assert session.feature_matrix is None
    assert session.feature_names is None
    assert session.pending_figure is None
    assert "unigrams" not in session.artifacts
    assert "count_vectorizer" not in session.artifacts
    assert "patterns_test" not in session.artifacts

    # Verify labels match remaining dataframe's category names in exact row order
    expected_cat_col = "category_name" if "category_name" in session.dataframe.columns else "label"
    assert list(session.labels) == session.dataframe[expected_cat_col].tolist()
    assert set(session.categories) == {"catA", "catB"}

    # Verify prior results_store entries are strictly preserved
    for res_id, entry in prior_stored_entries.items():
        assert res_id in session.results_store
        assert session.results_store[res_id] == entry


def test_sample_data_tool_sample_fixture():
    session = _load_fixture_session()
    data_tools = make_data_tools(session)
    sample_data_tool = next(t for t in data_tools if t.name == "sample_data_tool")
    tokenize_tool = next(t for t in data_tools if t.name == "tokenize_tool")

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})
    tokenize_tool.invoke({})

    # Seed row-dependent artifacts & pending_figure prior to sampling
    session.artifacts["patterns_sample_test"] = [{"pattern": ["gamma"], "support": 4}]
    session.pending_figure = plt.figure()

    # Save existing results_store entries to verify preservation
    prior_stored_entries = dict(session.results_store)

    # Known Answer #13: n=4, random_state=42 returns exactly indices [1, 5, 0, 7]
    res = sample_data_tool.invoke({"n": 4, "random_state": 42})
    assert "error" not in res
    assert res["n_sampled"] == 4
    assert res["sampled_indices"] == [1, 5, 0, 7]
    assert session.dataframe.index.tolist() == [1, 5, 0, 7]

    # Verify downstream state invalidation
    assert session.feature_matrix is None
    assert session.feature_names is None
    assert session.pending_figure is None
    assert "unigrams" not in session.artifacts
    assert "count_vectorizer" not in session.artifacts
    assert "patterns_sample_test" not in session.artifacts

    # Verify labels match remaining dataframe's category names in exact row order
    expected_cat_col = "category_name" if "category_name" in session.dataframe.columns else "label"
    assert list(session.labels) == session.dataframe[expected_cat_col].tolist()
    assert set(session.categories) == {"catA", "catB"}

    # Verify prior results_store entries are strictly preserved
    for res_id, entry in prior_stored_entries.items():
        assert res_id in session.results_store
        assert session.results_store[res_id] == entry


def test_describe_data_tool_sample_fixture():
    session = _load_fixture_session()
    data_tools = make_data_tools(session)
    describe_data_tool = next(t for t in data_tools if t.name == "describe_data_tool")

    # Known Answer #18
    res = describe_data_tool.invoke({})
    assert "error" not in res

    # Verify overall stats
    overall = res["overall_stats"]
    assert overall["count"] == 8
    assert overall["mean"] == pytest.approx(19.875, abs=1e-3)
    assert overall["std"] == pytest.approx(9.4330, abs=1e-3)
    assert overall["min"] == 0
    assert overall["25%"] == pytest.approx(17.75, abs=1e-3)
    assert overall["50%"] == pytest.approx(20.5, abs=1e-3)
    assert overall["75%"] == pytest.approx(25.25, abs=1e-3)
    assert overall["max"] == 30

    # Verify per-category stats
    per_cat = res["per_category_stats"]
    assert "catA" in per_cat and "catB" in per_cat
    assert per_cat["catA"]["mean"] == pytest.approx(17.25, abs=1e-3)
    assert per_cat["catB"]["mean"] == pytest.approx(22.5, abs=1e-3)

    # Verify boxplot figure
    assert session.pending_figure is not None
    assert isinstance(session.pending_figure, plt.Figure)
    plt.close(session.pending_figure)

    # Verify strict JSON serialization safety
    json_str = json.dumps(res, allow_nan=False)
    assert isinstance(json_str, str)


def test_describe_data_tool_one_row_json_safety():
    session = SessionState()
    # Create single row dataframe (where std dev is NaN)
    session.dataframe = pd.DataFrame({"text": ["single doc"], "category_name": ["catA"]})
    session.set_labels(session.dataframe["category_name"].values)

    data_tools = make_data_tools(session)
    describe_data_tool = next(t for t in data_tools if t.name == "describe_data_tool")

    res = describe_data_tool.invoke({})
    assert "error" not in res
    assert res["overall_stats"]["std"] is None  # NaN converted to None

    # Must pass strict JSON serialization without NaN
    json_str = json.dumps(res, allow_nan=False)
    assert isinstance(json_str, str)

    if session.pending_figure is not None:
        plt.close(session.pending_figure)


def test_inspect_data_tool_cases():
    session = _load_fixture_session()
    data_tools = make_data_tools(session)
    inspect_data_tool = next(t for t in data_tools if t.name == "inspect_data_tool")
    tokenize_tool = next(t for t in data_tools if t.name == "tokenize_tool")

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    orig_df_len = len(session.dataframe)
    orig_matrix_copy = session.feature_matrix.copy()
    orig_names = list(session.feature_names)
    orig_labels = list(session.labels)

    # --- Test 1: Default call (n=5) ---
    res_default = inspect_data_tool.invoke({})
    assert "error" not in res_default
    assert res_default["total_rows"] == 8
    assert res_default["n_rows"] == 5
    assert res_default["row_indices"] == [0, 1, 2, 3, 4]
    assert len(res_default["data"]) == 5

    # Check result storage and args
    res_id = res_default["result_id"]
    assert res_id in session.results_store
    assert session.results_store[res_id]["record"].args == {"n": 5}

    # --- Test 2: Oversized n (n=20) ---
    res_over = inspect_data_tool.invoke({"n": 20})
    assert "error" not in res_over
    assert res_over["n_rows"] == 8
    assert res_over["row_indices"] == list(range(8))

    # --- Test 3: Invalid n <= 0 error ---
    err_zero = inspect_data_tool.invoke({"n": 0})
    assert "error" in err_zero

    # --- Test 4: Inspection after tokenization (list-valued 'unigrams' column) ---
    tokenize_tool.invoke({})
    res_tok = inspect_data_tool.invoke({"n": 5})
    assert "error" not in res_tok
    assert "unigrams" in res_tok["columns"]
    # Check that list-valued unigrams serialized cleanly without truth-value errors
    assert res_tok["data"][0]["unigrams"] == ["always", "alpha", "alpha", "alpha", "beta"]

    # --- Test 5: Boolean, float, and NaN/None preservation in custom dataframe ---
    session_custom = SessionState()
    session_custom.dataframe = pd.DataFrame(
        {
            "is_valid": [True, False, True],
            "score": [98.5, np.nan, float("inf")],
            "label": ["A", "B", "C"],
        }
    )
    inspect_custom = next(t for t in make_data_tools(session_custom) if t.name == "inspect_data_tool")
    res_custom = inspect_custom.invoke({"n": 3})
    assert "error" not in res_custom

    rec0 = res_custom["data"][0]
    rec1 = res_custom["data"][1]
    rec2 = res_custom["data"][2]

    # Booleans preserved
    assert rec0["is_valid"] is True
    assert rec1["is_valid"] is False

    # Finite floats unrounded; NaN / Inf mapped to None
    assert rec0["score"] == 98.5
    assert rec1["score"] is None
    assert rec2["score"] is None

    # Strict JSON serialization safety
    json_str = json.dumps(res_custom, allow_nan=False)
    assert isinstance(json_str, str)

    # --- Test 6: Verify State Immutability ---
    assert len(session.dataframe) == orig_df_len
    assert (session.feature_matrix != orig_matrix_copy).nnz == 0
    assert session.feature_names == orig_names
    assert list(session.labels) == orig_labels
