import pandas as pd
import pandas.testing as pdt
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_patterns import make_tools as make_patterns_tools


def test_mine_patterns_tool_sample_fixture():
    session = SessionState()

    # 1. Load dataset using premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    # 2. Build global DTM
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Record global state for immutability assertions (deep copy of dataframe)
    orig_df_copy = session.dataframe.copy(deep=True)
    orig_matrix_copy = session.feature_matrix.copy()
    orig_names = list(session.feature_names)
    orig_labels = list(session.labels)
    orig_vectorizer = session.artifacts["count_vectorizer"]

    patterns_tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in patterns_tools if t.name == "mine_patterns_tool")

    # Helper function to check stored result, args, and artifact consistency across calls
    def verify_result_storage(res, expected_args, artifact_key):
        res_id = res["result_id"]
        assert res_id in session.results_store
        stored = session.results_store[res_id]
        assert stored["record"].args == expected_args
        assert artifact_key in session.artifacts
        assert session.artifacts[artifact_key] == res["patterns"]

    # --- Test 1: Parameter Validation Errors ---
    err_missing_minsup = mine_patterns_tool.invoke(
        {"category_name": "catA", "filtering_method": "tfidf", "algorithm": "fpgrowth"}
    )
    assert "error" in err_missing_minsup

    err_missing_k = mine_patterns_tool.invoke(
        {"category_name": "catA", "filtering_method": "tfidf", "algorithm": "topk"}
    )
    assert "error" in err_missing_k

    # --- Test 2: Known Answer #17 - FPGrowth with TF-IDF filtering ---
    # catA, min_sup=1
    args_catA_sup1 = {
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1,
        "k": None,
    }
    res_catA_sup1 = mine_patterns_tool.invoke(args_catA_sup1)
    assert "error" not in res_catA_sup1
    assert res_catA_sup1["surviving_terms_count"] == 1
    assert res_catA_sup1["patterns_count"] == 1
    assert res_catA_sup1["patterns"] == [{"pattern": ["alpha"], "support": 3}]
    art_key_catA_fpgrowth = "patterns_catA_tfidf_fpgrowth"
    verify_result_storage(res_catA_sup1, args_catA_sup1, art_key_catA_fpgrowth)

    # catA, min_sup=4 (0 patterns expected) - artifact must update to empty list
    args_catA_sup4 = {
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 4,
        "k": None,
    }
    res_catA_sup4 = mine_patterns_tool.invoke(args_catA_sup4)
    assert "error" not in res_catA_sup4
    assert res_catA_sup4["patterns_count"] == 0
    assert res_catA_sup4["patterns"] == []
    verify_result_storage(res_catA_sup4, args_catA_sup4, art_key_catA_fpgrowth)
    assert session.artifacts[art_key_catA_fpgrowth] == []

    # catB, min_sup=1
    args_catB_sup1 = {
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1,
        "k": None,
    }
    res_catB_sup1 = mine_patterns_tool.invoke(args_catB_sup1)
    assert "error" not in res_catB_sup1
    assert res_catB_sup1["surviving_terms_count"] == 1
    assert res_catB_sup1["patterns_count"] == 1
    assert res_catB_sup1["patterns"] == [{"pattern": ["gamma"], "support": 4}]
    verify_result_storage(
        res_catB_sup1, args_catB_sup1, "patterns_catB_tfidf_fpgrowth"
    )

    # --- Test 3: Known Answer #20 - Top-K algorithm ---
    args_catA_topk = {
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "min_sup": None,
        "k": 1,
    }
    res_catA_topk = mine_patterns_tool.invoke(args_catA_topk)
    assert "error" not in res_catA_topk
    assert res_catA_topk["patterns_count"] == 1
    assert res_catA_topk["patterns"] == [{"pattern": ["alpha"], "support": 3}]
    verify_result_storage(res_catA_topk, args_catA_topk, "patterns_catA_tfidf_topk")

    args_catB_topk = {
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "min_sup": None,
        "k": 1,
    }
    res_catB_topk = mine_patterns_tool.invoke(args_catB_topk)
    assert "error" not in res_catB_topk
    assert res_catB_topk["patterns_count"] == 1
    assert res_catB_topk["patterns"] == [{"pattern": ["gamma"], "support": 4}]
    verify_result_storage(res_catB_topk, args_catB_topk, "patterns_catB_tfidf_topk")

    # --- Test 4: Known Answer #20 - MaxFPGrowth algorithm ---
    args_catA_max = {
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1,
        "k": None,
    }
    res_catA_max = mine_patterns_tool.invoke(args_catA_max)
    assert "error" not in res_catA_max
    assert res_catA_max["patterns_count"] == 0
    assert res_catA_max["patterns"] == []
    verify_result_storage(
        res_catA_max, args_catA_max, "patterns_catA_tfidf_maxfpgrowth"
    )

    args_catB_max = {
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1,
        "k": None,
    }
    res_catB_max = mine_patterns_tool.invoke(args_catB_max)
    assert "error" not in res_catB_max
    assert res_catB_max["patterns_count"] == 0
    assert res_catB_max["patterns"] == []
    verify_result_storage(
        res_catB_max, args_catB_max, "patterns_catB_tfidf_maxfpgrowth"
    )

    # --- Test 5: Zero-Surviving Terms Filters ---
    args_var = {
        "category_name": "catA",
        "filtering_method": "variance",
        "algorithm": "fpgrowth",
        "min_sup": 1,
        "k": None,
    }
    res_var = mine_patterns_tool.invoke(args_var)
    assert "error" not in res_var
    assert res_var["surviving_terms_count"] == 0
    assert res_var["patterns_count"] == 0
    assert res_var["patterns"] == []
    assert "note" in res_var
    verify_result_storage(res_var, args_var, "patterns_catA_variance_fpgrowth")

    args_tf = {
        "category_name": "catA",
        "filtering_method": "term_frequency",
        "algorithm": "fpgrowth",
        "min_sup": 1,
        "k": None,
    }
    res_tf = mine_patterns_tool.invoke(args_tf)
    assert "error" not in res_tf
    assert res_tf["surviving_terms_count"] == 0
    assert res_tf["patterns_count"] == 0
    assert res_tf["patterns"] == []
    assert "note" in res_tf
    verify_result_storage(res_tf, args_tf, "patterns_catA_term_frequency_fpgrowth")

    # --- Test 6: Verify Strict Immutability of Global State ---
    pdt.assert_frame_equal(session.dataframe, orig_df_copy)
    assert (session.feature_matrix != orig_matrix_copy).nnz == 0
    assert session.feature_names == orig_names
    assert list(session.labels) == orig_labels
    assert session.artifacts["count_vectorizer"] is orig_vectorizer
