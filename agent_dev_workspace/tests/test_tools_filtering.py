import numpy as np
import pytest
from scipy.sparse import csr_matrix
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools


def test_variance_filter_tool_sample_fixture():
    session = SessionState()

    # Load dataset using premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    # Build DTM first
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Record and copy state before filtering for strict immutability checks
    orig_matrix_copy = session.feature_matrix.copy()
    orig_names = list(session.feature_names)

    # Initialize filtering tool
    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(
        t for t in filtering_tools if t.name == "variance_filter_tool"
    )

    # Test threshold = 0.15 (Known Answer #5)
    res_015 = variance_filter_tool.invoke({"threshold": 0.15})

    assert set(res_015["kept_terms"]) == {"alpha", "beta", "delta", "gamma"}
    assert res_015["removed_terms"] == ["always"]
    assert res_015["n_kept"] == 4
    assert res_015["n_removed"] == 1

    # Verify state remains completely unchanged after threshold 0.15 call
    assert (session.feature_matrix != orig_matrix_copy).nnz == 0
    assert session.feature_names == orig_names

    # Check full_report stored in results_store
    result_id = res_015["result_id"]
    stored_entry = session.results_store[result_id]
    full_report = stored_entry["full_report"]

    assert list(full_report.columns[:2]) == ["term", "variance"]
    # Check descending sort order
    variances_in_report = full_report["variance"].tolist()
    assert variances_in_report == sorted(variances_in_report, reverse=True)

    # Map terms to variance values and verify all 5 known variances from TEST_FIXTURE.md #5
    var_dict = dict(zip(full_report["term"], full_report["variance"]))
    expected_variances = {
        "alpha": 1.1875,
        "always": 0.109375,
        "beta": 0.234375,
        "delta": 0.25,
        "gamma": 1.109375,
    }

    for term, exp_var in expected_variances.items():
        assert var_dict[term] == pytest.approx(exp_var, abs=1e-4)

    # Test threshold = 0.25 (Verify >= boundary condition: delta has variance 0.25 and must be kept)
    res_025 = variance_filter_tool.invoke({"threshold": 0.25})
    assert "delta" in res_025["kept_terms"]
    assert "always" in res_025["removed_terms"]
    assert "beta" in res_025["removed_terms"]  # beta is 0.234375 < 0.25

    # Verify state remains completely unchanged after threshold 0.25 call
    assert (session.feature_matrix != orig_matrix_copy).nnz == 0
    assert session.feature_names == orig_names


def test_pearson_filter_tool_sample_fixture():
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

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    orig_matrix_copy = session.feature_matrix.copy()
    orig_names = list(session.feature_names)
    orig_labels = list(session.labels)

    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = next(
        t for t in filtering_tools if t.name == "pearson_filter_tool"
    )

    # Known Answer #6 for target_class="catB" and threshold=0.0
    res = pearson_filter_tool.invoke({"target_class": "catB", "threshold": 0.0})

    assert set(res["kept_terms"]) == {"delta", "gamma", "always"}
    assert set(res["removed_terms"]) == {"alpha", "beta"}

    # Immutability checks
    assert (session.feature_matrix != orig_matrix_copy).nnz == 0
    assert session.feature_names == orig_names
    assert list(session.labels) == orig_labels

    # Check full_report format and sorting
    stored_entry = session.results_store[res["result_id"]]
    full_report = stored_entry["full_report"]

    assert list(full_report.columns[:2]) == ["term", "pearson_r"]
    r_vals = full_report["pearson_r"].tolist()
    assert r_vals == sorted(r_vals, reverse=True)

    # Known Pearson r values from TEST_FIXTURE.md Known Answer 6 (tested with abs=1e-4)
    expected_pearson = {
        "alpha": -0.6882,
        "always": 0.3780,
        "beta": -0.7746,
        "delta": 1.0000,
        "gamma": 0.8307,
    }
    r_dict = dict(zip(full_report["term"], full_report["pearson_r"]))
    for term, exp_r in expected_pearson.items():
        assert r_dict[term] == pytest.approx(exp_r, abs=1e-4)


def test_spearman_filter_tool_sample_fixture():
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

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    orig_matrix_copy = session.feature_matrix.copy()
    orig_names = list(session.feature_names)
    orig_labels = list(session.labels)

    filtering_tools = make_filtering_tools(session)
    spearman_filter_tool = next(
        t for t in filtering_tools if t.name == "spearman_filter_tool"
    )

    # Known Answer #6 for target_class="catB" and threshold=0.0
    res = spearman_filter_tool.invoke({"target_class": "catB", "threshold": 0.0})

    assert set(res["kept_terms"]) == {"delta", "gamma", "always"}
    assert set(res["removed_terms"]) == {"alpha", "beta"}

    # Immutability checks
    assert (session.feature_matrix != orig_matrix_copy).nnz == 0
    assert session.feature_names == orig_names
    assert list(session.labels) == orig_labels

    # Check full_report format and sorting
    stored_entry = session.results_store[res["result_id"]]
    full_report = stored_entry["full_report"]

    assert list(full_report.columns[:2]) == ["term", "spearman_r"]
    r_vals = full_report["spearman_r"].tolist()
    assert r_vals == sorted(r_vals, reverse=True)

    # Known Spearman r values from TEST_FIXTURE.md Known Answer 6 (tested with abs=1e-4)
    expected_spearman = {
        "alpha": -0.7500,
        "always": 0.3780,
        "beta": -0.7746,
        "delta": 1.0000,
        "gamma": 0.9363,
    }
    r_dict = dict(zip(full_report["term"], full_report["spearman_r"]))
    for term, exp_r in expected_spearman.items():
        assert r_dict[term] == pytest.approx(exp_r, abs=1e-4)


def test_correlation_edge_cases():
    session = SessionState()
    filtering_tools = make_filtering_tools(session)
    pearson_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")
    spearman_tool = next(
        t for t in filtering_tools if t.name == "spearman_filter_tool"
    )

    # Setup 4 documents, 2 terms (1 varying, 1 constant column)
    session.feature_matrix = csr_matrix([[1, 5], [2, 5], [0, 5], [0, 5]])
    session.feature_names = ["var_term", "const_term"]
    session.labels = np.array(["catA", "catA", "catB", "catB"])

    # Test 1: Mismatched label length error for both tools
    session.labels = np.array(["catA", "catB"])  # length 2 vs row count 4
    assert "error" in pearson_tool.invoke({"target_class": "catA"})
    assert "error" in spearman_tool.invoke({"target_class": "catA"})

    # Reset valid labels
    session.labels = np.array(["catA", "catA", "catB", "catB"])

    # Test 2: Constant binary target errors for both tools (all zeros: non-existent class; all ones: matching target)
    assert "error" in pearson_tool.invoke({"target_class": "non_existent"})
    assert "error" in spearman_tool.invoke({"target_class": "non_existent"})

    session.labels = np.array(["catA", "catA", "catA", "catA"])  # all match catA (all ones)
    assert "error" in pearson_tool.invoke({"target_class": "catA"})
    assert "error" in spearman_tool.invoke({"target_class": "catA"})

    # Reset valid labels
    session.labels = np.array(["catA", "catA", "catB", "catB"])

    # Test 3: Constant feature column fallback to 0.0 for both tools (no NaN in result)
    res_pearson = pearson_tool.invoke({"target_class": "catA"})
    stored_p = session.results_store[res_pearson["result_id"]]["full_report"]
    p_dict = dict(zip(stored_p["term"], stored_p["pearson_r"]))
    assert p_dict["const_term"] == 0.0
    assert not np.isnan(p_dict["const_term"])

    res_spearman = spearman_tool.invoke({"target_class": "catA"})
    stored_s = session.results_store[res_spearman["result_id"]]["full_report"]
    s_dict = dict(zip(stored_s["term"], stored_s["spearman_r"]))
    assert s_dict["const_term"] == 0.0
    assert not np.isnan(s_dict["const_term"])
