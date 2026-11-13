import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_exploration import make_tools as make_exploration_tools


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
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})
    return session


def test_cosine_similarity_tool_sample_fixture():
    session = _load_fixture_session()
    exploration_tools = make_exploration_tools(session)
    cosine_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")

    orig_matrix_copy = session.feature_matrix.copy()
    orig_names = list(session.feature_names)
    orig_labels = list(session.labels)

    # Known Answer #16 test pairs (tested with abs=1e-4)
    # (0, 1): 0.9847
    res_01 = cosine_tool.invoke({"doc_index1": 0, "doc_index2": 1})
    assert "error" not in res_01
    assert res_01["similarity"] == pytest.approx(0.9847, abs=1e-4)

    # (0, 3): 0.0909
    res_03 = cosine_tool.invoke({"doc_index1": 0, "doc_index2": 3})
    assert "error" not in res_03
    assert res_03["similarity"] == pytest.approx(0.0909, abs=1e-4)

    # (3, 4): 0.9847
    res_34 = cosine_tool.invoke({"doc_index1": 3, "doc_index2": 4})
    assert "error" not in res_34
    assert res_34["similarity"] == pytest.approx(0.9847, abs=1e-4)

    # (0, 7) zero-vector doc: 0.0
    res_07 = cosine_tool.invoke({"doc_index1": 0, "doc_index2": 7})
    assert "error" not in res_07
    assert res_07["similarity"] == pytest.approx(0.0, abs=1e-4)

    # Verify result storage and args
    res_id = res_01["result_id"]
    assert res_id in session.results_store
    assert session.results_store[res_id]["record"].args == {"doc_index1": 0, "doc_index2": 1}

    # Verify invalid document index validations
    err_out1 = cosine_tool.invoke({"doc_index1": -1, "doc_index2": 0})
    assert "error" in err_out1

    err_out2 = cosine_tool.invoke({"doc_index1": 0, "doc_index2": 10})
    assert "error" in err_out2

    # Verify state immutability
    assert (session.feature_matrix != orig_matrix_copy).nnz == 0
    assert session.feature_names == orig_names
    assert list(session.labels) == orig_labels


def test_feature_correlation_matrix_tool_sample_fixture():
    session = _load_fixture_session()
    exploration_tools = make_exploration_tools(session)
    corr_tool = next(t for t in exploration_tools if t.name == "feature_correlation_matrix_tool")

    orig_matrix_copy = session.feature_matrix.copy()
    orig_names = list(session.feature_names)
    orig_labels = list(session.labels)

    # Known Answer #19
    res = corr_tool.invoke({"top_n": 20})
    assert "error" not in res

    # Known Answer #19 feature order by descending variance
    expected_terms = ["alpha", "gamma", "delta", "beta", "always"]
    assert res["terms"] == expected_terms

    corr_mat = res["correlation_matrix"]
    assert len(corr_mat) == 5
    assert len(corr_mat[0]) == 5

    # Map terms to matrix indices
    term_idx = {term: idx for idx, term in enumerate(expected_terms)}

    # Check nonconstant diagonals are 1.0
    for i in range(5):
        assert corr_mat[i][i] == pytest.approx(1.0, abs=1e-4)

    # Check matrix symmetry
    for i in range(5):
        for j in range(5):
            assert corr_mat[i][j] == corr_mat[j][i]

    # Known Answer #19 specified correlation values (tested with abs=1e-4)
    assert corr_mat[term_idx["alpha"]][term_idx["beta"]] == pytest.approx(0.8885, abs=1e-4)
    assert corr_mat[term_idx["gamma"]][term_idx["delta"]] == pytest.approx(0.8307, abs=1e-4)
    assert corr_mat[term_idx["alpha"]][term_idx["gamma"]] == pytest.approx(-0.5718, abs=1e-4)
    assert corr_mat[term_idx["alpha"]][term_idx["delta"]] == pytest.approx(-0.6882, abs=1e-4)
    assert corr_mat[term_idx["gamma"]][term_idx["beta"]] == pytest.approx(-0.6435, abs=1e-4)

    # Verify pending_figure set to Matplotlib Figure and close figure
    assert session.pending_figure is not None
    assert isinstance(session.pending_figure, plt.Figure)
    plt.close(session.pending_figure)

    # Verify result storage and args
    res_id = res["result_id"]
    assert res_id in session.results_store
    assert session.results_store[res_id]["record"].args == {"top_n": 20}

    # Verify strict JSON serialization
    json_str = json.dumps(res, allow_nan=False)
    assert isinstance(json_str, str)

    # Verify state immutability
    assert (session.feature_matrix != orig_matrix_copy).nnz == 0
    assert session.feature_names == orig_names
    assert list(session.labels) == orig_labels


def test_feature_correlation_constant_column_and_sample_validations():
    session = SessionState()

    # 1. Test dataset with fewer than 2 documents (<2 docs error)
    session.feature_matrix = csr_matrix([[1.0, 2.0]])
    session.feature_names = ["term1", "term2"]
    session.labels = np.array(["catA"])

    exploration_tools = make_exploration_tools(session)
    corr_tool = next(t for t in exploration_tools if t.name == "feature_correlation_matrix_tool")

    err_samples = corr_tool.invoke({"top_n": 20})
    assert "error" in err_samples

    # 2. Test dataset with 3 docs and 1 constant column (const_term) and 1 varying column (var_term)
    session.feature_matrix = csr_matrix([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    session.feature_names = ["var_term", "const_term"]
    session.labels = np.array(["catA", "catA", "catB"])

    corr_tool_2 = next(t for t in make_exploration_tools(session) if t.name == "feature_correlation_matrix_tool")
    res_const = corr_tool_2.invoke({"top_n": 20})
    assert "error" not in res_const

    terms = res_const["terms"]
    c_mat = res_const["correlation_matrix"]

    var_idx = terms.index("var_term")
    const_idx = terms.index("const_term")

    # Nonconstant term diagonal must be 1.0
    assert c_mat[var_idx][var_idx] == pytest.approx(1.0, abs=1e-4)

    # Constant column entries (including its diagonal and cross correlations) must be None
    assert c_mat[const_idx][const_idx] is None
    assert c_mat[var_idx][const_idx] is None
    assert c_mat[const_idx][var_idx] is None

    # Must pass strict JSON serialization without NaN
    json_str = json.dumps(res_const, allow_nan=False)
    assert isinstance(json_str, str)

    if session.pending_figure is not None:
        plt.close(session.pending_figure)
