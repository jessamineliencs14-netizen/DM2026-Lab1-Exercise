import matplotlib.pyplot as plt
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools


def test_build_dtm_tool_sample_fixture():
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

    # Initialize tools under test
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    res = build_dtm_tool.invoke({})

    # Check known answer 4 values from TEST_FIXTURE.md
    assert res["shape"] == [8, 5]
    assert res["n_documents"] == 8
    assert res["vocabulary_size"] == 5
    assert res["non_zero"] == 21
    assert res["sparsity_pct"] == pytest.approx(47.5, abs=1e-3)

    # Check state updates
    assert session.feature_matrix is not None
    assert session.feature_matrix.shape == (8, 5)
    assert session.feature_names == ["alpha", "always", "beta", "delta", "gamma"]

    # Check artifacts
    assert "count_vectorizer" in session.artifacts
    assert session.artifacts["count_vectorizer"] is not None

    # Check result_id registration in results_store
    assert "result_id" in res
    assert res["result_id"] in session.results_store


def test_term_frequency_tool_sample_fixture():
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
    term_frequency_tool = next(t for t in dtm_tools if t.name == "term_frequency_tool")

    build_dtm_tool.invoke({})

    orig_matrix_copy = session.feature_matrix.copy()
    orig_names = list(session.feature_names)

    res = term_frequency_tool.invoke({})

    # Known Answer #14 totals
    expected_freqs = {
        "alpha": 6,
        "always": 7,
        "beta": 3,
        "delta": 4,
        "gamma": 7,
    }
    assert res["term_frequencies"] == expected_freqs
    assert res["total_terms"] == 5

    # Check state immutability
    assert (session.feature_matrix != orig_matrix_copy).nnz == 0
    assert session.feature_names == orig_names

    # Check full_report format and sorting
    stored_entry = session.results_store[res["result_id"]]
    full_report = stored_entry["full_report"]
    assert list(full_report.columns[:2]) == ["term", "frequency"]
    freq_vals = full_report["frequency"].tolist()
    assert freq_vals == sorted(freq_vals, reverse=True)

    # Test top_n error validation
    err_res = term_frequency_tool.invoke({"top_n": 0})
    assert "error" in err_res


def test_dtm_heatmap_tool_sample_fixture():
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
    dtm_heatmap_tool = next(t for t in dtm_tools if t.name == "dtm_heatmap_tool")

    build_dtm_tool.invoke({})

    orig_matrix_copy = session.feature_matrix.copy()
    orig_names = list(session.feature_names)

    # Test 1: Default call (oversized slice: n_terms=20, n_documents=20) - Known Answer #21 full matrix
    res_default = dtm_heatmap_tool.invoke({"n_terms": 20, "n_documents": 20})

    expected_full_matrix = [
        [3, 1, 1, 0, 0],
        [2, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 0, 1, 3],
        [0, 1, 0, 1, 2],
        [0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0],
    ]

    assert res_default["matrix_slice"] == expected_full_matrix
    assert res_default["n_documents"] == 8
    assert res_default["n_terms"] == 5
    assert res_default["terms"] == ["alpha", "always", "beta", "delta", "gamma"]
    assert res_default["document_ids"] == list(range(8))

    # Verify pending_figure type
    assert session.pending_figure is not None
    assert isinstance(session.pending_figure, plt.Figure)
    plt.close(session.pending_figure)

    # Test 2: Custom slice (n_terms=2, n_documents=3) - Known Answer #21 top-left corner
    res_custom = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})

    expected_custom_slice = [
        [3, 1],
        [2, 1],
        [1, 1],
    ]

    assert res_custom["matrix_slice"] == expected_custom_slice
    assert res_custom["n_documents"] == 3
    assert res_custom["n_terms"] == 2
    assert res_custom["terms"] == ["alpha", "always"]
    assert res_custom["document_ids"] == [0, 1, 2]

    assert session.pending_figure is not None
    assert isinstance(session.pending_figure, plt.Figure)
    plt.close(session.pending_figure)

    # Test 3: Check state immutability
    assert (session.feature_matrix != orig_matrix_copy).nnz == 0
    assert session.feature_names == orig_names

    # Test 4: Verify result registration in results_store
    assert res_default["result_id"] in session.results_store
    assert res_custom["result_id"] in session.results_store

    # Test 5: Error validation for non-positive slice dimensions
    err_res_1 = dtm_heatmap_tool.invoke({"n_terms": 0, "n_documents": 5})
    assert "error" in err_res_1

    err_res_2 = dtm_heatmap_tool.invoke({"n_terms": 5, "n_documents": -1})
    assert "error" in err_res_2
