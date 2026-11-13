import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_reduction import make_tools as make_reduction_tools


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


def test_pca_reduction_sample_fixture():
    session = _load_fixture_session()
    reduction_tools = make_reduction_tools(session)
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    orig_matrix_copy = session.feature_matrix.copy()
    orig_names = list(session.feature_names)
    orig_labels = list(session.labels)

    # Known Answer #7: method="pca", n_components=2
    res = reduce_tool.invoke({"method": "pca", "n_components": 2, "random_state": 42})
    assert "error" not in res

    # Known Answer #7 expected PCA coordinates
    expected_coords = [
        [2.3669, 0.8871],
        [1.7039, 0.2642],
        [1.0408, -0.3587],
        [-2.0759, 1.0129],
        [-1.4554, 0.3405],
        [-0.8349, -0.3319],
        [-0.8349, -0.3319],
        [0.0894, -1.4822],
    ]
    coords = res["coordinates"]
    assert len(coords) == 8
    for i in range(8):
        assert coords[i] == pytest.approx(expected_coords[i], abs=1e-3)

    # Known Answer #7 expected explained_variance_ratio
    assert res["explained_variance_ratio"] == pytest.approx([0.7532, 0.1965], abs=1e-3)

    # Verify pending_figure was set on EVERY run
    assert session.pending_figure is not None
    assert isinstance(session.pending_figure, plt.Figure)
    plt.close(session.pending_figure)

    # Verify artifacts, results_store, and state immutability
    res_id = res["result_id"]
    assert res_id in session.results_store
    assert session.results_store[res_id]["record"].args == {
        "method": "pca",
        "n_components": 2,
        "random_state": 42,
        "perplexity": 30.0,
        "n_neighbors": 15,
    }
    assert session.artifacts["reduced_coords_pca"] == coords
    assert (session.feature_matrix != orig_matrix_copy).nnz == 0
    assert session.feature_names == orig_names
    assert list(session.labels) == orig_labels


def test_tsne_reduction_sample_fixture():
    session = _load_fixture_session()
    reduction_tools = make_reduction_tools(session)
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    # Known Answer #8: method="tsne", perplexity=2, random_state=42
    res = reduce_tool.invoke(
        {"method": "tsne", "n_components": 2, "perplexity": 2.0, "random_state": 42}
    )
    assert "error" not in res
    coords = res["coordinates"]
    assert len(coords) == 8
    assert len(coords[0]) == 2

    # Verify all coordinates are finite float numbers
    for row in coords:
        for val in row:
            assert isinstance(val, float)
            assert not np.isnan(val) and not np.isinf(val)

    # Verify figure set on session.pending_figure
    assert session.pending_figure is not None
    assert isinstance(session.pending_figure, plt.Figure)
    plt.close(session.pending_figure)

    assert "reduced_coords_tsne" in session.artifacts


def test_umap_reduction_sample_fixture():
    session = _load_fixture_session()
    reduction_tools = make_reduction_tools(session)
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    # Known Answer #9: method="umap", n_neighbors=3, random_state=42
    res = reduce_tool.invoke(
        {"method": "umap", "n_components": 2, "n_neighbors": 3, "random_state": 42}
    )
    assert "error" not in res
    coords = res["coordinates"]
    assert len(coords) == 8
    assert len(coords[0]) == 2

    # Verify all coordinates are finite float numbers
    for row in coords:
        for val in row:
            assert isinstance(val, float)
            assert not np.isnan(val) and not np.isinf(val)

    # Verify figure set on session.pending_figure
    assert session.pending_figure is not None
    assert isinstance(session.pending_figure, plt.Figure)
    plt.close(session.pending_figure)

    assert "reduced_coords_umap" in session.artifacts


def test_reduction_parameter_and_alignment_validations():
    session = _load_fixture_session()
    reduction_tools = make_reduction_tools(session)
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    # 1. Invalid label alignment
    session.labels = np.array(["catA", "catB"])  # length 2 vs row count 8
    err_align = reduce_tool.invoke({"method": "pca"})
    assert "error" in err_align
    session.labels = session.dataframe["category_name"].values

    # 2. PCA n_components exceeding min(n_samples, n_features) (8 docs, 5 terms -> max 5)
    err_pca = reduce_tool.invoke({"method": "pca", "n_components": 10})
    assert "error" in err_pca

    # 3. t-SNE component limit (Barnes-Hut supports <= 3)
    err_tsne_comp = reduce_tool.invoke({"method": "tsne", "n_components": 4, "perplexity": 2})
    assert "error" in err_tsne_comp

    # 4. t-SNE invalid perplexity (must be < n_samples 8)
    err_tsne_perp = reduce_tool.invoke({"method": "tsne", "perplexity": 30.0})
    assert "error" in err_tsne_perp

    # 5. UMAP invalid n_neighbors (must be < n_samples 8)
    err_umap_neigh = reduce_tool.invoke({"method": "umap", "n_neighbors": 15})
    assert "error" in err_umap_neigh


def test_pca_constant_input_json_safety():
    session = SessionState()

    # Setup constant matrix (8 rows, 2 constant terms -> 0 variance)
    from scipy.sparse import csr_matrix
    session.feature_matrix = csr_matrix([[1.0, 1.0]] * 8)
    session.feature_names = ["const1", "const2"]
    session.labels = np.array(["catA"] * 8)

    reduction_tools = make_reduction_tools(session)
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    res = reduce_tool.invoke({"method": "pca", "n_components": 2})
    assert "error" not in res

    # Must pass strict JSON serialization without NaN/Inf
    json_str = json.dumps(res, allow_nan=False)
    assert isinstance(json_str, str)

    if session.pending_figure is not None:
        plt.close(session.pending_figure)


def test_binarize_labels_tool_cases():
    session = _load_fixture_session()
    reduction_tools = make_reduction_tools(session)
    binarize_tool = next(t for t in reduction_tools if t.name == "binarize_labels_tool")

    # --- Test 1: Binary class binarization (sample_fixture has 2 classes: catA, catB) ---
    res_bin = binarize_tool.invoke({})
    assert "error" not in res_bin
    assert res_bin["categories"] == ["catA", "catB"]
    assert res_bin["one_hot_shape"] == [8, 2]
    assert res_bin["class_mapping"] == {"catA": 0, "catB": 1}

    art = session.artifacts["binarized_labels"]
    matrix = art["one_hot_matrix"]
    assert len(matrix) == 8
    assert len(matrix[0]) == 2
    # Verify 0/1 values for catA ([1, 0]) and catB ([0, 1])
    assert matrix[0] == [1, 0]  # row 0 is catA
    assert matrix[3] == [0, 1]  # row 3 is catB

    # --- Test 2: Single class binarization ---
    session_single = SessionState()
    session_single.labels = np.array(["catA"] * 5)
    binarize_single = next(t for t in make_reduction_tools(session_single) if t.name == "binarize_labels_tool")

    res_single = binarize_single.invoke({})
    assert "error" not in res_single
    assert res_single["categories"] == ["catA"]
    assert res_single["one_hot_shape"] == [5, 1]
    assert session_single.artifacts["binarized_labels"]["one_hot_matrix"] == [[1], [1], [1], [1], [1]]

    # --- Test 3: Multiclass binarization ---
    session_multi = SessionState()
    session_multi.labels = np.array(["catA", "catB", "catC", "catA"])
    binarize_multi = next(t for t in make_reduction_tools(session_multi) if t.name == "binarize_labels_tool")

    res_multi = binarize_multi.invoke({})
    assert "error" not in res_multi
    assert res_multi["categories"] == ["catA", "catB", "catC"]
    assert res_multi["one_hot_shape"] == [4, 3]
    assert res_multi["class_mapping"] == {"catA": 0, "catB": 1, "catC": 2}

    multi_matrix = session_multi.artifacts["binarized_labels"]["one_hot_matrix"]
    assert multi_matrix[0] == [1, 0, 0]  # catA
    assert multi_matrix[1] == [0, 1, 0]  # catB
    assert multi_matrix[2] == [0, 0, 1]  # catC
    assert multi_matrix[3] == [1, 0, 0]  # catA

    # --- Test 4: Empty labels rejection ---
    session_empty = SessionState()
    session_empty.labels = np.array([])
    binarize_empty = next(t for t in make_reduction_tools(session_empty) if t.name == "binarize_labels_tool")
    err_empty = binarize_empty.invoke({})
    assert "error" in err_empty
