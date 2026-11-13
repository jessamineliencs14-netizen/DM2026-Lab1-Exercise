import math
from typing import Literal, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from langchain_core.tools import tool
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelBinarizer



def _clean_val(v):
    """Helper to convert float to round 4 decimal place float or None if NaN/Inf."""
    if isinstance(v, (float, np.floating)):
        if math.isnan(v) or math.isinf(v):
            return None
        return round(float(v), 4)
    return v


def make_tools(session):
    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"] = "pca",
        n_components: int = 2,
        random_state: Optional[int] = 42,
        perplexity: float = 30.0,
        n_neighbors: int = 15,
    ) -> dict:
        """Reduce DTM feature matrix dimensionality to 2D/3D and generate a scatter plot.

        Reduces session.feature_matrix using PCA, t-SNE, or UMAP. On EVERY run, builds a
        2D scatter plot colored by category_name and sets it on session.pending_figure.
        Does not modify session.feature_matrix, session.feature_names, or session.dataframe.

        Args:
            method: Reduction algorithm ('pca', 'tsne', or 'umap'). Default: 'pca'.
            n_components: Target dimensions (default: 2).
            random_state: Seed for reproducible output (default: 42).
            perplexity: t-SNE perplexity parameter (must be < n_samples).
            n_neighbors: UMAP n_neighbors parameter (must be < n_samples).

        Returns:
            dict: Summary containing result_id, method, n_components, coordinates (2D list),
                  explained_variance_ratio (PCA only), and category_labels.
        """
        if session.feature_matrix is None:
            return {
                "error": "No feature_matrix found in session. Run build_dtm_tool first."
            }

        X = session.feature_matrix.toarray()
        n_samples, n_features = X.shape

        if n_samples == 0:
            return {"error": "feature_matrix is empty."}

        # Convert labels to local array and validate dimensionality and alignment without mutating session.labels
        if session.labels is not None and len(session.labels) > 0:
            local_labels = np.asarray(session.labels)
            if local_labels.ndim != 1:
                return {"error": "session.labels must be 1-dimensional."}
            if len(local_labels) != n_samples:
                return {
                    "error": f"Label count ({len(local_labels)}) does not match feature_matrix row count ({n_samples})."
                }
            categories = local_labels.tolist()
        else:
            categories = ["Category"] * n_samples

        if n_components <= 0:
            return {"error": "n_components must be a positive integer."}

        # Method-specific validation against algorithm initialization limits
        if method == "pca":
            if n_components > min(n_samples, n_features):
                return {
                    "error": f"PCA n_components ({n_components}) cannot exceed min(n_samples, n_features) ({min(n_samples, n_features)})."
                }
        elif method == "tsne":
            # Default Barnes-Hut method requires n_components <= 3
            if n_components > 3:
                return {
                    "error": f"t-SNE (Barnes-Hut default) supports at most 3 components, got {n_components}."
                }
            if n_components > min(n_samples, n_features):
                return {
                    "error": f"t-SNE (with default PCA initialization) n_components ({n_components}) cannot exceed min(n_samples, n_features) ({min(n_samples, n_features)})."
                }
            if perplexity >= n_samples or perplexity <= 0:
                return {
                    "error": f"t-SNE perplexity ({perplexity}) must be strictly less than n_samples ({n_samples}) and > 0."
                }
        elif method == "umap":
            # UMAP spectral initialization requires n_components + 1 < n_samples
            if n_components + 1 >= n_samples:
                return {
                    "error": f"UMAP (spectral initialization) requires n_components + 1 < n_samples, got n_components={n_components} for n_samples={n_samples}."
                }
            if n_neighbors >= n_samples or n_neighbors <= 1:
                return {
                    "error": f"UMAP n_neighbors ({n_neighbors}) must be strictly less than n_samples ({n_samples}) and > 1."
                }

        explained_variance_ratio = None

        if method == "pca":
            reducer = PCA(n_components=n_components, random_state=random_state)
            coords = reducer.fit_transform(X)
            explained_variance_ratio = [
                _clean_val(v) for v in reducer.explained_variance_ratio_
            ]

        elif method == "tsne":
            reducer = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                random_state=random_state,
            )
            coords = reducer.fit_transform(X)

        elif method == "umap":
            import umap
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                random_state=random_state,
            )
            coords = reducer.fit_transform(X)

        else:
            return {"error": f"Unsupported reduction method '{method}'."}

        coords_list = [[_clean_val(val) for val in row] for row in coords]

        # Build compulsory scatter plot and set on session.pending_figure without plt.show()
        fig, ax = plt.subplots(figsize=(7, 5))
        plot_df = pd.DataFrame(
            {
                "Dim1": coords[:, 0],
                "Dim2": coords[:, 1] if n_components > 1 else np.zeros(n_samples),
                "Category": categories,
            }
        )
        sns.scatterplot(
            data=plot_df, x="Dim1", y="Dim2", hue="Category", style="Category", s=100, ax=ax
        )
        ax.set_title(f"Dimensionality Reduction ({method.upper()})")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("reduce_dimensions")
        summary = {
            "result_id": result_id,
            "method": method,
            "n_components": n_components,
            "coordinates": coords_list,
            "category_labels": categories,
        }
        if explained_variance_ratio is not None:
            summary["explained_variance_ratio"] = explained_variance_ratio

        # Store artifacts
        session.artifacts[f"reduced_coords_{method}"] = coords_list

        session.store_result(
            "reduce_dimensions_tool",
            {
                "method": method,
                "n_components": n_components,
                "random_state": random_state,
                "perplexity": perplexity,
                "n_neighbors": n_neighbors,
            },
            summary,
        )

        return summary

    @tool
    def binarize_labels_tool() -> dict:
        """One-hot encode category labels in session into binary (0/1) columns.

        Uses deterministic class order and stores the resulting binary array and mapping
        in session.artifacts['binarized_labels']. Does not mutate session.dataframe or DTM.

        Returns:
            dict: Summary containing result_id, categories, and one_hot_shape.
        """
        raw_labels = None
        if session.labels is not None and len(session.labels) > 0:
            raw_labels = session.labels
        elif session.dataframe is not None:
            if "category_name" in session.dataframe.columns:
                raw_labels = session.dataframe["category_name"].values
            elif "label" in session.dataframe.columns:
                raw_labels = session.dataframe["label"].values

        if raw_labels is None or len(raw_labels) == 0:
            return {"error": "No labels found in session or labels array is empty."}

        labels_arr = np.asarray(raw_labels)
        if labels_arr.ndim != 1:
            return {"error": "Labels must be 1-dimensional."}

        if len(labels_arr) == 0:
            return {"error": "Labels array is empty."}

        if session.dataframe is not None and len(labels_arr) != len(session.dataframe):
            return {
                "error": f"Label count ({len(labels_arr)}) does not match dataframe row count ({len(session.dataframe)})."
            }

        unique_classes = sorted(list(set(labels_arr)))
        n_classes = len(unique_classes)

        if n_classes == 1:
            # Single class -> n_rows x 1 matrix of ones
            binary_matrix = np.ones((len(labels_arr), 1), dtype=int)
            classes = [str(unique_classes[0])]
        elif n_classes == 2:
            # Two classes -> 2 columns [is_cls0, is_cls1] matching class order
            lb = LabelBinarizer()
            raw = lb.fit_transform(labels_arr)
            binary_matrix = np.hstack([1 - raw, raw]).astype(int)
            classes = [str(c) for c in lb.classes_]
        else:
            # Multiclass -> n_classes columns
            lb = LabelBinarizer()
            binary_matrix = lb.fit_transform(labels_arr).astype(int)
            classes = [str(c) for c in lb.classes_]

        binary_list = binary_matrix.tolist()
        class_mapping = {cls: idx for idx, cls in enumerate(classes)}

        artifact_payload = {
            "one_hot_matrix": binary_list,
            "classes": classes,
            "class_mapping": class_mapping,
        }
        session.artifacts["binarized_labels"] = artifact_payload

        result_id = session.next_result_id("binarize_labels")
        summary = {
            "result_id": result_id,
            "categories": classes,
            "one_hot_shape": [len(labels_arr), len(classes)],
            "class_mapping": class_mapping,
        }

        session.store_result("binarize_labels_tool", {}, summary)
        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]
