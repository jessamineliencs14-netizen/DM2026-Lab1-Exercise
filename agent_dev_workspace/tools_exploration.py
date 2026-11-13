import math
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from langchain_core.tools import tool
from sklearn.metrics.pairwise import cosine_similarity


def _clean_val(v):
    """Helper to convert float to round 4 decimal place float or None if NaN/Inf."""
    if isinstance(v, (float, np.floating)):
        if math.isnan(v) or math.isinf(v):
            return None
        return round(float(v), 4)
    return v


def make_tools(session):
    @tool
    def cosine_similarity_tool(doc_index1: int, doc_index2: int) -> dict:
        """Compute pairwise cosine similarity between two documents in session.feature_matrix.

        Uses raw DTM term counts. Zero-vector documents score 0.0 without raising NaN/errors.
        Does not modify session.feature_matrix, session.feature_names, or session.dataframe.

        Args:
            doc_index1: 0-based index of the first document row.
            doc_index2: 0-based index of the second document row.

        Returns:
            dict: Summary containing result_id, doc_index1, doc_index2, and similarity score.
        """
        if session.feature_matrix is None:
            return {
                "error": "No feature_matrix found in session. Run build_dtm_tool first."
            }

        X = session.feature_matrix
        n_samples = X.shape[0]

        if doc_index1 < 0 or doc_index1 >= n_samples or doc_index2 < 0 or doc_index2 >= n_samples:
            return {
                "error": f"doc_index1 ({doc_index1}) and doc_index2 ({doc_index2}) must be between 0 and {n_samples - 1}."
            }

        vec1 = X[doc_index1 : doc_index1 + 1].toarray()
        vec2 = X[doc_index2 : doc_index2 + 1].toarray()

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        # Zero-vector document check (e.g. empty text row) yields exactly 0.0
        if norm1 == 0 or norm2 == 0:
            sim_score = 0.0
        else:
            raw_sim = float(cosine_similarity(vec1, vec2)[0, 0])
            sim_score = _clean_val(raw_sim)
            if sim_score is None:
                sim_score = 0.0

        result_id = session.next_result_id("cosine_sim")
        summary = {
            "result_id": result_id,
            "doc_index1": doc_index1,
            "doc_index2": doc_index2,
            "similarity": sim_score,
        }

        session.store_result(
            "cosine_similarity_tool",
            {"doc_index1": doc_index1, "doc_index2": doc_index2},
            summary,
        )

        return summary

    @tool
    def feature_correlation_matrix_tool(top_n: int = 20) -> dict:
        """Compute feature-vs-feature Pearson correlation heatmap for top terms by variance.

        Selects up to top_n terms by descending population variance across all documents in
        session.feature_matrix using sparse operations. Computes Pearson correlation matrix,
        building a Seaborn heatmap and setting it on session.pending_figure.

        Args:
            top_n: Maximum number of top-variance terms to include (default: 20).

        Returns:
            dict: Summary containing result_id, terms, and correlation_matrix (2D list with None for constant columns).
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "error": "No feature_matrix or feature_names found in session. Run build_dtm_tool first."
            }

        if top_n <= 0:
            return {"error": "top_n must be a positive integer."}

        X_sparse = session.feature_matrix
        n_samples, n_features = X_sparse.shape

        if n_samples < 2:
            return {
                "error": f"feature_correlation_matrix_tool requires at least 2 documents, got {n_samples}."
            }
        if n_features == 0:
            return {"error": "feature_matrix has 0 terms/columns."}

        # 1. Compute population variance efficiently using sparse operations
        means = np.array(X_sparse.mean(axis=0)).ravel()
        sq_means = np.array(X_sparse.power(2).mean(axis=0)).ravel()
        variances = sq_means - (means**2)

        actual_top_n = min(top_n, n_features)

        # 2. Select top_n terms by variance descending (stable sort)
        top_indices = np.argsort(-variances, kind="stable")[:actual_top_n]
        selected_terms = [session.feature_names[i] for i in top_indices]

        # Convert ONLY the top_n selected columns to dense
        X_sub = X_sparse[:, top_indices].toarray()

        # 3. Compute Pearson correlation matrix across selected features
        stds = np.std(X_sub, axis=0, ddof=0)
        corr_mat_float = np.empty((actual_top_n, actual_top_n), dtype=object)
        plot_corr_mat = np.full((actual_top_n, actual_top_n), np.nan, dtype=float)

        for i in range(actual_top_n):
            for j in range(actual_top_n):
                if stds[i] == 0 or stds[j] == 0:
                    # Constant column correlation is undefined -> None in JSON, NaN in plot_corr_mat
                    corr_mat_float[i, j] = None
                    plot_corr_mat[i, j] = np.nan
                else:
                    if i == j:
                        r_val = 1.0
                    else:
                        cov = np.mean((X_sub[:, i] - np.mean(X_sub[:, i])) * (X_sub[:, j] - np.mean(X_sub[:, j])))
                        r_val = cov / (stds[i] * stds[j])
                        r_val = _clean_val(r_val)
                    corr_mat_float[i, j] = r_val
                    plot_corr_mat[i, j] = r_val if r_val is not None else np.nan

        corr_list = corr_mat_float.tolist()

        # 4. Build heatmap figure and set on session.pending_figure without plt.show()
        # Mask NaN/constant entries in seaborn heatmap
        mask = np.isnan(plot_corr_mat)
        fig, ax = plt.subplots(figsize=(max(6, actual_top_n * 0.8), max(5, actual_top_n * 0.7)))
        sns.heatmap(
            plot_corr_mat,
            mask=mask,
            xticklabels=selected_terms,
            yticklabels=selected_terms,
            annot=True,
            fmt=".4f",
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
            ax=ax,
        )
        ax.set_title(f"Feature Correlation Matrix (Top {actual_top_n} Terms by Variance)")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("feature_corr")
        summary = {
            "result_id": result_id,
            "terms": selected_terms,
            "correlation_matrix": corr_list,
        }

        full_report_df = pd.DataFrame(plot_corr_mat, index=selected_terms, columns=selected_terms)

        session.store_result(
            "feature_correlation_matrix_tool",
            {"top_n": top_n},
            summary,
            full_report=full_report_df,
        )

        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
