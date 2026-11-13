import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def variance_filter_tool(threshold: float = 0.0) -> dict:
        """Filter DTM features by variance across documents without modifying session.feature_matrix.

        Computes population variance (ddof=0) for each term in session.feature_matrix.
        Terms with variance >= threshold are kept; terms with variance < threshold are flagged as removed.

        Args:
            threshold: Minimum variance threshold to keep a feature (default: 0.0).

        Returns:
            dict: Summary containing result_id, threshold, n_kept, n_removed, kept_terms, and removed_terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "error": "No feature_matrix or feature_names found in session. Run build_dtm_tool first."
            }

        X = session.feature_matrix
        feature_names = session.feature_names

        # Compute mean and variance per column for scipy sparse matrix (ddof=0)
        means = np.array(X.mean(axis=0)).ravel()
        sq_means = np.array(X.power(2).mean(axis=0)).ravel()
        variances = sq_means - (means**2)

        df_report = pd.DataFrame(
            {"term": feature_names, "variance": variances}
        ).sort_values(by="variance", ascending=False)

        kept_terms = df_report[df_report["variance"] >= threshold]["term"].tolist()
        removed_terms = df_report[df_report["variance"] < threshold]["term"].tolist()

        result_id = session.next_result_id("variance_filter")
        summary = {
            "result_id": result_id,
            "threshold": threshold,
            "n_kept": len(kept_terms),
            "n_removed": len(removed_terms),
            "kept_terms": kept_terms,
            "removed_terms": removed_terms,
        }

        session.store_result(
            "variance_filter_tool",
            {"threshold": threshold},
            summary,
            full_report=df_report,
        )

        return summary

    @tool
    def pearson_filter_tool(target_class: str, threshold: float = 0.0) -> dict:
        """Filter DTM features by Pearson correlation with a target class (One-vs-Rest).

        Computes Pearson correlation (r) between each term's count and a binary target array
        (1 if document label == target_class, 0 otherwise). Selection and sorting use signed
        correlation: terms with pearson_r >= threshold are kept, and full_report is sorted
        in descending order of signed pearson_r.

        Args:
            target_class: The target category name to correlate against (e.g. 'catB').
            threshold: Minimum signed Pearson correlation to keep a feature (default: 0.0).

        Returns:
            dict: Summary containing result_id, target_class, threshold, n_kept, n_removed, kept_terms, and removed_terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "error": "No feature_matrix or feature_names found in session. Run build_dtm_tool first."
            }
        if session.labels is None or len(session.labels) == 0:
            return {"error": "No labels found in session."}

        labels = np.array(session.labels)
        X = session.feature_matrix
        feature_names = session.feature_names

        if X.shape[0] != len(labels):
            return {
                "error": f"Label length ({len(labels)}) does not match feature_matrix row count ({X.shape[0]})."
            }

        y = (labels == target_class).astype(float)
        if np.all(y == 0) or np.all(y == 1):
            return {
                "error": f"Target class '{target_class}' produces constant binary labels (all 0 or all 1)."
            }

        # Calculate Pearson r for each feature column against y
        # r = Cov(X_i, y) / (std(X_i) * std(y))
        y_mean = np.mean(y)
        y_std = np.std(y, ddof=0)

        pearson_r_list = []
        for i in range(X.shape[1]):
            col = X[:, i].toarray().ravel()
            col_std = np.std(col, ddof=0)
            if col_std == 0:
                pearson_r_list.append(0.0)
            else:
                cov = np.mean((col - np.mean(col)) * (y - y_mean))
                r = cov / (col_std * y_std)
                # Ensure no NaN or floating precision overflow
                pearson_r_list.append(0.0 if np.isnan(r) else float(r))

        df_report = pd.DataFrame(
            {"term": feature_names, "pearson_r": pearson_r_list}
        ).sort_values(by="pearson_r", ascending=False)

        kept_terms = df_report[df_report["pearson_r"] >= threshold]["term"].tolist()
        removed_terms = df_report[df_report["pearson_r"] < threshold]["term"].tolist()

        result_id = session.next_result_id("pearson_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "n_kept": len(kept_terms),
            "n_removed": len(removed_terms),
            "kept_terms": kept_terms,
            "removed_terms": removed_terms,
        }

        session.store_result(
            "pearson_filter_tool",
            {"target_class": target_class, "threshold": threshold},
            summary,
            full_report=df_report,
        )

        return summary

    @tool
    def spearman_filter_tool(target_class: str, threshold: float = 0.0) -> dict:
        """Filter DTM features by Spearman rank correlation with a target class (One-vs-Rest).

        Computes Spearman correlation (r) between each term's count and a binary target array
        (1 if document label == target_class, 0 otherwise) handling tied ranks correctly.
        Selection and sorting use signed correlation: terms with spearman_r >= threshold are kept,
        and full_report is sorted in descending order of signed spearman_r.

        Args:
            target_class: The target category name to correlate against (e.g. 'catB').
            threshold: Minimum signed Spearman correlation to keep a feature (default: 0.0).

        Returns:
            dict: Summary containing result_id, target_class, threshold, n_kept, n_removed, kept_terms, and removed_terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "error": "No feature_matrix or feature_names found in session. Run build_dtm_tool first."
            }
        if session.labels is None or len(session.labels) == 0:
            return {"error": "No labels found in session."}

        labels = np.array(session.labels)
        X = session.feature_matrix
        feature_names = session.feature_names

        if X.shape[0] != len(labels):
            return {
                "error": f"Label length ({len(labels)}) does not match feature_matrix row count ({X.shape[0]})."
            }

        y = (labels == target_class).astype(float)
        if np.all(y == 0) or np.all(y == 1):
            return {
                "error": f"Target class '{target_class}' produces constant binary labels (all 0 or all 1)."
            }

        spearman_r_list = []
        for i in range(X.shape[1]):
            col = X[:, i].toarray().ravel()
            if np.std(col) == 0:
                spearman_r_list.append(0.0)
            else:
                res = spearmanr(col, y)
                r = res.statistic if hasattr(res, "statistic") else res.correlation
                spearman_r_list.append(0.0 if np.isnan(r) else float(r))

        df_report = pd.DataFrame(
            {"term": feature_names, "spearman_r": spearman_r_list}
        ).sort_values(by="spearman_r", ascending=False)

        kept_terms = df_report[df_report["spearman_r"] >= threshold]["term"].tolist()
        removed_terms = df_report[df_report["spearman_r"] < threshold]["term"].tolist()

        result_id = session.next_result_id("spearman_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "n_kept": len(kept_terms),
            "n_removed": len(removed_terms),
            "kept_terms": kept_terms,
            "removed_terms": removed_terms,
        }

        session.store_result(
            "spearman_filter_tool",
            {"target_class": target_class, "threshold": threshold},
            summary,
            full_report=df_report,
        )

        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
