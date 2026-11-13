import math
import os
from typing import Optional
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from langchain_core.tools import tool

# Ensure NLTK punkt data is downloaded
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


def _invalidate_row_dependent_state(session):
    """Helper to invalidate DTM, figures, and row-dependent artifacts when session.dataframe rows change."""
    session.feature_matrix = None
    session.feature_names = None
    session.labels = None
    session.categories = []
    session.pending_figure = None

    # Clear row-dependent artifacts
    keys_to_clear = [k for k in session.artifacts if k == "unigrams" or k.startswith("patterns_") or k == "count_vectorizer"]
    for k in keys_to_clear:
        del session.artifacts[k]


def _clean_stats_dict(d: dict) -> dict:
    """Helper to ensure describe statistics contain no NaN/Inf so json.dumps(..., allow_nan=False) succeeds."""
    clean_d = {}
    for k, v in d.items():
        if isinstance(v, (float, np.floating)):
            if math.isnan(v) or math.isinf(v):
                clean_d[str(k)] = None
            else:
                clean_d[str(k)] = round(float(v), 4)
        elif isinstance(v, (int, np.integer)):
            clean_d[str(k)] = int(v)
        elif v is None:
            clean_d[str(k)] = None
        else:
            clean_d[str(k)] = v
    return clean_d


def _to_json_safe(val):
    """Recursively convert values (including lists/tuples/dicts) to Python native JSON-safe types."""
    if isinstance(val, (list, tuple)):
        return [_to_json_safe(item) for item in val]
    if isinstance(val, dict):
        return {str(k): _to_json_safe(v) for k, v in val.items()}

    # Check bool before int (since bool inherits from int in Python)
    if isinstance(val, (bool, np.bool_)):
        return bool(val)

    if isinstance(val, (int, np.integer)):
        return int(val)

    if isinstance(val, (float, np.floating)):
        if math.isnan(val) or math.isinf(val):
            return None
        return float(val)

    if pd.isna(val):
        return None

    return str(val) if val is not None else None


def make_tools(session):
    @tool
    def inspect_data_tool(n: int = 5) -> dict:
        """Preview the first n rows of session.dataframe without modifying session data.

        Args:
            n: Number of initial rows to preview (default: 5).

        Returns:
            dict: Summary containing result_id, total_rows, n_rows, columns, row_indices, and data records.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session."}

        if n <= 0:
            return {"error": "n must be a positive integer."}

        df = session.dataframe
        total_rows = len(df)
        actual_n = min(n, total_rows)

        preview_df = df.head(actual_n)

        # Convert records to JSON-safe dicts using recursive _to_json_safe helper
        records = []
        for idx, row in preview_df.iterrows():
            row_dict = {col: _to_json_safe(row[col]) for col in df.columns}
            records.append(row_dict)

        result_id = session.next_result_id("inspect_data")
        summary = {
            "result_id": result_id,
            "total_rows": total_rows,
            "n_rows": actual_n,
            "columns": list(df.columns),
            "row_indices": preview_df.index.tolist(),
            "data": records,
        }

        session.store_result("inspect_data_tool", {"n": n}, summary)
        return summary

    @tool
    def tokenize_tool() -> dict:
        """Tokenize each document's text in session.dataframe into unigram tokens (lowercased).

        Applies NLTK word_tokenize on lowercased text and populates a 'unigrams' column in session.dataframe.

        Returns:
            dict: Summary containing result_id, n_documents, and sample_tokens.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session."}
        if "text" not in session.dataframe.columns:
            return {"error": "session.dataframe does not contain a 'text' column."}

        def _tok(val):
            if pd.isna(val) or val is None or str(val).strip() == "":
                return []
            return nltk.word_tokenize(str(val).lower())

        session.dataframe["unigrams"] = session.dataframe["text"].apply(_tok)
        tokens_list = session.dataframe["unigrams"].tolist()

        result_id = session.next_result_id("tokenize")
        summary = {
            "result_id": result_id,
            "n_documents": len(tokens_list),
            "total_tokens": sum(len(t) for t in tokens_list),
            "sample_tokens": tokens_list[:3],
        }

        session.artifacts["unigrams"] = tokens_list
        session.store_result("tokenize_tool", {}, summary)
        return summary

    @tool
    def list_files_tool(subdirectory: str = "") -> dict:
        """List files and directories in a repository workspace folder.

        Args:
            subdirectory: Subdirectory path relative to workspace root (default: "").

        Returns:
            dict: Summary containing result_id, subdirectory, directories, and files lists.
        """
        target_path = os.path.normpath(os.path.join(".", subdirectory))
        if not os.path.exists(target_path) or not os.path.isdir(target_path):
            return {"error": f"Directory '{subdirectory}' does not exist."}

        entries = os.listdir(target_path)
        dirs = []
        files = []
        for entry in entries:
            full_p = os.path.join(target_path, entry)
            if os.path.isdir(full_p):
                dirs.append(entry)
            else:
                files.append(entry)

        result_id = session.next_result_id("list_files")
        summary = {
            "result_id": result_id,
            "subdirectory": subdirectory,
            "directories": sorted(dirs),
            "files": sorted(files),
        }

        session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary)
        return summary

    @tool
    def check_missing_tool() -> dict:
        """Check for missing or empty text values in session.dataframe.

        Returns:
            dict: Summary containing result_id, total_rows, missing_count, and missing_indices.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session."}
        if "text" not in session.dataframe.columns:
            return {"error": "session.dataframe does not contain a 'text' column."}

        texts = session.dataframe["text"]
        missing_mask = texts.isna() | (texts.astype(str).str.strip() == "")
        missing_indices = session.dataframe.index[missing_mask].tolist()

        result_id = session.next_result_id("check_missing")
        summary = {
            "result_id": result_id,
            "total_rows": len(session.dataframe),
            "missing_count": int(missing_mask.sum()),
            "missing_indices": missing_indices,
        }

        session.store_result("check_missing_tool", {}, summary)
        return summary

    @tool
    def check_duplicates_tool(drop: bool = False) -> dict:
        """Check for and optionally drop duplicate documents in session.dataframe.

        Detects extra duplicate copies using keep='first' (reports duplicate_count).
        When drop=True, drops ALL copies using keep=False matching Master pipeline,
        reports actual dropped_count, and invalidates downstream DTM state.

        Args:
            drop: Whether to mutate session.dataframe by dropping duplicate rows (default: False).

        Returns:
            dict: Summary containing result_id, duplicate_count, dropped_count, remaining_rows, and duplicate_indices.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session."}

        df = session.dataframe
        dup_cols = [c for c in ["text", "label", "category_name"] if c in df.columns]
        if not dup_cols:
            dup_cols = [c for c in df.columns if c != "unigrams"]

        # 1. Detection count uses keep='first' (identifies extra duplicate copies)
        dup_mask_first = df.duplicated(subset=dup_cols, keep="first")
        dup_count = int(dup_mask_first.sum())
        dup_indices = df.index[dup_mask_first].tolist()

        dropped_count = 0
        if drop and dup_count > 0:
            initial_rows = len(df)
            # 2. Dropping uses keep=False matching Master's X.drop_duplicates(keep=False)
            session.dataframe = df.drop_duplicates(subset=dup_cols, keep=False)
            dropped_count = initial_rows - len(session.dataframe)

            _invalidate_row_dependent_state(session)
            if "category_name" in session.dataframe.columns:
                session.set_labels(session.dataframe["category_name"].values)
            elif "label" in session.dataframe.columns:
                session.set_labels(session.dataframe["label"].values)

        result_id = session.next_result_id("check_duplicates")
        summary = {
            "result_id": result_id,
            "duplicate_count": dup_count,
            "dropped_count": dropped_count,
            "remaining_rows": len(session.dataframe),
            "duplicate_indices": dup_indices,
        }

        session.store_result("check_duplicates_tool", {"drop": drop}, summary)
        return summary

    @tool
    def sample_data_tool(n: int = 4, random_state: int = 42) -> dict:
        """Subsample rows from session.dataframe.

        Subsamples session.dataframe in place, invalidating downstream DTM state,
        and synchronizing labels. Original dataframe indices are preserved in sampled_indices.

        Args:
            n: Number of samples to draw (default: 4).
            random_state: Seed for reproducible random sampling (default: 42).

        Returns:
            dict: Summary containing result_id, n_requested, n_sampled, and sampled_indices.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session."}

        if n <= 0:
            return {"error": "n must be a positive integer."}

        actual_n = min(n, len(session.dataframe))
        sampled_df = session.dataframe.sample(n=actual_n, random_state=random_state)
        sampled_indices = sampled_df.index.tolist()

        session.dataframe = sampled_df
        _invalidate_row_dependent_state(session)
        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].values)
        elif "label" in session.dataframe.columns:
            session.set_labels(session.dataframe["label"].values)

        result_id = session.next_result_id("sample_data")
        summary = {
            "result_id": result_id,
            "n_requested": n,
            "n_sampled": actual_n,
            "random_state": random_state,
            "sampled_indices": sampled_indices,
        }

        session.store_result(
            "sample_data_tool", {"n": n, "random_state": random_state}, summary
        )
        return summary

    @tool
    def describe_data_tool() -> dict:
        """Compute character length statistics and a boxplot by category for session.dataframe.

        Calculates text_length as df['text'].apply(len) and returns standard describe()
        summary statistics overall and broken down per category_name. Sets the resulting
        boxplot on session.pending_figure.

        Returns:
            dict: Summary containing result_id, overall_stats, and per_category_stats.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session."}
        if "text" not in session.dataframe.columns:
            return {"error": "session.dataframe does not contain a 'text' column."}

        df = session.dataframe
        cat_col = "category_name" if "category_name" in df.columns else ("label" if "label" in df.columns else None)

        lengths = df["text"].fillna("").astype(str).apply(len)

        overall_desc = _clean_stats_dict(lengths.describe().to_dict())
        per_cat_desc = {}
        if cat_col:
            for cat, group in df.groupby(cat_col):
                g_lens = group["text"].fillna("").astype(str).apply(len)
                per_cat_desc[str(cat)] = _clean_stats_dict(g_lens.describe().to_dict())

        # Build boxplot figure and set on session.pending_figure without plt.show()
        fig, ax = plt.subplots(figsize=(7, 4))
        plot_df = pd.DataFrame({"text_length": lengths})
        if cat_col:
            plot_df["category"] = df[cat_col].values
            sns.boxplot(data=plot_df, x="category", y="text_length", ax=ax)
            ax.set_xlabel("Category")
        else:
            sns.boxplot(data=plot_df, y="text_length", ax=ax)
        ax.set_ylabel("Text Length (chars)")
        ax.set_title("Document Text Length Distribution by Category")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("describe_data")
        summary = {
            "result_id": result_id,
            "n_documents": len(df),
            "overall_stats": overall_desc,
            "per_category_stats": per_cat_desc,
        }

        session.store_result("describe_data_tool", {}, summary)
        return summary

    return [
        inspect_data_tool,
        tokenize_tool,
        list_files_tool,
        check_missing_tool,
        check_duplicates_tool,
        sample_data_tool,
        describe_data_tool,
    ]
