from typing import List, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from langchain_core.tools import tool
from sklearn.feature_extraction.text import CountVectorizer


def make_tools(session):
    @tool
    def build_dtm_tool(
        ngram_range: List[int] = [1, 1],
        max_features: Optional[int] = None,
        min_df: Union[int, float] = 1,
        max_df: Union[int, float] = 1.0,
    ) -> dict:
        """Build a document-term matrix (DTM) from document text stored in session.dataframe.

        Fits a CountVectorizer on session.dataframe['text'], populates session.feature_matrix
        (sparse matrix), session.feature_names (vocabulary list), and updates session.labels and
        session.categories. Stores the fitted vectorizer in session.artifacts['count_vectorizer'].

        Args:
            ngram_range: Lower and upper boundary of the range of n-values for different n-grams to be extracted.
                         Must be a list of 2 integers [min_n, max_n] with 1 <= min_n <= max_n.
            max_features: Build a vocabulary that only considers the top max_features ordered by term frequency.
            min_df: Ignore terms that have a document frequency strictly lower than the given threshold (int count >= 1 or float fraction).
            max_df: Ignore terms that have a document frequency strictly higher than the given threshold (int count or float fraction <= 1.0).

        Returns:
            dict: Summary containing result_id, vocabulary_size, n_documents, shape, non_zero, total_elements, and sparsity_pct.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session."}
        if "text" not in session.dataframe.columns:
            return {"error": "session.dataframe does not contain a 'text' column."}

        if len(ngram_range) != 2:
            return {"error": "ngram_range must be a list of exactly 2 integers [min_n, max_n]."}
        min_n, max_n = ngram_range[0], ngram_range[1]
        if not (isinstance(min_n, int) and isinstance(max_n, int)):
            return {"error": "ngram_range elements must be integers."}
        if min_n < 1:
            return {"error": "min_n in ngram_range must be >= 1."}
        if min_n > max_n:
            return {"error": "min_n must be <= max_n in ngram_range."}

        vectorizer = CountVectorizer(
            ngram_range=(min_n, max_n),
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
        )

        texts = session.dataframe["text"].fillna("").astype(str)
        feature_matrix = vectorizer.fit_transform(texts)
        feature_names = list(vectorizer.get_feature_names_out())

        session.feature_matrix = feature_matrix
        session.feature_names = feature_names

        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].values)
        elif "label" in session.dataframe.columns:
            session.set_labels(session.dataframe["label"].values)

        session.artifacts["count_vectorizer"] = vectorizer

        n_rows, n_cols = feature_matrix.shape
        non_zero = int(feature_matrix.nnz)
        total_elements = int(n_rows * n_cols)
        sparsity_pct = (
            100.0 * (1.0 - non_zero / total_elements) if total_elements > 0 else 0.0
        )

        result_id = session.next_result_id("dtm")
        summary = {
            "result_id": result_id,
            "vocabulary_size": n_cols,
            "n_documents": n_rows,
            "shape": [n_rows, n_cols],
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4),
        }

        args = {
            "ngram_range": ngram_range,
            "max_features": max_features,
            "min_df": min_df,
            "max_df": max_df,
        }

        stored_id = session.store_result("build_dtm_tool", args, summary)
        summary["result_id"] = stored_id
        return summary

    @tool
    def term_frequency_tool(top_n: Optional[int] = None) -> dict:
        """Aggregate total term frequencies across all documents in the document-term matrix.

        Sums columns of session.feature_matrix to calculate overall occurrence count per term.
        Does not modify session.feature_matrix or session.feature_names.

        Args:
            top_n: Optional integer to limit the summary to top N most frequent terms.

        Returns:
            dict: Summary containing result_id, total_terms, and term_frequencies dictionary.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "error": "No feature_matrix or feature_names found in session. Run build_dtm_tool first."
            }

        if top_n is not None and top_n <= 0:
            return {"error": "top_n must be a positive integer."}

        X = session.feature_matrix
        feature_names = session.feature_names

        col_sums = np.array(X.sum(axis=0)).ravel().astype(int)

        df_report = pd.DataFrame(
            {"term": feature_names, "frequency": col_sums}
        ).sort_values(by="frequency", ascending=False)

        term_freq_dict = dict(zip(df_report["term"], df_report["frequency"]))

        if top_n is not None:
            df_report_subset = df_report.head(top_n)
            summary_freqs = dict(
                zip(df_report_subset["term"], df_report_subset["frequency"])
            )
        else:
            summary_freqs = term_freq_dict

        result_id = session.next_result_id("term_frequency")
        summary = {
            "result_id": result_id,
            "total_terms": len(feature_names),
            "term_frequencies": summary_freqs,
        }

        session.store_result(
            "term_frequency_tool",
            {"top_n": top_n},
            summary,
            full_report=df_report,
        )

        return summary

    @tool
    def dtm_heatmap_tool(n_terms: int = 20, n_documents: int = 20) -> dict:
        """Generate a heatmap of a positional slice of the document-term matrix.

        Slices session.feature_matrix[0:n_documents, 0:n_terms] in raw positional order
        without filtering or sorting. Sets the resulting figure on session.pending_figure.

        Args:
            n_terms: Number of terms (columns) to include from vocabulary order (default: 20).
            n_documents: Number of documents (rows) to include from document order (default: 20).

        Returns:
            dict: Summary containing result_id, n_documents, n_terms, terms, document_ids, and matrix_slice (2D list).
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "error": "No feature_matrix or feature_names found in session. Run build_dtm_tool first."
            }

        if n_terms <= 0 or n_documents <= 0:
            return {"error": "n_terms and n_documents must be positive integers."}

        X = session.feature_matrix
        feature_names = session.feature_names
        n_rows, n_cols = X.shape

        actual_n_docs = min(n_documents, n_rows)
        actual_n_terms = min(n_terms, n_cols)

        # Slice sparse matrix first, then convert ONLY the slice to dense array
        sparse_slice = X[0:actual_n_docs, 0:actual_n_terms]
        dense_slice = sparse_slice.toarray()

        slice_terms = feature_names[0:actual_n_terms]
        slice_doc_ids = list(range(actual_n_docs))

        # Generate plot and set on session.pending_figure without calling plt.show()
        fig, ax = plt.subplots(figsize=(max(6, actual_n_terms * 0.8), max(4, actual_n_docs * 0.5)))
        sns.heatmap(
            dense_slice,
            xticklabels=slice_terms,
            yticklabels=slice_doc_ids,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            ax=ax,
        )
        ax.set_title(f"DTM Heatmap Slice ({actual_n_docs} docs x {actual_n_terms} terms)")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Document Index")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("dtm_heatmap")
        summary = {
            "result_id": result_id,
            "n_documents": actual_n_docs,
            "n_terms": actual_n_terms,
            "terms": slice_terms,
            "document_ids": slice_doc_ids,
            "matrix_slice": dense_slice.tolist(),
        }

        session.store_result(
            "dtm_heatmap_tool",
            {"n_terms": n_terms, "n_documents": n_documents},
            summary,
        )

        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
