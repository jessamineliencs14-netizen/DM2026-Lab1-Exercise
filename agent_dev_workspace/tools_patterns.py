import ast
import os
import tempfile
from typing import Literal, Optional
import numpy as np
import pandas as pd
from langchain_core.tools import tool
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import PAMI.extras.convert.DF2DB as db_converter
from PAMI.frequentPattern.basic import FPGrowth
from PAMI.frequentPattern.maximal import MaxFPGrowth
from PAMI.frequentPattern.topk import FAE


def _normalize_pami_pattern_key(pat_key) -> list[str]:
    """Helper to convert a PAMI pattern key into a list of term strings."""
    if isinstance(pat_key, (list, tuple)):
        return [str(item) for item in pat_key]

    if isinstance(pat_key, str):
        s = pat_key.strip()
        # Check if key is a stringified tuple or list representation (e.g. "('alpha',)" or "['alpha']")
        if (s.startswith("(") and s.endswith(")")) or (s.startswith("[") and s.endswith("]")):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)):
                    return [str(item) for item in parsed]
            except (ValueError, SyntaxError):
                pass

        # Fallback for plain whitespace or tab-separated string patterns
        return [item.strip() for item in s.split() if item.strip()]

    return [str(pat_key)]


def make_tools(session):
    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: Literal["variance", "tfidf", "term_frequency"],
        algorithm: Literal["fpgrowth", "topk", "maxfpgrowth"] = "fpgrowth",
        min_sup: Optional[int] = None,
        k: Optional[int] = None,
    ) -> dict:
        """Mine frequent word patterns for a specific document category using PAMI algorithms.

        Filters session.dataframe by category_name and builds a fresh, category-local DTM
        using CountVectorizer(stop_words='english'). Applies the chosen local filtering method:
        - 'variance' or 'term_frequency': keeps middle 90% (5th < score < 95th percentile).
        - 'tfidf': keeps top 80% (score >= 20th percentile).
        Converts surviving terms into transactions and mines patterns with PAMI algorithms:
        - 'fpgrowth': FPGrowth (requires min_sup).
        - 'topk': FAE (requires k).
        - 'maxfpgrowth': MaxFPGrowth (requires min_sup).

        Args:
            category_name: Category to mine (e.g. 'catA').
            filtering_method: Local vocabulary filtering method ('variance', 'tfidf', or 'term_frequency').
            algorithm: Mining algorithm ('fpgrowth', 'topk', or 'maxfpgrowth'). Default: 'fpgrowth'.
            min_sup: Integer minimum support count threshold (required for 'fpgrowth' and 'maxfpgrowth').
            k: Top-k pattern count threshold (required for 'topk').

        Returns:
            dict: Summary containing result_id, category_name, filtering_method, algorithm,
                  surviving_terms_count, patterns_count, patterns list, and note if empty.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session."}

        df = session.dataframe
        if "category_name" in df.columns:
            cat_col = "category_name"
        elif "label" in df.columns:
            cat_col = "label"
        else:
            return {"error": "session.dataframe does not contain a category or label column."}

        if "text" not in df.columns:
            return {"error": "session.dataframe does not contain a 'text' column."}

        cat_df = df[df[cat_col] == category_name]
        if len(cat_df) == 0:
            return {"error": f"No documents found for category '{category_name}'."}

        # Parameter validation
        if algorithm in ["fpgrowth", "maxfpgrowth"]:
            if min_sup is None or min_sup <= 0:
                return {"error": f"Algorithm '{algorithm}' requires a positive integer 'min_sup'."}
        elif algorithm == "topk":
            if k is None or k <= 0:
                return {"error": "Algorithm 'topk' requires a positive integer 'k'."}
        else:
            return {"error": f"Unsupported algorithm '{algorithm}'."}

        effective_args = {
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "min_sup": min_sup,
            "k": k,
        }

        # 1. Build fresh, category-local DTM with English stop words
        vectorizer = CountVectorizer(stop_words="english")
        texts = cat_df["text"].fillna("").astype(str)
        try:
            X_cat = vectorizer.fit_transform(texts)
            cat_terms = np.array(vectorizer.get_feature_names_out())
        except ValueError:
            # Handle empty vocabulary after stopword removal
            cat_terms = np.array([])

        if len(cat_terms) == 0:
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "surviving_terms_count": 0,
                "patterns_count": 0,
                "patterns": [],
                "note": "no terms survived stopword removal",
            }
            artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
            session.artifacts[artifact_key] = []
            session.store_result("mine_patterns_tool", effective_args, summary)
            return summary

        # 2. Local vocabulary filtering
        dense_X = X_cat.toarray()
        if filtering_method == "variance":
            scores = np.var(dense_X, axis=0, ddof=0)
            p5, p95 = np.percentile(scores, 5), np.percentile(scores, 95)
            mask = (scores > p5) & (scores < p95)
        elif filtering_method == "term_frequency":
            scores = np.sum(dense_X, axis=0)
            p5, p95 = np.percentile(scores, 5), np.percentile(scores, 95)
            mask = (scores > p5) & (scores < p95)
        elif filtering_method == "tfidf":
            tfidf_mat = TfidfTransformer().fit_transform(X_cat).toarray()
            scores = np.mean(tfidf_mat, axis=0)
            p20 = np.percentile(scores, 20)
            mask = scores >= p20
        else:
            return {"error": f"Unsupported filtering_method '{filtering_method}'."}

        surviving_terms = cat_terms[mask]
        if len(surviving_terms) == 0:
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "surviving_terms_count": 0,
                "patterns_count": 0,
                "patterns": [],
                "note": "no terms survived filtering",
            }
            artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
            session.artifacts[artifact_key] = []
            session.store_result("mine_patterns_tool", effective_args, summary)
            return summary

        # 3. Create transactional database dataframe (items present with count >= 1)
        surviving_counts = dense_X[:, mask]
        binary_df = pd.DataFrame(
            (surviving_counts >= 1).astype(int), columns=surviving_terms
        )

        # Drop transactions with 0 items
        binary_df = binary_df[binary_df.sum(axis=1) > 0]
        if len(binary_df) == 0:
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "surviving_terms_count": len(surviving_terms),
                "patterns_count": 0,
                "patterns": [],
                "note": "no non-empty transactions found after filtering",
            }
            artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
            session.artifacts[artifact_key] = []
            session.store_result("mine_patterns_tool", effective_args, summary)
            return summary

        # 4. Convert to PAMI Transactional DB file and run PAMI algorithm
        temp_db_file = None
        mined_patterns = []
        try:
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp:
                temp_db_file = tmp.name

            # Convert DF to DB file using exact PAMI DF2DB signatures
            conv = db_converter.DF2DB(binary_df, DFtype="dense")
            conv.convert2TransactionalDatabase(
                temp_db_file, condition=">=", thresholdValue=1
            )

            # Execute requested PAMI algorithm
            if algorithm == "fpgrowth":
                miner = FPGrowth.FPGrowth(iFile=temp_db_file, minSup=min_sup, sep="\t")
                miner.startMine()
                patterns_dict = miner.getPatterns()
            elif algorithm == "topk":
                miner = FAE.FAE(iFile=temp_db_file, k=k, sep="\t")
                miner.startMine()
                patterns_dict = miner.getPatterns()
            elif algorithm == "maxfpgrowth":
                miner = MaxFPGrowth.MaxFPGrowth(iFile=temp_db_file, minSup=min_sup, sep="\t")
                miner.startMine()
                patterns_dict = miner.getPatterns()

            # Convert patterns dict to standard list of dicts using safe pattern key normalization
            if patterns_dict:
                for pat_key, sup_val in patterns_dict.items():
                    pat_items = _normalize_pami_pattern_key(pat_key)
                    mined_patterns.append({"pattern": pat_items, "support": int(sup_val)})

        except Exception as e:
            return {"error": f"PAMI pattern mining execution failed: {str(e)}"}
        finally:
            if temp_db_file and os.path.exists(temp_db_file):
                try:
                    os.remove(temp_db_file)
                except OSError:
                    pass

        result_id = session.next_result_id("mine_patterns")
        summary = {
            "result_id": result_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "surviving_terms_count": len(surviving_terms),
            "patterns_count": len(mined_patterns),
            "patterns": mined_patterns,
        }

        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
        session.artifacts[artifact_key] = mined_patterns

        session.store_result(
            "mine_patterns_tool",
            effective_args,
            summary,
        )

        return summary

    return [mine_patterns_tool]
