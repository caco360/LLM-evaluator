from pathlib import Path
import sqlite3

import pandas as pd
import streamlit as st


DB_PATH = Path(__file__).parent / "eval.db"


@st.cache_data
def load_runs() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as con:
        return pd.read_sql_query(
            """
            SELECT
                run_id,
                created_at,
                prompt_version,
                model,
                dataset_file,
                total_cases,
                success_count,
                error_count,
                overall_category_accuracy,
                overall_summary_score,
                total_tokens,
                avg_latency_ms
            FROM eval_runs
            ORDER BY created_at ASC
            """,
            con,
        )


@st.cache_data
def load_case_results(run_id: str) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as con:
        return pd.read_sql_query(
            """
            SELECT
                case_id,
                expected_category,
                actual_category,
                category_match,
                summary_score,
                latency_ms,
                total_tokens,
                status,
                error
            FROM eval_case_results
            WHERE run_id = ?
            ORDER BY case_id
            """,
            con,
            params=(run_id,),
        )


def format_percent(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.1%}"


st.set_page_config(page_title="Model Regression Dashboard", layout="wide")
st.title("Model Regression Dashboard")

if not DB_PATH.exists():
    st.warning(f"No database found at {DB_PATH}. Run an eval first.")
    st.stop()

runs = load_runs()

if runs.empty:
    st.info("No eval runs found yet.")
    st.stop()

latest = runs.iloc[-1]

metric_cols = st.columns(4)
metric_cols[0].metric("Latest Accuracy", format_percent(latest["overall_category_accuracy"]))
metric_cols[1].metric("Avg Summary Score", round(latest["overall_summary_score"], 2))
metric_cols[2].metric("Total Tokens", int(latest["total_tokens"]))
metric_cols[3].metric("Avg Latency", f"{round(latest['avg_latency_ms'], 2)} ms")

st.subheader("Accuracy Over Time")
chart_data = runs[["created_at", "overall_category_accuracy"]].copy()
chart_data["created_at"] = pd.to_datetime(chart_data["created_at"], errors="coerce")
chart_data = chart_data.dropna(subset=["created_at"]).set_index("created_at")
st.line_chart(chart_data)

st.subheader("Recent Runs")
recent_runs = runs.sort_values("created_at", ascending=False).head(5).copy()
recent_runs["overall_category_accuracy"] = recent_runs["overall_category_accuracy"].map(format_percent)
recent_runs["overall_summary_score"] = recent_runs["overall_summary_score"].round(2)
recent_runs["avg_latency_ms"] = recent_runs["avg_latency_ms"].round(2)
st.dataframe(
    recent_runs[
        [
            "created_at",
            "run_id",
            "prompt_version",
            "model",
            "total_cases",
            "success_count",
            "error_count",
            "overall_category_accuracy",
            "overall_summary_score",
            "total_tokens",
            "avg_latency_ms",
        ]
    ],
    use_container_width=True,
    hide_index=True,
)

st.subheader("Run Details")
selected_run_id = st.selectbox(
    "Select a run",
    options=runs.sort_values("created_at", ascending=False)["run_id"].tolist(),
)

case_results = load_case_results(selected_run_id)

if case_results.empty:
    st.info("No case results found for this run.")
else:
    successful_cases = case_results[case_results["status"] == "success"].copy()

    if successful_cases.empty:
        st.warning("This run has no successful cases to break down.")
    else:
        category_breakdown = (
            successful_cases.groupby("expected_category")
            .agg(
                cases=("case_id", "count"),
                accuracy=("category_match", "mean"),
                avg_summary_score=("summary_score", "mean"),
                avg_latency_ms=("latency_ms", "mean"),
            )
            .reset_index()
        )
        category_breakdown["accuracy"] = category_breakdown["accuracy"].map(format_percent)
        category_breakdown["avg_summary_score"] = category_breakdown["avg_summary_score"].round(2)
        category_breakdown["avg_latency_ms"] = category_breakdown["avg_latency_ms"].round(2)

        st.caption("Per-category accuracy helps reveal regressions hidden by the overall score.")
        st.dataframe(category_breakdown, use_container_width=True, hide_index=True)

    failed_cases = case_results[case_results["category_match"] == 0]
    with st.expander(f"Failed Category Matches ({len(failed_cases)})"):
        st.dataframe(failed_cases, use_container_width=True, hide_index=True)
