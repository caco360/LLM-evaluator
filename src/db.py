import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "runs" / "eval.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

con = sqlite3.connect(DB_PATH)
cur = con.cursor()

def initDB():
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS eval_runs (
            run_id TEXT PRIMARY KEY,
            prompt_version TEXT NOT NULL,
            prompt_file TEXT NOT NULL,
            model TEXT NOT NULL,
            dataset_file TEXT NOT NULL,
            created_at TEXT NOT NULL,
            total_cases INTEGER NOT NULL,
            success_count INTEGER NOT NULL,
            error_count INTEGER NOT NULL,
            overall_category_accuracy REAL,
            overall_summary_score REAL,
            total_tokens INTEGER,
            avg_latency_ms REAL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS eval_case_results (
            run_id TEXT NOT NULL,
            case_id TEXT NOT NULL,
            input TEXT NOT NULL,
            expected_category TEXT NOT NULL,
            expected_summary TEXT NOT NULL,
            actual_category TEXT,
            actual_summary TEXT,
            raw_output TEXT,
            category_match INTEGER,
            summary_score INTEGER,
            latency_ms REAL,
            input_tokens INTEGER,
            output_tokens INTEGER,
            total_tokens INTEGER,
            status TEXT NOT NULL,
            error TEXT,
            PRIMARY KEY (run_id, case_id),
            FOREIGN KEY (run_id) REFERENCES eval_runs(run_id)
        )
        """
    )
    con.commit()

def insert_run(summary:dict)->None:
    cur.execute(
        """
        INSERT INTO eval_runs
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
        summary["run_id"],
        summary["prompt_version"],
        summary["prompt_file"],
        summary["model"],
        summary["dataset_file"],
        summary["created_at"],
        summary["total_cases"],
        summary["success_count"],
        summary["error_count"],
        summary["overall_category_accuracy"],
        summary["overall_summary_score"],
        summary["total_tokens"],
        summary["avg_latency_ms"],
        )
    )
    con.commit()


def insert_case_results(run_id: str, case_results: list[dict]) -> None:
    for case_result in case_results:
        expected_output = case_result["expected_output"]
        model_output = case_result.get("model_output") or {}
        usage = case_result.get("usage") or {}

        cur.execute(
            """
            INSERT INTO eval_case_results
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                str(case_result["case_id"]),
                case_result["input"],
                expected_output["category"],
                expected_output["summary"],
                model_output.get("category"),
                model_output.get("summary"),
                case_result.get("raw_output"),
                int(case_result["category_match"]) if "category_match" in case_result else None,
                case_result.get("summary_score"),
                case_result.get("latency_ms"),
                usage.get("input_tokens"),
                usage.get("output_tokens"),
                usage.get("total_tokens"),
                case_result["status"],
                case_result.get("error"),
            )
        )
    con.commit()

def get_last_two_runs() -> list[dict]:
    res = con.execute("""
    SELECT run_id, created_at, overall_category_accuracy, overall_summary_score, total_tokens, avg_latency_ms
    FROM eval_runs
    ORDER BY created_at DESC
    LIMIT 2
""")
    return [
        {
            "run_id": row[0],
            "created_at": row[1],
            "overall_category_accuracy": row[2],
            "overall_summary_score": row[3],
            "total_tokens": row[4],
            "avg_latency_ms": row[5],
        }
        for row in res.fetchall()
    ]


def get_latest_run() -> dict | None:
    res = con.execute("""
    SELECT run_id, prompt_version, created_at, overall_category_accuracy, overall_summary_score, total_tokens, avg_latency_ms
    FROM eval_runs
    ORDER BY created_at DESC
    LIMIT 1
""")
    row = res.fetchone()
    if row is None:
        return None

    return {
        "run_id": row[0],
        "prompt_version": row[1],
        "created_at": row[2],
        "overall_category_accuracy": row[3],
        "overall_summary_score": row[4],
        "total_tokens": row[5],
        "avg_latency_ms": row[6],
    }


def get_latest_run_for_prompt_version(prompt_version: str, exclude_run_id: str | None = None) -> dict | None:
    if exclude_run_id:
        res = con.execute("""
        SELECT run_id, prompt_version, created_at, overall_category_accuracy, overall_summary_score, total_tokens, avg_latency_ms
        FROM eval_runs
        WHERE prompt_version = ? AND run_id != ?
        ORDER BY created_at DESC
        LIMIT 1
    """, (prompt_version, exclude_run_id))
    else:
        res = con.execute("""
        SELECT run_id, prompt_version, created_at, overall_category_accuracy, overall_summary_score, total_tokens, avg_latency_ms
        FROM eval_runs
        WHERE prompt_version = ?
        ORDER BY created_at DESC
        LIMIT 1
    """, (prompt_version,))

    row = res.fetchone()
    if row is None:
        return None

    return {
        "run_id": row[0],
        "prompt_version": row[1],
        "created_at": row[2],
        "overall_category_accuracy": row[3],
        "overall_summary_score": row[4],
        "total_tokens": row[5],
        "avg_latency_ms": row[6],
    }
