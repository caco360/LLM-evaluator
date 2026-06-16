from pathlib import Path
import sqlite3

from jinja2 import Environment, FileSystemLoader, select_autoescape


BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "eval.db"
TEMPLATE_DIR = BASE_DIR / "templates"
REPORT_DIR = BASE_DIR / "reports"
TREND_RUN_LIMIT = 8


def get_runs(limit: int, newest_first: bool = True) -> list[dict]:
    order = "DESC" if newest_first else "ASC"
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            SELECT
                run_id,
                prompt_version,
                prompt_file,
                model,
                dataset_file,
                created_at,
                total_cases,
                success_count,
                error_count,
                overall_category_accuracy,
                overall_summary_score,
                total_tokens,
                avg_latency_ms
            FROM eval_runs
            ORDER BY created_at """ + order + """
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [dict(row) for row in rows]


def get_latest_run() -> dict | None:
    runs = get_runs(1)
    return runs[0] if runs else None


def get_latest_two_runs() -> tuple[dict, dict | None]:
    runs = get_runs(2)
    if not runs:
        raise RuntimeError("No eval runs found. Run an evaluation before generating a report.")
    latest = runs[0]
    previous = runs[1] if len(runs) > 1 else None
    return latest, previous


def get_trend_runs(limit: int = TREND_RUN_LIMIT) -> list[dict]:
    runs = get_runs(limit)
    return list(reversed(runs))


def get_case_results(run_id: str) -> list[dict]:
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            SELECT
                case_id,
                input,
                expected_category,
                expected_summary,
                actual_category,
                actual_summary,
                category_match,
                summary_score,
                latency_ms,
                total_tokens,
                status,
                error
            FROM eval_case_results
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchall()

    return [dict(row) for row in rows]


def format_percent(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.1%}"


def format_delta(value: float | int | None, percent: bool = False) -> str:
    if value is None:
        return "N/A"
    sign = "+" if value > 0 else ""
    if percent:
        return f"{sign}{value:.1%}"
    return f"{sign}{value:.2f}"


def build_scorecard(latest_run: dict, previous_run: dict | None) -> list[dict]:
    metrics = [
        ("Category Accuracy", "overall_category_accuracy", True, "higher"),
        ("Summary Score", "overall_summary_score", False, "higher"),
        ("Avg Latency ms", "avg_latency_ms", False, "lower"),
        ("Total Tokens", "total_tokens", False, "lower"),
    ]

    scorecard = []
    for label, key, is_percent, better_when in metrics:
        latest = latest_run.get(key)
        previous = previous_run.get(key) if previous_run else None
        delta = latest - previous if latest is not None and previous is not None else None
        is_good_delta = None
        if delta is not None:
            is_good_delta = delta > 0 if better_when == "higher" else delta < 0
        scorecard.append(
            {
                "label": label,
                "latest": latest,
                "previous": previous,
                "delta": delta,
                "is_percent": is_percent,
                "is_good_delta": is_good_delta,
            }
        )
    return scorecard


def find_category_regressions(previous_run: dict | None, latest_run: dict) -> list[dict]:
    if previous_run is None:
        return []

    previous_cases = {
        str(case["case_id"]): case
        for case in get_case_results(previous_run["run_id"])
    }
    latest_cases = {
        str(case["case_id"]): case
        for case in get_case_results(latest_run["run_id"])
    }

    regressions = []
    for case_id in sorted(set(previous_cases) & set(latest_cases)):
        previous_case = previous_cases[case_id]
        latest_case = latest_cases[case_id]

        if bool(previous_case["category_match"]) and not bool(latest_case["category_match"]):
            regressions.append(
                {
                    "case_id": case_id,
                    "input": latest_case["input"],
                    "expected_category": latest_case["expected_category"],
                    "previous_output": {
                        "category": previous_case["actual_category"],
                        "summary": previous_case["actual_summary"],
                    },
                    "latest_output": {
                        "category": latest_case["actual_category"],
                        "summary": latest_case["actual_summary"],
                    },
                }
            )

    return regressions


def generate_latest_run_report() -> Path:
    latest_run, previous_run = get_latest_two_runs()
    scorecard = build_scorecard(latest_run, previous_run)
    regressions = find_category_regressions(previous_run, latest_run)
    trend_runs = get_trend_runs(TREND_RUN_LIMIT)

    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html", "xml"]),
    )
    env.filters["percent"] = format_percent
    env.filters["delta"] = format_delta

    template = env.get_template("latest_run_report.html")
    html = template.render(
        run=latest_run,
        previous_run=previous_run,
        scorecard=scorecard,
        regressions=regressions,
        trend_runs=trend_runs,
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "latest_run_report.html"
    report_path.write_text(html, encoding="utf-8")

    return report_path


if __name__ == "__main__":
    path = generate_latest_run_report()
    print(f"Report generated: {path}")
