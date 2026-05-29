# Project Instructions

This project is a model regression detection system for an LLM-powered customer support email classifier. The goal is to test prompt and model changes against a golden dataset before they reach users.

## Why This Project Matters

AI teams often change prompts or models without a strong safety net. This project treats prompt and model behavior like production code: every change should be tested, compared against a baseline, and blocked or flagged if it causes quality regressions.

The interview signal is that this project focuses on what happens after an LLM feature ships: evaluation quality, regression detection, alerting, and operational readiness.

## Tech Stack

| Component | Tool / Library | Why |
| --- | --- | --- |
| Language | Python 3.11+ | Standard language for ML and evaluation tooling. |
| LLM provider | OpenAI API, `gpt-4o` / `gpt-4o-mini` | Widely recognized and easy to swap later. |
| Eval framework | Custom eval logic, optionally RAGAS or DeepEval later | Shows understanding beyond simple accuracy checks. |
| Data storage | JSON files + SQLite | JSON for human-labeled golden data; SQLite for generated eval runs and history. |
| Alerting | Slack Webhooks | Matches how real teams receive CI and quality alerts. |
| Scheduling / CI | GitHub Actions | Runs on PRs and can block risky prompt changes. |
| Visualization | Simple HTML report or Streamlit | Gives quick diff views and trend inspection. |
| Containerization | Docker | Shows production readiness and repeatable execution. |

## LLM Feature Under Test

Build and maintain a customer support email classifier.

The classifier reads one customer email and returns structured JSON with:

```json
{
  "category": "billing | technical | account | general | irrelevant",
  "summary": "one sentence summary"
}
```

Supported categories:

- `billing`: payments, invoices, refunds, charges, pricing, subscriptions, coupons, or receipts.
- `technical`: bugs, errors, broken product behavior, outages, integrations, performance issues, or broken buttons/features.
- `account`: login, password, verification, profile settings, permissions, ownership, security, or account deletion.
- `general`: product/company questions, feedback, support-contact questions, sales questions, or requests that do not clearly fit billing, technical, or account.
- `irrelevant`: spam, unrelated text, personal requests, keyboard mashes, or messages that are not customer support requests.

The feature should use the prompt and model from a YAML prompt file, not hardcoded values.

## Prompt Versioning

Prompts live in the `prompts/` directory as versioned YAML files.

Each prompt file should follow this layout:

```yaml
version_id: support_classifier_v1_balanced
created_at: "2026-05-28T00:00:00-04:00"
description: Short explanation of what this prompt is trying to improve.
model: gpt-4o-mini
system_prompt: |
  Prompt instructions go here.
few_shot_examples:
  - input: "Customer email example"
    output:
      category: billing
      summary: "Expected one-sentence summary."
```

The prompt files are treated like code. Any prompt change should be testable by the eval pipeline.

## Interface Contract

Keep the interface typed with Pydantic.

The prompt config should represent the YAML prompt file:

- `version_id`
- `created_at`
- `description`
- `model`
- `system_prompt`
- `few_shot_examples`

The classifier output should validate:

- `category`: one of `billing`, `technical`, `account`, `general`, `irrelevant`
- `summary`: string

## Golden Dataset

The golden dataset is stored as JSON at:

```text
src/data/golden_dataset.json
```

The file is a single JSON array. Each dataset entry repeats this shape:

```json
{
  "id": 1,
  "input": "Customer email text",
  "expected_output": {
    "category": "billing",
    "summary": "Human-written expected summary."
  },
  "difficulty": "easy",
  "note": "Why this case matters or how it should be interpreted."
}
```

Allowed difficulty values:

- `easy`
- `medium`
- `hard`
- `edge`

The expected category must always be one of the five supported classifier categories.

The golden dataset should be human-labeled. Public datasets can be used for inspiration, but final entries should be manually reviewed and adapted to this project's taxonomy.

## Evaluation Goal

The eval pipeline should eventually:

1. Load a prompt YAML file.
2. Load the golden dataset JSON file.
3. Run each `input` through the classifier.
4. Compare the model output to `expected_output`.
5. Track category accuracy, summary quality, latency, token usage, and invalid JSON failures.
6. Compare new runs against previous runs to detect regressions.

## Regression Rules

Later phases should flag regressions such as:

- category accuracy drop
- cases that moved from pass to fail
- invalid JSON outputs
- worse summary quality
- latency or token usage increases
- slow quality drift across multiple runs

Use configurable thresholds for warnings and critical failures.

## Reporting And CI

Later phases should generate an HTML diff report and integrate with CI.

The intended CI behavior:

- Run evals when files in `prompts/` change.
- Compare the new prompt behavior against a baseline.
- Produce a readable report.
- Alert Slack when regressions are detected.
- Block merges only for critical regressions.

## Build Roadmap

### Phase 1: Define The LLM Feature Under Test

Current implementation should focus on:

1. Build a customer support email classifier that reads an email and returns one category plus a one-sentence summary.
2. Use five categories: `billing`, `technical`, `account`, `general`, and `irrelevant`.
3. Keep the prompt configurable by loading it from a YAML file in `prompts/`.
4. Use the model value from the YAML file instead of hardcoding it in the feature function.
5. Keep the interface typed with Pydantic:
   - `PromptConfig` for the prompt YAML contract.
   - `EmailClassificationOutput` for the model response contract.

Phase 1 is complete when a prompt YAML file and an email input can produce validated structured JSON.

### Phase 2: Build The Golden Dataset

Current dataset decision:

1. Store the golden dataset as a JSON array at `src/data/golden_dataset.json`.
2. Each test case should use this shape:

```json
{
  "id": 1,
  "input": "Customer email text",
  "expected_output": {
    "category": "billing",
    "summary": "Human-written expected summary."
  },
  "difficulty": "easy",
  "note": "Why this case matters or how it should be interpreted."
}
```

3. Use the five supported categories in `expected_output.category`.
4. Use these difficulty values: `easy`, `medium`, `hard`, `edge`.
5. Curate 50-100 realistic examples by hand across all five categories.
6. Include edge cases deliberately: short text, typos, mixed language, sarcasm, angry messages, ambiguous billing/technical cases, irrelevant text, and empty or near-empty inputs.
7. Keep the summaries human-written and one sentence.

Interview talking point: explain that the dataset is hand-labeled, privacy-safe, and designed around both standard cases and failure-prone edge cases. Evaluation quality is bounded by data quality.

### Phase 3: Build The Evaluation Engine

1. Create a runner that accepts a `PromptConfig` and the golden dataset.
2. Run every test case through the classifier.
3. Store raw model output, parsed output, category match, latency, token usage, and errors.
4. Add multi-dimensional scoring:
   - exact category match
   - valid JSON
   - summary relevance, possibly with an LLM-as-judge
   - latency
   - token usage
5. Use async batching later to reduce runtime and cost.
6. Store generated run results in SQLite, not in the golden dataset file.

### Phase 4: Build Comparison, Reporting, And Alerting

1. Compare each eval run against the previous run or a selected baseline.
2. Calculate:
   - overall pass rate delta
   - per-category accuracy delta
   - regressions, meaning cases that flipped from pass to fail
   - improvements, meaning cases that flipped from fail to pass
3. Use thresholds:
   - warning when quality drops more than a configurable small threshold, such as 3%
   - critical when quality drops more than a configurable larger threshold, such as 8%
4. Generate an HTML report with:
   - run metadata
   - prompt version
   - model
   - scorecard
   - regressed cases
   - old vs new outputs
   - trend chart over recent runs
5. Send Slack alerts through incoming webhooks with pass/warn/fail status and headline numbers.
6. Add slow drift detection using a rolling average across recent runs.

### Phase 5: Wire Into CI/CD

1. Add a GitHub Actions workflow.
2. Trigger evals when files in `prompts/` change.
3. Run the eval pipeline in CI.
4. Generate the report as a CI artifact.
5. Post a PR summary comment with pass/warn/fail status.
6. Fail the workflow when critical regressions are detected.
7. Add a Dockerfile that packages the eval runner, prompt files, golden dataset, and reporting code.
8. Use environment variables for:
   - `OPENAI_API_KEY`
   - `SLACK_WEBHOOK_URL`
   - regression thresholds

### Phase 6: Polish For Portfolio

1. Write a README that reads like internal teammate onboarding documentation.
2. Include setup instructions, dataset instructions, threshold configuration, and architecture decisions.
3. Record a short walkthrough showing:
   - prompt change
   - eval run
   - regression report
   - Slack alert
4. Add a short project writeup explaining:
   - the problem of blind prompt changes
   - the CI/CD approach for model behavior
   - one design decision, such as tracking slow drift separately from per-run regressions.
