# Contribution and pull request collaboration

This guide governs changes to paco-observability: the PACO observability analysis software release; preserve the reproducibility scope, governed-input boundaries, and scientific-evidence limits in README.md. Read the applicable ancestor and repository `AGENTS.md`, existing contribution instructions, and relevant `.agents/skills` before editing. Preserve their project-specific constraints. If instructions or ownership conflict, coordinate with the user or responsible agent before touching the affected files.

## Isolate work and preserve the checkout

- Use one branch for each bug or feature. Keep each pull request small and focused; split unrelated fixes and separate operational changes from code review.
- Work in an independent worktree or isolated checkout. Do not patch a production checkout directly. Concurrent agents each need their own workspace and a declared file scope.
- Before editing, inspect the branch, tracked and staged changes, untracked paths, worktrees, and nested repository boundaries. Preserve all existing WIP, untracked files, and nested repositories, including ignored runtime data and browser state. Existing files are not disposable merely because Git does not track them.
- Do not automatically `reset`, `clean`, `stash`, overwrite, or absorb another task's changes. Do not switch or repoint a shared working branch. If the same file has existing WIP, or agent responsibilities overlap, coordinate first and base the work on an agreed snapshot.
- Use a small documentation-only branch or sparse isolated worktree for policy changes. Do not copy large datasets, model weights, dependencies, or generated outputs into a new workspace, and do not initialize Git for a pure experiment/data directory.

## Review and publication

- Publication requires the user's authorization. When publication is authorized, open a draft PR by default and keep it draft until its scope and evidence are ready for review.
- Describe the purpose and concrete problem, scope and excluded work, validation commands and their actual results, risks, and a practical rollback/revert plan. Identify skipped checks, missing inputs, and remaining limitations.
- A merge into the main/default branch requires review and explicit authorization. Merge approval and deployment approval are separate decisions; one does not authorize the other.
- This guide describes a collaboration workflow. It does not authorize pushing, merging, deployment, changing remote branch protection, or changing repository push permissions. It does not mean remote protection is enabled or enforcement/CI is working.
- Do not claim a policy is active in the default branch while it exists only on a local documentation branch. Report the branch/commit and a reviewable diff; integrate only the reviewed documentation change after authorization, using an isolated integration worktree if the main checkout has WIP.

## Validate honestly

Run the project's actual relevant tests, lint, and type checks for code changes, using the existing configuration and bounded inputs. Documentation-only work should receive an appropriate diff, content, and link check; do not run services or a heavy build solely for this guide. Record exactly what ran, the result, and any skipped or blocked checks. Never fabricate a successful test or infer local success from the presence of a CI file.

### Existing project commands and limits

- Existing unit tests: `pytest -q`; the default-branch `pyproject.toml` selects `tests/`.
- Existing CI syntax check: `python -m compileall -q scripts tests`; `.github/workflows/tests.yml` runs this and `pytest -q` on Python 3.10 and 3.12.
- Ruff is configured and listed in the existing development dependencies. When lint is relevant and the installed tool is available, use a bounded check such as `python -m ruff check scripts tests` with that configuration; report its result separately. The inspected workflow does not run Ruff.
- No standalone type-check command is defined in the inspected default-branch baseline.
- Synthetic software tests do not establish governed-data validity or scientific conclusions. Do not run the canonical analysis pipeline, download data, or regenerate manuscript figures solely for a documentation change.

## Operational and data boundaries

- Do not upgrade critical dependencies, migrate databases, restart services, deploy, or delete data without explicit authorization for that action. Documentation work does not authorize these operations.
- Do not commit secrets, credentials, private environment files, browser profiles/session state, datasets, model weights, caches, or generated artifacts. Review the exact staged file list and diff before committing; stage only the agreed task files.
- If an existing project policy needs a generated release asset, resolve that exception explicitly for the release; do not use it to collect unrelated artifacts in an ordinary PR.
- Keep runtime, data, dependency/reference repositories, and research/manuscript workspaces within their established boundaries. Preserve provenance and existing evidence claims; passing software checks is not scientific or production validation.

## Pull request description checklist

- Purpose and problem or requested behavior
- Scope, affected components, and excluded work
- Verification: exact commands, real outcomes, and skipped/blocked checks
- Risks and rollback/revert steps
- Review status, outstanding decisions, and separately authorized deployment steps
