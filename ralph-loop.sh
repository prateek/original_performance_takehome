#!/bin/bash
set -euo pipefail

# adapted from: https://gist.github.com/prateek/14fae59c71921710a3e055d74f30c8af
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <iterations>"
  exit 1
fi

if ! [[ "$1" =~ ^[0-9]+$ ]] || [[ "$1" -lt 1 ]]; then
  echo "Iterations must be a positive integer"
  exit 1
fi

TARGET_CYCLES_MIN="${TARGET_CYCLES_MIN:-200}"
TARGET_CYCLES_MAX="${TARGET_CYCLES_MAX:-500}"

PROMISE_FILE="I_PROMISE_ALL_TASKS_IN_THE_PRD_ARE_DONE_I_AM_NOT_LYING_I_SWEAR"
WORKLOG_FILE=".logs/ralph-worklog.md"
ITERATIONS_LOG=".logs/iterations.log"

ts_utc() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

repo_root() {
  git rev-parse --show-toplevel
}

ensure_pr_number() {
  if ! command -v gh >/dev/null 2>&1; then
    return 0
  fi

  local pr_number=""
  pr_number="$(gh pr view --json number -q .number 2>/dev/null || true)"
  if [[ -n "${pr_number}" ]]; then
    printf '%s' "${pr_number}"
    return 0
  fi

  gh pr create --fill --base main --head "$(git branch --show-current)" >/dev/null 2>&1 || true
  pr_number="$(gh pr view --json number -q .number 2>/dev/null || true)"
  printf '%s' "${pr_number}"
}

post_pr_comment() {
  local pr_number="${1}"
  local body_file="${2}"

  if [[ -z "${pr_number}" ]]; then
    return 0
  fi
  if ! command -v gh >/dev/null 2>&1; then
    return 0
  fi

  gh pr comment "${pr_number}" --body-file "${body_file}" >/dev/null 2>&1 || true
}

update_pr_body_status() {
  local pr_number="${1}"
  local status_file="${2}"

  if [[ -z "${pr_number}" ]]; then
    return 0
  fi
  if ! command -v gh >/dev/null 2>&1; then
    return 0
  fi

  local current_body=""
  current_body="$(gh pr view "${pr_number}" --json body -q .body 2>/dev/null || true)"
  if [[ -z "${current_body}" ]]; then
    return 0
  fi

  python -c 'import re, sys
status_path = sys.argv[1]
with open(status_path, "r", encoding="utf-8") as f:
    status_block = f.read().rstrip() + "\n"

body = sys.stdin.read()
start = "<!-- ralph-status:start -->"
end = "<!-- ralph-status:end -->"
pattern = re.compile(re.escape(start) + r".*?" + re.escape(end) + r"\\n?", re.DOTALL)

if pattern.search(body):
    body = pattern.sub(status_block, body, count=1)
else:
    body = body.rstrip() + "\\n\\n" + status_block

sys.stdout.write(body)
' "${status_file}" <<<"${current_body}" > .logs/.pr_body_next.txt

  gh pr edit "${pr_number}" --body-file .logs/.pr_body_next.txt >/dev/null 2>&1 || true
}

extract_ralph_log() {
  local iter_log="${1}"
  awk '
    $0 == "RALPH_LOG_START" {in_block=1; next}
    $0 == "RALPH_LOG_END" {in_block=0}
    in_block {print}
  ' "${iter_log}"
}

measure_cycles() {
  local out_file="${1}"
  python - <<'PY' 2>&1 | tee "${out_file}"
from tests.submission_tests import do_kernel_test

res = do_kernel_test(10, 16, 256)
print(f"RALPH_CYCLES {res}")
PY
}

git_push_if_possible() {
  # Avoid killing the loop if push fails (offline, auth, etc.).
  git push >/dev/null 2>&1 || true
}

cd "$(repo_root)"

mkdir -p .logs
rm -f "$PROMISE_FILE"

if [[ ! -f "${WORKLOG_FILE}" ]]; then
  cat > "${WORKLOG_FILE}" <<'EOF'
# Ralph worklog

Append-only log for `ralph-loop.sh` iterations.

- Each iteration should record: what it did, what it tried, what worked, what didn’t, and next steps.
- Raw Codex output is also saved per-iteration under `.logs/`.

EOF
fi

for ((i = 1; i <= $1; i++)); do
  start_ts="$(ts_utc)"
  branch="$(git branch --show-current)"
  sha_before="$(git rev-parse --short HEAD)"
  pr_number="$(ensure_pr_number)"

  iter_codex_log=".logs/iteration-${i}.codex.log"
  iter_cycles_log=".logs/iteration-${i}.cycles.log"

  cat > .logs/.pr_comment_start.txt <<EOF
### Ralph iteration ${i} start

- Time (UTC): ${start_ts}
- Branch: \`${branch}\`
- Commit: \`${sha_before}\`
- Target cycles: ${TARGET_CYCLES_MIN}–${TARGET_CYCLES_MAX}
EOF
  post_pr_comment "${pr_number}" .logs/.pr_comment_start.txt

  codex_exit=0
  set +e
  codex --dangerously-bypass-approvals-and-sandbox exec <<'EOF' 2>&1 | tee -a "${ITERATIONS_LOG}" | tee "${iter_codex_log}"
1. Pick the single highest-priority task in PRD.md (use progress.md for context) and implement it.
2. Run `python tests/submission_tests.py` (or equivalent) and record the current cycle count.
3. Update PRD.md with what was done.
4. Append a short entry to progress.md.
5. Commit your changes.
ONLY WORK ON A SINGLE TASK.

Do not modify anything under tests/.

At the very end of your response, include a short, structured summary between the exact markers:
RALPH_LOG_START
Did: ...
Tried: ...
Worked: ...
Didn’t: ...
Next: ...
RALPH_LOG_END

If the PRD is complete, and there are NO tasks left, then and only then touch a file named I_PROMISE_ALL_TASKS_IN_THE_PRD_ARE_DONE_I_AM_NOT_LYING_I_SWEAR. Otherwise respond with a brief summary of changes/progress.
EOF
  codex_exit=$?
  set -e

  git_push_if_possible

  cycles=""
  if measure_cycles "${iter_cycles_log}" >/dev/null 2>&1; then
    cycles="$(awk '/^RALPH_CYCLES /{print $2}' "${iter_cycles_log}" | tail -n1)"
  fi

  sha_after="$(git rev-parse --short HEAD)"
  end_ts="$(ts_utc)"

  ralph_summary="$(extract_ralph_log "${iter_codex_log}" || true)"
  if [[ -z "${ralph_summary}" ]]; then
    ralph_summary="(No structured Ralph summary found in ${iter_codex_log})"
  fi

  {
    echo "## Iteration ${i} (${start_ts} → ${end_ts})"
    echo ""
    echo "- Branch: \`${branch}\`"
    echo "- Commit: \`${sha_before}\` → \`${sha_after}\`"
    echo "- Codex exit: ${codex_exit}"
    echo "- Target cycles: ${TARGET_CYCLES_MIN}–${TARGET_CYCLES_MAX}"
    if [[ -n "${cycles}" ]]; then
      echo "- Measured cycles: ${cycles}"
    else
      echo "- Measured cycles: (failed; see \`${iter_cycles_log}\`)"
    fi
    echo "- Codex log: \`${iter_codex_log}\`"
    echo "- Cycles log: \`${iter_cycles_log}\`"
    echo ""
    echo "### Summary"
    echo ""
    echo "${ralph_summary}"
    echo ""
  } >> "${WORKLOG_FILE}"

  cat > .logs/.pr_comment_end.txt <<EOF
### Ralph iteration ${i} end

- Time (UTC): ${end_ts}
- Branch: \`${branch}\`
- Commit: \`${sha_after}\`
- Codex exit: ${codex_exit}
- Measured cycles: ${cycles:-"(failed; see local ${iter_cycles_log})"}
- Target cycles: ${TARGET_CYCLES_MIN}–${TARGET_CYCLES_MAX}

${ralph_summary}
EOF
  post_pr_comment "${pr_number}" .logs/.pr_comment_end.txt

  cat > .logs/.pr_status_block.txt <<EOF
<!-- ralph-status:start -->
### Ralph status
- Target cycles: ${TARGET_CYCLES_MIN}–${TARGET_CYCLES_MAX}
- Latest measured cycles: ${cycles:-"(unknown)"} (commit \`${sha_after}\`, UTC ${end_ts})
- Worklog (local): \`${WORKLOG_FILE}\`
<!-- ralph-status:end -->
EOF
  update_pr_body_status "${pr_number}" .logs/.pr_status_block.txt

  if [[ -f "$PROMISE_FILE" ]]; then
    echo "PRD complete after $i iterations."
    exit 0
  fi
done

echo "PRD not complete after $1 iterations."
exit 1
