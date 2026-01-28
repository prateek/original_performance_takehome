#!/bin/bash
set -e

# adapted from: https://gist.github.com/prateek/14fae59c71921710a3e055d74f30c8af
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <iterations>"
  exit 1
fi

if ! [[ "$1" =~ ^[0-9]+$ ]] || [[ "$1" -lt 1 ]]; then
  echo "Iterations must be a positive integer"
  exit 1
fi

PROMISE_FILE="I_PROMISE_ALL_TASKS_IN_THE_PRD_ARE_DONE_I_AM_NOT_LYING_I_SWEAR"

mkdir -p .logs
rm -f "$PROMISE_FILE"

for ((i = 1; i <= $1; i++)); do
  codex --dangerously-bypass-approvals-and-sandbox exec <<'EOF' 2>&1 | tee -a ".logs/iterations.log"
1. Find the highest-priority task based on PRD.md and progress.md, and implement it.
2. Run your tests and type checks.
3. Update the PRD with what was done.
4. Append your progress to progress.txt.
5. Commit your changes.
ONLY WORK ON A SINGLE TASK.

If the PRD is complete, and there are NO tasks left, then and only then touch a file named I_PROMISE_ALL_TASKS_IN_THE_PRD_ARE_DONE_I_AM_NOT_LYING_I_SWEAR. Otherwise respond with a brief summary of changes/progress.
EOF

  if [[ -f "$PROMISE_FILE" ]]; then
    echo "PRD complete after $i iterations."
    exit 0
  fi
done

echo "PRD not complete after $1 iterations."
exit 1
