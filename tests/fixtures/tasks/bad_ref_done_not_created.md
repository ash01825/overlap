# Ref by done task not on disk

```yaml task
id: T0.01
title: Task One
phase: 0
status: done
needs: []
reopens: []
files:
  - tests/fixtures/tasks/phantom_file_that_does_not_exist.txt
refs:
  - README.md
failure_modes:
  - "test"
done_when:
  - "test"
evidence:
  - "evidence"
evidence_observed:
  - "done evidence"
commit:
  msg: "T0.01: one"
  sha: "12345"
strikes: 0
blocked_reason: ""
```

```yaml task
id: T0.02
title: Task Two
phase: 0
status: todo
needs: [T0.01]
reopens: [T0.01]
files:
  - file_b.txt
refs:
  - tests/fixtures/tasks/phantom_file_that_does_not_exist.txt
failure_modes:
  - "test"
done_when:
  - "test"
evidence:
  - "evidence"
evidence_observed: []
commit:
  msg: "T0.02: two"
  sha: null
strikes: 0
blocked_reason: ""
```
