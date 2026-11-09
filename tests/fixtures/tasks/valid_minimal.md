# Minimal Valid Task DAG

```yaml task
id: T0.01
title: First task
phase: 0
status: done
needs: []
reopens: []
files:
  - file_a.txt
refs:
  - README.md
failure_modes:
  - "first failure mode"
done_when:
  - "first done when"
evidence:
  - "first evidence"
evidence_observed:
  - "observed evidence"
commit:
  msg: "T0.01: first task"
  sha: "abc1234"
strikes: 0
blocked_reason: ""
```

```yaml task
id: T0.02
title: Second task
phase: 0
status: todo
needs: [T0.01]
reopens: [T0.01]
files:
  - file_b.txt
refs:
  - README.md
failure_modes:
  - "second failure mode"
done_when:
  - "second done when"
evidence:
  - "second evidence"
evidence_observed: []
commit:
  msg: "T0.02: second task"
  sha: null
strikes: 0
blocked_reason: ""
```
