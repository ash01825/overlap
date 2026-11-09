# Unordered tasks with overlapping files

```yaml task
id: T0.01
title: Task One
phase: 0
status: todo
needs: []
reopens: []
files:
  - shared_module.py
refs:
  - README.md
failure_modes:
  - "test"
done_when:
  - "test"
evidence:
  - "evidence"
evidence_observed: []
commit:
  msg: "T0.01: one"
  sha: null
strikes: 0
blocked_reason: ""
```

```yaml task
id: T0.02
title: Task Two
phase: 0
status: todo
needs: []
reopens: []
files:
  - shared_module.py
refs:
  - README.md
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
