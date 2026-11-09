# Duplicate ID

```yaml task
id: T0.01
title: Task One
phase: 0
status: todo
needs: []
reopens: []
files:
  - file_a.txt
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
id: T0.01
title: Task One Duplicate
phase: 0
status: todo
needs: []
reopens: []
files:
  - file_b.txt
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
  msg: "T0.01: one dup"
  sha: null
strikes: 0
blocked_reason: ""
```
