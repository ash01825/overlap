# Reopen not transitive

```yaml task
id: T0.01
title: Task One
phase: 0
status: done
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
evidence_observed:
  - "done"
commit:
  msg: "T0.01: one"
  sha: "1234"
strikes: 0
blocked_reason: ""
```

```yaml task
id: T0.02
title: Task Two
phase: 0
status: todo
needs: []
reopens: [T0.01]
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
  msg: "T0.02: two"
  sha: null
strikes: 0
blocked_reason: ""
```
