# Don't allocate memory in every apply

- STATUS: CLOSED
- PRIORITY: 100
- TAGS: performance

No description.

Closed on 2026-11-26: the Schwarz, coarse correction and two-level applies use preallocated members; the default multi-vector apply goes column by column through views.
