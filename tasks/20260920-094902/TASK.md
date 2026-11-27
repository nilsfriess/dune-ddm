# Ensure that R and A live on the same context

- STATUS: CLOSED
- PRIORITY: 100
- TAGS: backend

No description.

Closed on 2026-11-26: every MultiVec and LocalMat operation now checks that the backends match (DDM_CHECK ... backend mismatch).
