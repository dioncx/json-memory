🧪 [testing improvement] Cover SmartMemory.active_topics

**What:** Addressed the testing gap for the `active_topics` property in `SmartMemory`.
**Coverage:** The test covers empty state initialization, correct topic extraction and formatting (lowercasing), ordered insertion, and the 10-item limit.
**Result:** Increased test coverage and guarantees over the active topics feature which is responsible for returning the contextual topics for hybrid fallback scoring.
