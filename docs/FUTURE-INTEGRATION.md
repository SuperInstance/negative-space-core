# Future Integration: negative-space-core

## Current State

negative-space-core implements the foundational theory that "intelligence is what you learn to AVOID, not what you choose." Built on the empirically observed 294:1 avoid:choose ratio (`AVOIDANCE_RATIO`), it provides `AvoidanceTracker` for recording agent decisions, `ConservationLaw` verifying the avoidance ratio holds across population scales, `BatchAnalyzer` producing `BatchStatistics` across agent populations, `FeedbackLoop` for learning from avoidance patterns, `InferenceEngine` for deducing from gaps, and `NegativeSpace` representing unexplored regions. `Decision` enum captures Avoid/Choose with labels.

## Integration Opportunities

### "Communication Hurts Fitness" → Fleet Protocol Design

The core finding that communication hurts fitness (avoidance dominance) has profound implications for fleet design. If agents that communicate less perform better (because they avoid more, and avoidance is 294x more informative than choosing), then:

- **I2I messages should be sparse**: `ternary-protocol::MessageBus` should default to `Silence` (0). Only send `Signal` (+1) when the avoidance ratio drops below a threshold.
- **Ensign loading should be minimal**: Load only the ensigns that are absolutely needed. Every loaded ensign is a communication channel that reduces fitness.
- **Fleet coordination should be avoidance-driven**: Instead of broadcasting "what to do," broadcast "what to avoid." A fleet message that says "avoid the east quadrant" is 294x more valuable than "explore the west quadrant."

### ternary-cell → Avoidance-Driven Cell Death

The `ConservationLaw` maps directly to ternary-cell's `conservation()` phase. The `AVOIDANCE_RATIO = 294.0` constant determines the apoptosis threshold: if a cell's avoidance ratio drops below 294:1, it hasn't learned enough negative space and should be marked `CellState::Apoptotic`. The `BatchStatistics::conforms_to_law()` method becomes the fleet health check: if the fleet-wide avoidance ratio deviates from 294:1, the fleet is communicating too much and losing fitness.

### ternary-inference → Gap-Based Knowledge Discovery

`InferenceEngine` (from negative-space-core) and `DeductionEngine` (from ternary-inference) are complementary:

- `negative-space-core::InferenceEngine` infers from avoidance patterns across agents
- `ternary-inference::DeductionEngine` infers from gaps in ternary spaces
- Combined: use avoidance patterns to identify which gaps are most informative, then apply `DeductionEngine`'s analogy/interpolation/exclusion rules to those gaps

### ternary-science → Empirical Validation

`ternary-science`'s 51 tests validate negative space theory across multiple dimensions. The `laws` module proves conservation laws experimentally. `species` identifies universal strategy species from GPU runs. `cross_validation` confirms results across Python/Rust/C/WASM. This empirical backing makes negative-space-core the theoretically grounded foundation — every other crate's design should respect the 294:1 ratio.

## Potential in Mature Systems

negative-space-core becomes the "physics" of the ternary ecosystem. Every agent, every cell, every room obeys the avoidance conservation law. Fleet metrics track `AVOIDANCE_RATIO` in real-time. When the ratio drops (agents communicating too much), the fleet automatically reduces message frequency. When the ratio is healthy, the fleet is learning efficiently. The `FeedbackLoop` mechanism enables self-correction: agents that deviate from the conservation law get feedback to increase avoidance.

## Cross-Pollination Ideas

- **Avoidance ratio → ternary-attention**: Weight attention toward avoidance signals. An attention head that attends 294x more to negative space than to positive space naturally learns the avoidance-dominant representation.
- **Conservation law → ternary-econ portfolio**: Portfolio optimization that respects avoidance conservation. Allocate 294 units to "avoid" assets for every 1 unit allocated to "choose" assets.
- **Negative space → ternary-steganography**: Hide data in the avoidance regions of ternary spaces. The 294:1 ratio means there's vastly more avoidance space to embed in.

## Dependencies for Next Steps

1. Integration with `ternary-cell` conservation phase (apoptosis threshold)
2. Fleet-wide avoidance ratio monitoring via `BatchAnalyzer`
3. `InferenceEngine` → `ternary-inference::DeductionEngine` bridge
4. Communication policy: reduce message frequency when avoidance ratio drops
5. `ternary-science` test suite as continuous validation in CI
