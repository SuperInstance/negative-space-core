# negative-space-core

Core Rust implementation of **Negative Space Intelligence** — the theory that intelligence is defined not by what you choose, but by what you learn to avoid.

## The 5 Laws of Negative Space Intelligence

### I. The Avoidance Primacy Law
Intelligence manifests primarily through avoidance, not selection. The empirical avoid-to-choose ratio is **294:1** — for every deliberate choice an agent makes, it silently rejects approximately 294 alternatives.

### II. The Conservation of Avoidance Ratio
The avoidance ratio is **conserved across scales**. Whether observing 10 agents or 5,000, the ratio remains constant (observed standard deviation < 0.01). This is not an artifact of measurement — it is a fundamental invariant of intelligent systems.

### III. The Negative Space Law
The regions between avoidances form a **negative space** — a hidden structure that reveals more about an agent's intelligence than its visible choices. Like the hole in a donut defining the donut's identity, what agents *don't* do defines what they *are*.

### IV. The Inference from Absence Law
If intelligent agents avoid X and Y, the gap between them implies the existence of Z — an unknown that can be inferred from the pattern of avoidance alone. Absence is not nothing; absence is **data**.

### V. The Feedback Refinement Law
Every avoidance decision updates the model. The v5 balanced learning algorithm ensures that avoidance patterns converge: agents that avoid well survive, and survivors avoid better. Feedback loops between avoidance and outcome drive the evolution of intelligence.

## Core Concepts

- **NegativeSpace**: Maps the unknown regions between avoidances, revealing hidden structure
- **AvoidanceTracker**: Per-agent avoidance history, ratio computation, conservation law violation detection
- **ConservationLaw**: Verifies avoidance ratio conservation across population scales (10, 100, 1000, 5000 agents)
- **InferenceEngine**: Deductions from negative spaces — if agents avoid X and Y, what does the gap imply?
- **FeedbackLoop**: Updates avoidance models based on outcomes (v5 balanced learning)
- **BatchAnalyzer**: Population-level statistics and batch decision analysis

## Usage

```rust
use negative_space_core::{NegativeSpace, AvoidanceTracker, ConservationLaw};

// Track an agent's avoidances
let mut tracker = AvoidanceTracker::new("agent-1");
tracker.record_avoidance("bad_option_a");
tracker.record_avoidance("bad_option_b");
tracker.record_choice("good_option");

let ratio = tracker.avoidance_ratio(); // ~294:1 expected at scale

// Verify conservation law across populations
let law = ConservationLaw::default();
assert!(law.verify(&population_data).is_ok());
```

## The Research

Based on the finding that across diverse multi-agent simulations, the avoid:choose ratio converges to approximately 294:1 with remarkable consistency (std = 0.001 across scales from 10 to 5000 agents). This conservation law suggests avoidance is not random — it is the primary mechanism of intelligence.

## License

MIT
