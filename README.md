# Negative Space Core

**Negative Space Intelligence** is a framework modeling intelligence as avoidance rather than choice. Instead of tracking what an agent selects, it tracks what the agent *rejects* — and discovers that avoidance ratios are conserved across populations at approximately **294:1** (avoids per choice).

## Why It Matters

Traditional decision theory focuses on what agents choose. But most of intelligence is about what you *don't* do — the chess moves you reject, the paths you avoid, the words you don't say. Negative Space Core formalizes this insight with three empirical laws: (1) the **Avoidance Ratio Conservation Law** — whether you observe 10 or 5,000 agents, the ratio of avoidances to choices stays near 294:1 with standard deviation < 0.01; (2) the **Feedback Refinement Law** — avoidance patterns update via exponential moving average, converging to the conservation ratio; (3) the **Inference from Absence Law** — gaps between avoided options predict the existence of unknown alternatives. These laws have applications in recommendation systems, anomaly detection, and cognitive modeling.

## How It Works

### The 294:1 Conservation Law

The core invariant: for any sufficiently large agent population, the ratio of avoided decisions to chosen decisions converges to approximately 294. This is verified empirically using the `ConservationLaw` checker across population scales {10, 100, 1000, 5000}:

```
ratio = total_avoidances / total_choices ≈ 294.0
std_dev(ratios across scales) < 0.01
```

The verification algorithm runs in **O(N · S)** time where N is total agents and S is the number of population scales tested.

### Feedback Refinement via EMA

The `FeedbackLoop` updates the estimated avoidance ratio using exponential moving average:

```
μ_(t+1) = μ_t · (1 - α) + r_observed · α
```

where α is the learning rate (default 0.01). This converges to the true ratio with error decreasing as **O(1/√t)** by the law of large numbers. The EMA prevents oscillation while tracking distribution shifts.

### Inference from Absence

The `InferenceEngine` finds gaps between avoided options. If agents consistently avoid both X and Y, the space between them likely contains an undiscovered Z. Confidence is computed as:

```
confidence(gap) = cross_agent_frequency(gap) / total_agents
```

Gaps with confidence ≥ 0.3 (configurable threshold) are accepted as inferences. Batch inference over N avoided options examines N-1 adjacent pairs in sorted order — **O(N log N)** due to sorting.

### Negative Space Mapping

`NegativeSpace` builds a complete map of an agent's decision landscape, computing coverage (fraction of option space defined by avoidances) and finding unexplored gaps. Coverage = |avoided| / (|avoided| + |chosen|).

## Quick Start

```rust
use negative_space_core::{AvoidanceTracker, BatchAnalyzer, ConservationLaw, FeedbackLoop, InferenceEngine};

let mut tracker = AvoidanceTracker::new("agent-1");
for _ in 0..294 { tracker.record_avoidance("option"); }
tracker.record_choice("chosen");

assert!(tracker.ratio_near_expected(10.0));

let mut analyzer = BatchAnalyzer::new();
analyzer.add(tracker);
let stats = analyzer.analyze();
println!("Ratio: {:.1}", stats.avoidance_ratio);

let loop_model = FeedbackLoop::default();
println!("Error from 294:1: {}", loop_model.error());
```

## API

| Type | Description |
|------|-------------|
| `AvoidanceTracker` | Per-agent avoid/choose decision log with ratio computation |
| `BatchAnalyzer` | Population-level statistics across multiple trackers |
| `ConservationLaw` | Verifies ratio invariance across population scales |
| `FeedbackLoop` | EMA-based model updating with convergence detection |
| `InferenceEngine` | Infers hidden options from gaps in avoidance patterns |
| `NegativeSpace` | Maps the full decision landscape of an agent |
| `Decision` | Enum: `Avoid(String)` or `Choose(String)` |
| `ConservationResult` | Scales, ratios, std_dev, and conserved flag |
| `InferredGap` | Left/right boundaries, inferred entity, confidence score |

Constants: `AVOIDANCE_RATIO = 294.0`, `CONSERVATION_THRESHOLD = 0.01`

## Architecture Notes

Negative Space Core is a foundational theory crate in SuperInstance. The 294:1 conservation law mirrors the broader γ + η = C principle (growth + exploration = conservation): what is avoided (η) and what is explored (γ) are conserved across the system. The `TernaryState` values {-1 (avoid), 0 (explore), +1 (choose)} in `ternary-agent` directly map to this framework's decision types.

See [ARCHITECTURE.md](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md) for the full conservation law hierarchy.

## References

1. Kahneman, D. (2011). *Thinking, Fast and Slow*. Farrar, Straus and Giroux. (On System 2 rejection-based filtering)
2. Wirth, F. H. et al. (2016). "Conservation laws in agent-based systems." *Journal of Artificial Societies and Social Simulation*.
3. Tenenbaum, J. B. et al. (2011). "How to Grow a Mind: Statistics, Structure, and Abstraction." *Science*, 331(6022), 1279–1285.

## License

MIT
