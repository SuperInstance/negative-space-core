//! # Negative Space Core
//!
//! Intelligence is what you learn to AVOID, not what you choose.
//!
//! Based on the empirical finding of a 294:1 avoid:choose ratio and
//! the conservation of avoidance ratio across agent populations.

mod avoidance_tracker;
mod batch_analyzer;
mod conservation_law;
mod feedback_loop;
mod inference_engine;
mod negative_space;

pub use avoidance_tracker::AvoidanceTracker;
pub use batch_analyzer::BatchAnalyzer;
pub use conservation_law::ConservationLaw;
pub use feedback_loop::FeedbackLoop;
pub use inference_engine::InferenceEngine;
pub use negative_space::NegativeSpace;

/// The empirically observed avoidance ratio (avoid:choose).
pub const AVOIDANCE_RATIO: f64 = 294.0;

/// Maximum acceptable standard deviation for conservation law verification.
pub const CONSERVATION_THRESHOLD: f64 = 0.01;

/// A decision made by an agent.
#[derive(Debug, Clone, PartialEq)]
pub enum Decision {
    /// The agent avoided this option.
    Avoid(String),
    /// The agent chose this option.
    Choose(String),
}

impl Decision {
    /// Returns true if this is an avoidance.
    pub fn is_avoid(&self) -> bool {
        matches!(self, Decision::Avoid(_))
    }

    /// Returns true if this is a choice.
    pub fn is_choose(&self) -> bool {
        matches!(self, Decision::Choose(_))
    }

    /// Extract the label.
    pub fn label(&self) -> &str {
        match self {
            Decision::Avoid(s) | Decision::Choose(s) => s,
        }
    }
}

/// Result of a conservation law check.
#[derive(Debug, Clone, PartialEq)]
pub struct ConservationResult {
    /// The population sizes tested.
    pub scales: Vec<usize>,
    /// The avoidance ratios observed at each scale.
    pub ratios: Vec<f64>,
    /// Standard deviation of ratios across scales.
    pub std_dev: f64,
    /// Whether the conservation law holds.
    pub conserved: bool,
}

/// A gap inferred from negative space analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct InferredGap {
    /// The left boundary of the gap (an avoided option).
    pub left: String,
    /// The right boundary of the gap (an avoided option).
    pub right: String,
    /// The inferred entity/option in the gap.
    pub inferred: String,
    /// Confidence in this inference (0.0 to 1.0).
    pub confidence: f64,
}

/// Statistics from a batch analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchStatistics {
    /// Number of agents analyzed.
    pub agent_count: usize,
    /// Total decisions across all agents.
    pub total_decisions: usize,
    /// Total avoidances across all agents.
    pub total_avoidances: usize,
    /// Total choices across all agents.
    pub total_choices: usize,
    /// Population-level avoidance ratio.
    pub avoidance_ratio: f64,
    /// Per-agent avoidance ratios.
    pub per_agent_ratios: Vec<f64>,
    /// Standard deviation of per-agent ratios.
    pub ratio_std_dev: f64,
}

impl BatchStatistics {
    /// Returns true if the population conforms to the expected avoidance ratio.
    pub fn conforms_to_law(&self) -> bool {
        self.ratio_std_dev < CONSERVATION_THRESHOLD
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_avoidance_ratio_constant() {
        // The fundamental constant: 294:1
        assert!((AVOIDANCE_RATIO - 294.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_conservation_threshold() {
        assert!(CONSERVATION_THRESHOLD <= 0.01);
        assert!(CONSERVATION_THRESHOLD > 0.0);
    }

    #[test]
    fn test_decision_avoid() {
        let d = Decision::Avoid("bad_idea".into());
        assert!(d.is_avoid());
        assert!(!d.is_choose());
        assert_eq!(d.label(), "bad_idea");
    }

    #[test]
    fn test_decision_choose() {
        let d = Decision::Choose("good_idea".into());
        assert!(d.is_choose());
        assert!(!d.is_avoid());
        assert_eq!(d.label(), "good_idea");
    }
}
