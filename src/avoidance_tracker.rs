use crate::{Decision, AVOIDANCE_RATIO};

/// Tracks per-agent avoidance history and computes avoidance ratios.
///
/// The avoidance ratio is the fundamental metric: how many options does an agent
/// avoid for each one it chooses? Empirically this converges to ~294:1.
#[derive(Debug, Clone)]
pub struct AvoidanceTracker {
    agent_id: String,
    decisions: Vec<Decision>,
}

impl AvoidanceTracker {
    /// Create a new tracker for the given agent.
    pub fn new(agent_id: impl Into<String>) -> Self {
        Self {
            agent_id: agent_id.into(),
            decisions: Vec::new(),
        }
    }

    /// Get the agent's ID.
    pub fn agent_id(&self) -> &str {
        &self.agent_id
    }

    /// Record that the agent avoided an option.
    pub fn record_avoidance(&mut self, label: impl Into<String>) {
        self.decisions.push(Decision::Avoid(label.into()));
    }

    /// Record that the agent chose an option.
    pub fn record_choice(&mut self, label: impl Into<String>) {
        self.decisions.push(Decision::Choose(label.into()));
    }

    /// Record a raw decision.
    pub fn record(&mut self, decision: Decision) {
        self.decisions.push(decision);
    }

    /// Total number of decisions recorded.
    pub fn total_decisions(&self) -> usize {
        self.decisions.len()
    }

    /// Number of avoidances.
    pub fn avoidance_count(&self) -> usize {
        self.decisions.iter().filter(|d| d.is_avoid()).count()
    }

    /// Number of choices.
    pub fn choice_count(&self) -> usize {
        self.decisions.iter().filter(|d| d.is_choose()).count()
    }

    /// Compute the avoidance ratio (avoids / choices).
    ///
    /// Returns `f64::INFINITY` if no choices have been made.
    /// Returns 0.0 if no decisions have been recorded.
    pub fn avoidance_ratio(&self) -> f64 {
        let avoids = self.avoidance_count() as f64;
        let chooses = self.choice_count() as f64;
        if chooses == 0.0 {
            if avoids > 0.0 {
                f64::INFINITY
            } else {
                0.0
            }
        } else {
            avoids / chooses
        }
    }

    /// Check if this agent's avoidance ratio is close to the expected 294:1.
    pub fn ratio_near_expected(&self, tolerance: f64) -> bool {
        let ratio = self.avoidance_ratio();
        (ratio - AVOIDANCE_RATIO).abs() < tolerance
    }

    /// Get all decisions.
    pub fn decisions(&self) -> &[Decision] {
        &self.decisions
    }

    /// Get all avoided labels.
    pub fn avoided_labels(&self) -> Vec<&str> {
        self.decisions
            .iter()
            .filter(|d| d.is_avoid())
            .map(|d| d.label())
            .collect()
    }

    /// Get all chosen labels.
    pub fn chosen_labels(&self) -> Vec<&str> {
        self.decisions
            .iter()
            .filter(|d| d.is_choose())
            .map(|d| d.label())
            .collect()
    }

    /// Detect if the agent's avoidance pattern violates the conservation law.
    ///
    /// A violation means the ratio deviates significantly from 294:1.
    pub fn conservation_violation(&self, threshold: f64) -> bool {
        let ratio = self.avoidance_ratio();
        if ratio.is_infinite() || ratio.is_nan() {
            return false; // Not enough data to judge
        }
        (ratio - AVOIDANCE_RATIO).abs() > threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_empty_tracker() {
        let t = AvoidanceTracker::new("a1");
        assert_eq!(t.total_decisions(), 0);
        assert_eq!(t.avoidance_count(), 0);
        assert_eq!(t.choice_count(), 0);
        assert_eq!(t.avoidance_ratio(), 0.0);
    }

    #[test]
    fn test_only_avoids() {
        let mut t = AvoidanceTracker::new("a1");
        t.record_avoidance("x");
        t.record_avoidance("y");
        assert_eq!(t.avoidance_ratio(), f64::INFINITY);
    }

    #[test]
    fn test_exact_ratio() {
        let mut t = AvoidanceTracker::new("a1");
        for i in 0..294 {
            t.record_avoidance(format!("avoid_{i}"));
        }
        t.record_choice("chosen");
        assert!((t.avoidance_ratio() - 294.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ratio_near_expected() {
        let mut t = AvoidanceTracker::new("a1");
        for i in 0..290 {
            t.record_avoidance(format!("a{i}"));
        }
        t.record_choice("c");
        assert!(t.ratio_near_expected(10.0));
    }

    #[test]
    fn test_conservation_violation() {
        let mut t = AvoidanceTracker::new("a1");
        t.record_avoidance("x");
        t.record_choice("y");
        assert!(t.conservation_violation(100.0));
    }

    #[test]
    fn test_no_conservation_violation_with_enough_data() {
        let mut t = AvoidanceTracker::new("a1");
        for i in 0..2940 {
            t.record_avoidance(format!("a{i}"));
        }
        for i in 0..10 {
            t.record_choice(format!("c{i}"));
        }
        assert!(!t.conservation_violation(50.0));
    }
}
