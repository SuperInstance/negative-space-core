use crate::{AvoidanceTracker, AVOIDANCE_RATIO};

/// Updates avoidance model based on outcomes — implements v5 balanced learning.
///
/// The Feedback Refinement Law: every avoidance decision updates the model.
/// Agents that avoid well survive; survivors avoid better.
#[derive(Debug, Clone)]
pub struct FeedbackLoop {
    /// Learning rate (0.0 to 1.0). How quickly the model adapts.
    pub learning_rate: f64,
    /// Current model estimate of the avoidance ratio.
    estimated_ratio: f64,
    /// Number of updates processed.
    update_count: usize,
}

impl Default for FeedbackLoop {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            estimated_ratio: AVOIDANCE_RATIO,
            update_count: 0,
        }
    }
}

impl FeedbackLoop {
    /// Create a new feedback loop with custom learning rate.
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            estimated_ratio: AVOIDANCE_RATIO,
            update_count: 0,
        }
    }

    /// Get the current estimated avoidance ratio.
    pub fn estimated_ratio(&self) -> f64 {
        self.estimated_ratio
    }

    /// Get the number of updates processed.
    pub fn update_count(&self) -> usize {
        self.update_count
    }

    /// Update the model with an observed avoidance ratio from a single agent.
    ///
    /// Uses exponential moving average (v5 balanced learning) to avoid
    /// oscillation while still tracking real changes.
    pub fn update(&mut self, observed_ratio: f64) {
        if observed_ratio.is_finite() {
            self.estimated_ratio = self.estimated_ratio * (1.0 - self.learning_rate)
                + observed_ratio * self.learning_rate;
            self.update_count += 1;
        }
    }

    /// Process an agent's tracker and update the model.
    pub fn update_from_tracker(&mut self, tracker: &AvoidanceTracker) {
        let ratio = tracker.avoidance_ratio();
        self.update(ratio);
    }

    /// Batch update from multiple agents.
    pub fn update_from_population(&mut self, trackers: &[AvoidanceTracker]) {
        for tracker in trackers {
            self.update_from_tracker(tracker);
        }
    }

    /// Compute the error between current estimate and expected ratio.
    pub fn error(&self) -> f64 {
        (self.estimated_ratio - AVOIDANCE_RATIO).abs()
    }

    /// Check if the model has converged (error < tolerance).
    pub fn converged(&self, tolerance: f64) -> bool {
        self.error() < tolerance
    }

    /// Apply v5 balanced learning: weight updates by confidence.
    ///
    /// Agents with ratios closer to expected get higher weight.
    pub fn balanced_update(&mut self, trackers: &[AvoidanceTracker]) {
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;

        for tracker in trackers {
            let ratio = tracker.avoidance_ratio();
            if !ratio.is_finite() {
                continue;
            }
            // Weight inversely proportional to distance from expected
            let distance = (ratio - AVOIDANCE_RATIO).abs();
            let weight = 1.0 / (1.0 + distance);
            weighted_sum += ratio * weight;
            total_weight += weight;
        }

        if total_weight > 0.0 {
            let balanced_ratio = weighted_sum / total_weight;
            self.estimated_ratio = self.estimated_ratio * (1.0 - self.learning_rate)
                + balanced_ratio * self.learning_rate;
            self.update_count += trackers.len();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_starts_at_expected() {
        let fb = FeedbackLoop::default();
        assert!((fb.estimated_ratio() - AVOIDANCE_RATIO).abs() < f64::EPSILON);
    }

    #[test]
    fn test_update_converges() {
        let mut fb = FeedbackLoop::new(0.1);
        // Feed it observations near 294:1
        for _ in 0..100 {
            fb.update(295.0);
        }
        assert!(fb.estimated_ratio() > 294.0);
        assert!(fb.estimated_ratio() < 295.0);
    }

    #[test]
    fn test_update_ignores_nonfinite() {
        let mut fb = FeedbackLoop::default();
        let before = fb.estimated_ratio();
        fb.update(f64::INFINITY);
        fb.update(f64::NAN);
        assert!((fb.estimated_ratio() - before).abs() < f64::EPSILON);
    }

    #[test]
    fn test_update_from_tracker() {
        let mut fb = FeedbackLoop::new(0.1);
        let mut t = AvoidanceTracker::new("a1");
        for i in 0..588 {
            t.record_avoidance(format!("a{i}"));
        }
        t.record_choice("c");
        t.record_choice("c");
        fb.update_from_tracker(&t);
        assert_eq!(fb.update_count(), 1);
    }

    #[test]
    fn test_converged() {
        let fb = FeedbackLoop::default();
        assert!(fb.converged(1.0));
    }

    #[test]
    fn test_balanced_update() {
        let mut fb = FeedbackLoop::new(0.5);
        let trackers: Vec<AvoidanceTracker> = (0..10)
            .map(|i| {
                let mut t = AvoidanceTracker::new(format!("a{i}"));
                let avoids = 294 + (i % 5);
                for j in 0..avoids {
                    t.record_avoidance(format!("a{j}"));
                }
                t.record_choice("c");
                t
            })
            .collect();

        fb.balanced_update(&trackers);
        assert_eq!(fb.update_count(), 10);
        // Should be close to 294 still
        assert!(fb.error() < 5.0);
    }
}
