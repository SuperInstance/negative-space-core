use crate::{
    AvoidanceTracker, BatchStatistics, ConservationLaw, AVOIDANCE_RATIO,
};

/// Analyzes batches of agent decisions and computes population-level statistics.
///
/// BatchAnalyzer is the workhorse for empirical verification: feed it populations,
/// and it tells you whether the conservation law holds at scale.
#[derive(Debug, Clone)]
pub struct BatchAnalyzer {
    /// All trackers collected so far.
    trackers: Vec<AvoidanceTracker>,
}

impl Default for BatchAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl BatchAnalyzer {
    /// Create a new empty batch analyzer.
    pub fn new() -> Self {
        Self {
            trackers: Vec::new(),
        }
    }

    /// Add a tracker to the batch.
    pub fn add(&mut self, tracker: AvoidanceTracker) {
        self.trackers.push(tracker);
    }

    /// Add multiple trackers.
    pub fn extend(&mut self, trackers: impl IntoIterator<Item = AvoidanceTracker>) {
        self.trackers.extend(trackers);
    }

    /// Number of agents in the batch.
    pub fn len(&self) -> usize {
        self.trackers.len()
    }

    /// Check if the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.trackers.is_empty()
    }

    /// Compute statistics for the current batch.
    pub fn analyze(&self) -> BatchStatistics {
        let agent_count = self.trackers.len();
        let total_decisions: usize = self.trackers.iter().map(|t| t.total_decisions()).sum();
        let total_avoidances: usize = self.trackers.iter().map(|t| t.avoidance_count()).sum();
        let total_choices: usize = self.trackers.iter().map(|t| t.choice_count()).sum();

        let avoidance_ratio = if total_choices == 0 {
            if total_avoidances > 0 {
                f64::INFINITY
            } else {
                0.0
            }
        } else {
            total_avoidances as f64 / total_choices as f64
        };

        let per_agent_ratios: Vec<f64> = self
            .trackers
            .iter()
            .map(|t| t.avoidance_ratio())
            .filter(|r| r.is_finite())
            .collect();

        let ratio_std_dev = ConservationLaw::std_dev(&per_agent_ratios);

        BatchStatistics {
            agent_count,
            total_decisions,
            total_avoidances,
            total_choices,
            avoidance_ratio,
            per_agent_ratios,
            ratio_std_dev,
        }
    }

    /// Generate a synthetic population that conforms to the 294:1 law.
    ///
    /// Useful for testing and baseline comparisons.
    pub fn generate_synthetic(agent_count: usize, choices_per_agent: usize) -> Self {
        let avoids_per_agent = (AVOIDANCE_RATIO * choices_per_agent as f64) as usize;
        let trackers: Vec<AvoidanceTracker> = (0..agent_count)
            .map(|i| {
                let mut t = AvoidanceTracker::new(format!("synth_{i}"));
                for j in 0..avoids_per_agent {
                    t.record_avoidance(format!("avoid_{j}"));
                }
                for j in 0..choices_per_agent {
                    t.record_choice(format!("choose_{j}"));
                }
                t
            })
            .collect();

        Self { trackers }
    }

    /// Generate a synthetic population with noise around the 294:1 ratio.
    pub fn generate_noisy(
        agent_count: usize,
        choices_per_agent: usize,
        noise: usize,
    ) -> Self {
        let trackers: Vec<AvoidanceTracker> = (0..agent_count)
            .map(|i| {
                let mut t = AvoidanceTracker::new(format!("noisy_{i}"));
                let base_avoids = (AVOIDANCE_RATIO * choices_per_agent as f64) as usize;
                let avoids = base_avoids + (i % (2 * noise + 1)).saturating_sub(noise);
                for j in 0..avoids {
                    t.record_avoidance(format!("avoid_{j}"));
                }
                for j in 0..choices_per_agent {
                    t.record_choice(format!("choose_{j}"));
                }
                t
            })
            .collect();

        Self { trackers }
    }

    /// Split the batch into sub-populations by size for conservation law testing.
    pub fn split_by_scale(&self, scales: &[usize]) -> Vec<Vec<AvoidanceTracker>> {
        scales
            .iter()
            .map(|&size| {
                self.trackers
                    .iter()
                    .take(size)
                    .cloned()
                    .collect()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_batch() {
        let ba = BatchAnalyzer::new();
        let stats = ba.analyze();
        assert_eq!(stats.agent_count, 0);
        assert_eq!(stats.total_decisions, 0);
        assert_eq!(stats.avoidance_ratio, 0.0);
    }

    #[test]
    fn test_synthetic_population() {
        let ba = BatchAnalyzer::generate_synthetic(100, 10);
        let stats = ba.analyze();
        assert_eq!(stats.agent_count, 100);
        assert!((stats.avoidance_ratio - AVOIDANCE_RATIO).abs() < 0.01);
    }

    #[test]
    fn test_noisy_population_near_law() {
        let ba = BatchAnalyzer::generate_noisy(500, 10, 5);
        let stats = ba.analyze();
        // Should be close to 294:1 even with noise
        assert!((stats.avoidance_ratio - AVOIDANCE_RATIO).abs() < 1.0);
    }

    #[test]
    fn test_conforms_to_law() {
        let ba = BatchAnalyzer::generate_synthetic(1000, 10);
        let stats = ba.analyze();
        assert!(stats.conforms_to_law());
    }

    #[test]
    fn test_split_by_scale() {
        let ba = BatchAnalyzer::generate_synthetic(100, 5);
        let splits = ba.split_by_scale(&[10, 50, 100]);
        assert_eq!(splits.len(), 3);
        assert_eq!(splits[0].len(), 10);
        assert_eq!(splits[1].len(), 50);
        assert_eq!(splits[2].len(), 100);
    }

    #[test]
    fn test_add_and_extend() {
        let mut ba = BatchAnalyzer::new();
        let mut t1 = AvoidanceTracker::new("a1");
        t1.record_avoidance("x");
        t1.record_choice("y");

        let mut t2 = AvoidanceTracker::new("a2");
        t2.record_avoidance("a");
        t2.record_choice("b");

        ba.add(t1);
        ba.extend(vec![t2]);
        assert_eq!(ba.len(), 2);
    }
}
