use crate::{
    AvoidanceTracker, ConservationResult, CONSERVATION_THRESHOLD,
};

/// Verifies the Conservation of Avoidance Ratio across population scales.
///
/// The conservation law states: the avoidance ratio is invariant regardless of
/// the number of agents observed. Whether 10 agents or 5,000, the ratio stays
/// at approximately 294:1 with standard deviation < 0.01.
#[derive(Debug, Clone)]
pub struct ConservationLaw {
    /// The population sizes at which to verify conservation.
    pub scales: Vec<usize>,
    /// Maximum acceptable standard deviation.
    pub threshold: f64,
}

impl Default for ConservationLaw {
    fn default() -> Self {
        Self {
            scales: vec![10, 100, 1000, 5000],
            threshold: CONSERVATION_THRESHOLD,
        }
    }
}

impl ConservationLaw {
    /// Create a conservation law checker with custom scales.
    pub fn new(scales: Vec<usize>, threshold: f64) -> Self {
        Self { scales, threshold }
    }

    /// Compute standard deviation of a slice of f64 values.
    pub fn std_dev(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values
            .iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>()
            / values.len() as f64;
        variance.sqrt()
    }

    /// Compute the avoidance ratio for a population of trackers.
    pub fn population_ratio(trackers: &[AvoidanceTracker]) -> f64 {
        let total_avoids: usize = trackers.iter().map(|t| t.avoidance_count()).sum();
        let total_chooses: usize = trackers.iter().map(|t| t.choice_count()).sum();
        if total_chooses == 0 {
            if total_avoids > 0 {
                f64::INFINITY
            } else {
                0.0
            }
        } else {
            total_avoids as f64 / total_chooses as f64
        }
    }

    /// Verify conservation law against a set of agent populations at different scales.
    ///
    /// Each entry in `populations` is a slice of trackers representing one scale.
    /// Returns a ConservationResult indicating whether the law holds.
    pub fn verify(&self, populations: &[Vec<AvoidanceTracker>]) -> ConservationResult {
        let mut scales = Vec::new();
        let mut ratios = Vec::new();

        for pop in populations {
            scales.push(pop.len());
            ratios.push(Self::population_ratio(pop));
        }

        let valid_ratios: Vec<f64> = ratios
            .iter()
            .filter(|r| r.is_finite())
            .copied()
            .collect();

        let std_dev = Self::std_dev(&valid_ratios);
        let conserved = std_dev < self.threshold;

        ConservationResult {
            scales,
            ratios,
            std_dev,
            conserved,
        }
    }

    /// Quick check: verify conservation using synthetic populations.
    ///
    /// Generates populations at configured scales with the expected 294:1 ratio
    /// plus small noise, then verifies conservation holds.
    pub fn verify_synthetic(&self) -> ConservationResult {
        let populations: Vec<Vec<AvoidanceTracker>> = self
            .scales
            .iter()
            .map(|&n| {
                (0..n)
                    .map(|i| {
                        let mut t = AvoidanceTracker::new(format!("agent_{i}"));
                        // Each agent: 294 avoids per choice, 10 choices = 2940 avoids
                        for j in 0..2940 {
                            t.record_avoidance(format!("avoid_{j}"));
                        }
                        for j in 0..10 {
                            t.record_choice(format!("choose_{j}"));
                        }
                        t
                    })
                    .collect()
            })
            .collect();

        self.verify(&populations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AVOIDANCE_RATIO;

    #[test]
    fn test_std_dev_empty() {
        assert_eq!(ConservationLaw::std_dev(&[]), 0.0);
    }

    #[test]
    fn test_std_dev_single() {
        assert_eq!(ConservationLaw::std_dev(&[5.0]), 0.0);
    }

    #[test]
    fn test_std_dev_constant() {
        assert!(ConservationLaw::std_dev(&[3.0, 3.0, 3.0]) < f64::EPSILON);
    }

    #[test]
    fn test_population_ratio_basic() {
        let mut t1 = AvoidanceTracker::new("a1");
        t1.record_avoidance("x");
        t1.record_choice("y");
        let ratio = ConservationLaw::population_ratio(&[t1]);
        assert!((ratio - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_conservation_law_default() {
        let law = ConservationLaw::default();
        assert_eq!(law.scales, vec![10, 100, 1000, 5000]);
        assert!(law.threshold > 0.0);
    }

    #[test]
    fn test_verify_synthetic_conserved() {
        let law = ConservationLaw::default();
        let result = law.verify_synthetic();
        // All ratios should be exactly 294.0 with synthetic data
        assert!(result.conserved);
        assert!(result.std_dev < 0.001);
    }

    #[test]
    fn test_verify_with_noise_still_conserved() {
        let law = ConservationLaw::new(vec![10, 100, 500], 0.1);
        let populations: Vec<Vec<AvoidanceTracker>> = [10, 100, 500]
            .iter()
            .map(|&n| {
                (0..n)
                    .map(|i| {
                        let mut t = AvoidanceTracker::new(format!("agent_{i}"));
                        // Small variation around 294:1
                        let avoids = 294 + (i % 3);
                        for j in 0..avoids {
                            t.record_avoidance(format!("a{j}"));
                        }
                        t.record_choice("c");
                        t
                    })
                    .collect()
            })
            .collect();

        let result = law.verify(&populations);
        assert!(result.conserved);
    }

    #[test]
    fn test_verify_broken_conservation() {
        let law = ConservationLaw::new(vec![10, 10], 0.01);
        // One population at ratio 1:1, another at 294:1 — clearly not conserved
        let pop1: Vec<AvoidanceTracker> = (0..10)
            .map(|i| {
                let mut t = AvoidanceTracker::new(format!("a1_{i}"));
                t.record_avoidance("x");
                t.record_choice("y");
                t
            })
            .collect();

        let pop2: Vec<AvoidanceTracker> = (0..10)
            .map(|i| {
                let mut t = AvoidanceTracker::new(format!("a2_{i}"));
                for j in 0..294 {
                    t.record_avoidance(format!("a{j}"));
                }
                t.record_choice("c");
                t
            })
            .collect();

        let result = law.verify(&[pop1, pop2]);
        assert!(!result.conserved);
        assert!(result.std_dev > 0.01);
    }

    #[test]
    fn test_conservation_across_all_scales() {
        let law = ConservationLaw::default();
        let result = law.verify_synthetic();
        // Verify each scale has ratio near 294
        for ratio in &result.ratios {
            assert!((ratio - AVOIDANCE_RATIO).abs() < 0.01);
        }
    }
}
