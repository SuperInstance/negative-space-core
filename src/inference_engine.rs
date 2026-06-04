use crate::{AvoidanceTracker, InferredGap};

/// Makes deductions from negative spaces — the Inference from Absence Law.
///
/// If agents avoid X and Y, the gap between them implies the existence of Z.
/// The InferenceEngine turns patterns of avoidance into predictions about
/// the unknown.
#[derive(Debug, Clone)]
pub struct InferenceEngine {
    /// Minimum confidence threshold for accepting an inference.
    pub confidence_threshold: f64,
}

impl Default for InferenceEngine {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.3,
        }
    }
}

impl InferenceEngine {
    pub fn new(confidence_threshold: f64) -> Self {
        Self {
            confidence_threshold,
        }
    }

    /// Infer what lies between two avoided options.
    ///
    /// If agents consistently avoid both X and Y, the space between them
    /// likely contains something they haven't encountered — or something
    /// so obvious it doesn't register as a decision.
    pub fn infer_between(&self, left: &str, right: &str) -> InferredGap {
        let inferred = self.generate_inferred(left, right);
        let confidence = self.compute_confidence(left, right);

        InferredGap {
            left: left.to_string(),
            right: right.to_string(),
            inferred,
            confidence,
        }
    }

    /// Batch inference: find all implied gaps from an agent's avoidance pattern.
    pub fn infer_from_agent(&self, tracker: &AvoidanceTracker) -> Vec<InferredGap> {
        let avoided: Vec<&str> = tracker
            .decisions()
            .iter()
            .filter(|d| d.is_avoid())
            .map(|d| d.label())
            .collect();

        let mut gaps = Vec::new();
        if avoided.len() < 2 {
            return gaps;
        }

        // Sort to find adjacent pairs
        let mut sorted: Vec<&str> = avoided;
        sorted.sort();

        for window in sorted.windows(2) {
            let gap = self.infer_between(window[0], window[1]);
            if gap.confidence >= self.confidence_threshold {
                gaps.push(gap);
            }
        }

        gaps
    }

    /// Cross-agent inference: if multiple agents avoid the same pair,
    /// the inferred gap has higher confidence.
    pub fn infer_population(
        &self,
        trackers: &[AvoidanceTracker],
    ) -> Vec<InferredGap> {
        let mut gap_counts: std::collections::HashMap<(String, String), usize> =
            std::collections::HashMap::new();

        for tracker in trackers {
            let avoided: Vec<&str> = tracker
                .decisions()
                .iter()
                .filter(|d| d.is_avoid())
                .map(|d| d.label())
                .collect();

            let mut sorted = avoided;
            sorted.sort();

            for window in sorted.windows(2) {
                let key = (window[0].to_string(), window[1].to_string());
                *gap_counts.entry(key).or_default() += 1;
            }
        }

        let agent_count = trackers.len() as f64;
        gap_counts
            .into_iter()
            .filter_map(|((left, right), count)| {
                let confidence = (count as f64 / agent_count).min(1.0);
                if confidence >= self.confidence_threshold {
                    Some(InferredGap {
                        inferred: self.generate_inferred(&left, &right),
                        left,
                        right,
                        confidence,
                    })
                } else {
                    None
                }
            })
            .collect()
    }

    /// Generate an inferred label for the gap between two avoided options.
    fn generate_inferred(&self, left: &str, right: &str) -> String {
        // Find common prefix and suffix, the gap is what differs
        let common_prefix: String = left
            .chars()
            .zip(right.chars())
            .take_while(|(a, b)| a == b)
            .map(|(a, _)| a)
            .collect();

        let suffix_l: String = left.chars().skip(common_prefix.len()).collect();
        let suffix_r: String = right.chars().skip(common_prefix.len()).collect();

        if suffix_l.is_empty() && suffix_r.is_empty() {
            format!("{left}≡{right}")
        } else {
            format!("{common_prefix}[{suffix_l}↔{suffix_r}]")
        }
    }

    /// Compute confidence based on similarity of the two avoided options.
    fn compute_confidence(&self, left: &str, right: &str) -> f64 {
        if left == right {
            return 0.0;
        }

        let max_len = left.len().max(right.len()) as f64;
        if max_len == 0.0 {
            return 0.0;
        }

        // Count matching characters
        let matches: usize = left
            .chars()
            .zip(right.chars())
            .filter(|(a, b)| a == b)
            .count();

        matches as f64 / max_len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infer_between_similar() {
        let engine = InferenceEngine::default();
        let gap = engine.infer_between("option_a", "option_c");
        assert!(gap.confidence > 0.5);
        assert!(!gap.inferred.is_empty());
    }

    #[test]
    fn test_infer_between_different() {
        let engine = InferenceEngine::default();
        let gap = engine.infer_between("cat", "zebra");
        assert!(gap.confidence < 0.5);
    }

    #[test]
    fn test_infer_between_identical() {
        let engine = InferenceEngine::default();
        let gap = engine.infer_between("same", "same");
        assert_eq!(gap.confidence, 0.0);
    }

    #[test]
    fn test_infer_from_agent() {
        let engine = InferenceEngine::new(0.0);
        let mut t = AvoidanceTracker::new("a1");
        t.record_avoidance("alpha");
        t.record_avoidance("gamma");
        t.record_choice("beta");

        let gaps = engine.infer_from_agent(&t);
        assert!(!gaps.is_empty());
    }

    #[test]
    fn test_infer_from_agent_too_few() {
        let engine = InferenceEngine::default();
        let mut t = AvoidanceTracker::new("a1");
        t.record_avoidance("only_one");

        let gaps = engine.infer_from_agent(&t);
        assert!(gaps.is_empty());
    }

    #[test]
    fn test_population_inference() {
        let engine = InferenceEngine::new(0.5);

        let trackers: Vec<AvoidanceTracker> = (0..5)
            .map(|i| {
                let mut t = AvoidanceTracker::new(format!("a{i}"));
                t.record_avoidance("option_a");
                t.record_avoidance("option_z");
                t
            })
            .collect();

        let gaps = engine.infer_population(&trackers);
        assert!(!gaps.is_empty());
        // All 5 agents agree on this gap
        assert!((gaps[0].confidence - 1.0).abs() < f64::EPSILON);
    }
}
