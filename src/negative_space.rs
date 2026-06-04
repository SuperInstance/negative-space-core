use crate::{AvoidanceTracker, Decision, InferredGap};

/// Maps the unknown regions between avoidances to reveal hidden structure.
///
/// Negative space is the art of reading what isn't there. Just as the hole
/// defines the donut, the gaps between avoidances define the agent's intelligence.
#[derive(Debug, Clone)]
pub struct NegativeSpace {
    /// The agent whose negative space we're mapping.
    agent_id: String,
    /// Ordered list of avoided labels.
    avoided: Vec<String>,
    /// Ordered list of chosen labels.
    chosen: Vec<String>,
    /// Discovered gaps.
    gaps: Vec<InferredGap>,
}

impl NegativeSpace {
    /// Create a new negative space map for an agent.
    pub fn new(agent_id: impl Into<String>) -> Self {
        Self {
            agent_id: agent_id.into(),
            avoided: Vec::new(),
            chosen: Vec::new(),
            gaps: Vec::new(),
        }
    }

    /// Build negative space from an avoidance tracker.
    pub fn from_tracker(tracker: &AvoidanceTracker) -> Self {
        let mut ns = Self::new(tracker.agent_id());
        for d in tracker.decisions() {
            match d {
                Decision::Avoid(label) => ns.avoided.push(label.clone()),
                Decision::Choose(label) => ns.chosen.push(label.clone()),
            }
        }
        ns
    }

    /// Add an avoided option.
    pub fn add_avoided(&mut self, label: impl Into<String>) {
        self.avoided.push(label.into());
    }

    /// Add a chosen option.
    pub fn add_chosen(&mut self, label: impl Into<String>) {
        self.chosen.push(label.into());
    }

    /// Get the avoided labels.
    pub fn avoided(&self) -> &[String] {
        &self.avoided
    }

    /// Get the chosen labels.
    pub fn chosen(&self) -> &[String] {
        &self.chosen
    }

    /// Compute the coverage: fraction of option space "covered" by avoidances.
    ///
    /// Higher coverage means the agent's avoidances define its intelligence more precisely.
    pub fn coverage(&self) -> f64 {
        let total = self.avoided.len() + self.chosen.len();
        if total == 0 {
            0.0
        } else {
            self.avoided.len() as f64 / total as f64
        }
    }

    /// Find gaps between adjacent avoidances — regions of unexplored option space.
    ///
    /// Gaps represent options that were neither avoided nor chosen, implying
    /// they exist in a region the agent hasn't fully mapped.
    pub fn find_gaps(&mut self) -> &[InferredGap] {
        self.gaps.clear();

        if self.avoided.len() < 2 {
            return &self.gaps;
        }

        // Sort avoided labels to find adjacent gaps
        let mut sorted = self.avoided.clone();
        sorted.sort();

        for window in sorted.windows(2) {
            let left = &window[0];
            let right = &window[1];
            // Infer something in the gap
            let inferred = format!("{left}..{right}");
            // Confidence based on gap size (smaller gap = higher confidence)
            let gap_distance = string_distance(left, right);
            let confidence = if gap_distance == 0 {
                0.0
            } else {
                1.0 / (1.0 + gap_distance as f64)
            };

            self.gaps.push(InferredGap {
                left: left.clone(),
                right: right.clone(),
                inferred,
                confidence,
            });
        }

        &self.gaps
    }

    /// Get discovered gaps.
    pub fn gaps(&self) -> &[InferredGap] {
        &self.gaps
    }

    /// Compute the density of negative space — how tightly packed the avoidances are.
    pub fn density(&self) -> f64 {
        if self.avoided.len() <= 1 {
            return 0.0;
        }
        let gaps = self.avoided.len() - 1;
        1.0 / gaps as f64
    }
}

/// Simple string distance (number of differing character pairs).
fn string_distance(a: &str, b: &str) -> usize {
    let mut dist = 0;
    let (longer, shorter) = if a.len() > b.len() { (a, b) } else { (b, a) };
    dist += longer.len().saturating_sub(shorter.len());
    for (ca, cb) in a.chars().zip(b.chars()) {
        if ca != cb {
            dist += 1;
        }
    }
    dist
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_negative_space() {
        let ns = NegativeSpace::new("agent");
        assert!(ns.avoided().is_empty());
        assert!(ns.chosen().is_empty());
        assert_eq!(ns.coverage(), 0.0);
    }

    #[test]
    fn test_coverage_all_avoided() {
        let mut ns = NegativeSpace::new("agent");
        ns.add_avoided("a");
        ns.add_avoided("b");
        assert!((ns.coverage() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_coverage_mixed() {
        let mut ns = NegativeSpace::new("agent");
        ns.add_avoided("a");
        ns.add_avoided("b");
        ns.add_chosen("c");
        assert!((ns.coverage() - 2.0 / 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_find_gaps() {
        let mut ns = NegativeSpace::new("agent");
        ns.add_avoided("option_a");
        ns.add_avoided("option_c");
        let gaps = ns.find_gaps();
        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0].left, "option_a");
        assert_eq!(gaps[0].right, "option_c");
        assert!(gaps[0].confidence > 0.0);
    }

    #[test]
    fn test_no_gaps_single_avoid() {
        let mut ns = NegativeSpace::new("agent");
        ns.add_avoided("only");
        let gaps = ns.find_gaps();
        assert!(gaps.is_empty());
    }

    #[test]
    fn test_from_tracker() {
        let mut tracker = AvoidanceTracker::new("agent_42");
        tracker.record_avoidance("bad");
        tracker.record_choice("good");
        let ns = NegativeSpace::from_tracker(&tracker);
        assert_eq!(ns.avoided().len(), 1);
        assert_eq!(ns.chosen().len(), 1);
    }

    #[test]
    fn test_string_distance() {
        assert_eq!(string_distance("abc", "abc"), 0);
        assert_eq!(string_distance("abc", "abd"), 1);
        assert_eq!(string_distance("abc", "ab"), 1);
    }
}
