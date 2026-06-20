//! Fixation/saccade classification module
//!
//! Traces to:
//!   FR-EYE-INFER-003 — Fixation classification (velocity <30°/s for ≥100 ms)
//!   FR-EYE-INFER-004 — Saccade classification (velocity >50°/s, ≥50 ms duration)
//!
//! The classifier maintains a sliding window of gaze samples, computes smoothed
//! angular velocity from consecutive pairs (via `atan2` differences), and
//! transitions through a three-state machine (Unknown → Fixation / Saccade).
//! Blink rejection suppresses events when confidence drops and recovers within
//! 300 ms.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum angular velocity (deg/s) below which gaze is considered a fixation.
const FIXATION_VELOCITY_THRESHOLD_DEG_S: f32 = 30.0;

/// Minimum angular velocity (deg/s) above which gaze is considered a saccade.
const SACCADE_VELOCITY_THRESHOLD_DEG_S: f32 = 50.0;

/// How long the gaze must remain below the fixation threshold before we report
/// a fixation (FR-EYE-INFER-003).
const FIXATION_MIN_DURATION: Duration = Duration::from_millis(100);

/// Minimum saccade duration; shorter events are rejected as noise / blinks
/// (FR-EYE-INFER-004 blink rejection).
const SACCADE_MIN_DURATION: Duration = Duration::from_millis(50);

/// Confidence drop shorter than this is treated as a blink (ignored).
const BLINK_RECOVERY_DURATION: Duration = Duration::from_millis(300);

/// Default sliding-window size (number of samples).
const DEFAULT_WINDOW_SIZE: usize = 10;

/// Minimum confidence required for a sample to participate in classification.
const CONFIDENCE_THRESHOLD: f32 = 0.5;

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

/// A single gaze measurement stored in the sliding window.
#[derive(Debug, Clone)]
struct GazeSample {
    x: f32,
    y: f32,
    timestamp: Instant,
    confidence: f32,
}

/// Internal state of the classifier state‑machine.
#[derive(Debug, Clone, PartialEq)]
enum ClassifierState {
    /// No classification yet, or in between events.
    Unknown,
    /// Gaze is held steady on a region (FR-EYE-INFER-003).
    Fixation,
    /// Rapid eye movement in progress (FR-EYE-INFER-004).
    Saccade,
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// An event emitted when the classifier observes a state transition.
#[derive(Debug, Clone)]
pub enum GazeEvent {
    /// A fixation has just started.
    FixationStart {
        /// Mean gaze position (running average) so far.
        centroid: (f32, f32),
        /// Timestamp of the first sample classified as part of this fixation.
        timestamp: Instant,
    },
    /// A fixation has ended (usually because a saccade began, or tracking
    /// was lost).
    FixationEnd {
        /// Final centroid of the completed fixation.
        centroid: (f32, f32),
        /// How long the fixation lasted (start timestamp → last fixation sample).
        duration: Duration,
    },
    /// A saccade (rapid eye movement) has completed.
    Saccade {
        /// Position when the saccade started.
        start: (f32, f32),
        /// Position when the saccade ended.
        end: (f32, f32),
        /// Amplitude (Euclidean distance in the classifier's coordinate space).
        amplitude: f32,
        /// Highest instantaneous angular velocity observed during this saccade
        /// (deg/s).
        peak_velocity: f32,
        /// How long the saccade lasted.
        duration: Duration,
    },
}

/// Describes the classifier's current belief about the gaze state.
#[derive(Debug, Clone, PartialEq)]
pub enum GazeClassification {
    /// Gaze is fixated on a region.
    Fixation {
        /// Running centroid of the fixation.
        centroid: (f32, f32),
        /// How long the fixation has lasted so far.
        duration: Duration,
    },
    /// Rapid eye movement is in progress.
    Saccade,
    /// Not enough information to classify yet.
    Unknown,
}

// ---------------------------------------------------------------------------
// GazeClassifier
// ---------------------------------------------------------------------------

/// Sliding-window gaze classifier that emits fixation / saccade events based
/// on angular velocity thresholds.
///
/// # State machine
///
/// ```text
///                ┌──────────┐
///                │ Unknown  │
///                └────┬─────┘
///              ┌──────┴──────┐
///           slow ≥100ms   fast >50°/s
///              └──────┬──────┘
///              ┌──────┴──────┐
///        ┌─────▼────┐  ┌────▼──────┐
///        │ Fixation │  │  Saccade  │
///        └─────┬────┘  └────┬──────┘
///          fast │           │ slow <30°/s
///              └──────┬──────┘
///                 (via Unknown
///                  with dwell)
/// ```
///
/// # Blink rejection
///
/// Samples with `confidence < 0.5` are still added to the window (so the
/// velocity timeline is continuous) but no events are emitted.  If confidence
/// recovers within 300 ms the interruption is treated as a blink and the
/// classifier resets its internal state, preventing false saccade triggers.
/// Saccade events shorter than 50 ms are also suppressed.
#[derive(Debug, Clone)]
pub struct GazeClassifier {
    // -- configuration ---
    window_size: usize,

    // -- sliding window ---
    samples: VecDeque<GazeSample>,

    // -- state machine ---
    state: ClassifierState,

    // -- fixation tracking ---
    fixation_start_time: Option<Instant>,
    fixation_sum_x: f32,
    fixation_sum_y: f32,
    fixation_sample_count: u32,

    // -- saccade tracking ---
    saccade_start_x: f32,
    saccade_start_y: f32,
    saccade_start_time: Option<Instant>,
    saccade_peak_velocity: f32,

    // -- low-velocity dwell (used to enter fixation from Unknown) ---
    low_velocity_start: Option<Instant>,

    // -- blink / confidence tracking ---
    confidence_drop_start: Option<Instant>,

    /// Timestamp of the most recent sample (used by [`current_state`]).
    last_sample_time: Option<Instant>,
}

impl GazeClassifier {
    /// Create a new classifier with the given sliding-window capacity.
    ///
    /// `window_size` controls how many past samples are kept for velocity
    /// smoothing.  Larger windows are more stable but slow to detect state
    /// transitions.  Supply `0` to use the default (10).
    pub fn new(window_size: usize) -> Self {
        let ws = if window_size == 0 {
            DEFAULT_WINDOW_SIZE
        } else {
            window_size
        };
        Self {
            window_size: ws,
            samples: VecDeque::with_capacity(ws + 1),
            state: ClassifierState::Unknown,
            fixation_start_time: None,
            fixation_sum_x: 0.0,
            fixation_sum_y: 0.0,
            fixation_sample_count: 0,
            saccade_start_x: 0.0,
            saccade_start_y: 0.0,
            saccade_start_time: None,
            saccade_peak_velocity: 0.0,
            low_velocity_start: None,
            confidence_drop_start: None,
            last_sample_time: None,
        }
    }

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    /// Feed a new gaze sample into the classifier.
    ///
    /// Returns zero or more [`GazeEvent`]s triggered by this sample (fixation
    /// start / end, saccade completion).  Events are only returned on state
    /// transitions; most calls return an empty vec.
    pub fn update(
        &mut self,
        gaze_x: f32,
        gaze_y: f32,
        timestamp: Instant,
        confidence: f32,
    ) -> Vec<GazeEvent> {
        let mut events = Vec::new();

        // ------------------------------------------------------------------
        // Blink / low-confidence handling
        // ------------------------------------------------------------------
        if confidence < CONFIDENCE_THRESHOLD {
            if self.confidence_drop_start.is_none() {
                self.confidence_drop_start = Some(timestamp);
            }
            // Still add the sample to keep the window continuous.
            self.add_sample(gaze_x, gaze_y, timestamp, confidence);
            return events;
        }

        // Confidence has recovered – handle blink recovery.
        if let Some(drop_start) = self.confidence_drop_start.take() {
            let blink_duration = timestamp.duration_since(drop_start);
            if blink_duration < BLINK_RECOVERY_DURATION {
                // This was a blink – reset state so it looks like nothing
                // happened.
                self.reset_internal_state();
                self.add_sample(gaze_x, gaze_y, timestamp, confidence);
                return events;
            }
            // Gap was longer than blink threshold; treat as genuine tracking
            // loss.  The classifier starts fresh.
        }

        // The sample itself carries high confidence.
        self.add_sample(gaze_x, gaze_y, timestamp, confidence);

        // ------------------------------------------------------------------
        // Velocity computation
        // ------------------------------------------------------------------
        let velocity = match self.compute_smoothed_velocity() {
            Some(v) => v,
            None => return events, // Not enough samples in the window yet.
        };

        // ------------------------------------------------------------------
        // State-machine transitions
        // ------------------------------------------------------------------
        match self.state {
            ClassifierState::Unknown => {
                self.handle_unknown(velocity, gaze_x, gaze_y, timestamp, &mut events);
            }
            ClassifierState::Fixation => {
                self.handle_fixation(velocity, gaze_x, gaze_y, timestamp, &mut events);
            }
            ClassifierState::Saccade => {
                self.handle_saccade(velocity, gaze_x, gaze_y, timestamp, &mut events);
            }
        }

        events
    }

    /// Clear all internal state including the sample window.
    pub fn reset(&mut self) {
        self.samples.clear();
        self.reset_internal_state();
        self.last_sample_time = None;
    }

    /// Return the classifier's current belief about the gaze state.
    pub fn current_state(&self) -> GazeClassification {
        match self.state {
            ClassifierState::Fixation => {
                let centroid = self.fixation_centroid();
                let duration = match (self.fixation_start_time, self.last_sample_time) {
                    (Some(start), Some(last)) => last.duration_since(start),
                    _ => Duration::ZERO,
                };
                GazeClassification::Fixation { centroid, duration }
            }
            ClassifierState::Saccade => GazeClassification::Saccade,
            ClassifierState::Unknown => GazeClassification::Unknown,
        }
    }

    /// Convenience: `true` when the classifier thinks the user is fixating.
    pub fn is_fixating(&self) -> bool {
        self.state == ClassifierState::Fixation
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /// Add a sample to the sliding window, evicting the oldest if needed.
    fn add_sample(&mut self, x: f32, y: f32, timestamp: Instant, confidence: f32) {
        if self.samples.len() >= self.window_size {
            self.samples.pop_front();
        }
        self.samples.push_back(GazeSample {
            x,
            y,
            timestamp,
            confidence,
        });
        self.last_sample_time = Some(timestamp);
    }

    /// Compute the mean angular velocity (deg/s) across all consecutive pairs
    /// in the sliding window.
    ///
    /// Angular displacement between two samples is computed as the absolute
    /// difference of their `atan2(y, x)` angles, normalised to `[0, π]`.
    /// Instantaneous velocity for a pair is `Δθ / Δt` converted to °/s.
    fn compute_smoothed_velocity(&self) -> Option<f32> {
        if self.samples.len() < 2 {
            return None;
        }

        let mut total: f32 = 0.0;
        let mut count: u32 = 0;

        for i in 1..self.samples.len() {
            let prev = &self.samples[i - 1];
            let curr = &self.samples[i];

            let dt = curr.timestamp.duration_since(prev.timestamp);
            let dt_secs = dt.as_secs_f32();
            if dt_secs <= 0.0 {
                continue;
            }

            let angle_prev = prev.y.atan2(prev.x);
            let angle_curr = curr.y.atan2(curr.x);

            let mut diff = (angle_curr - angle_prev).abs();
            // Normalise angular distance to [0, π] – the shortest arc.
            if diff > std::f32::consts::PI {
                diff = 2.0 * std::f32::consts::PI - diff;
            }

            let vel_deg_s = (diff / dt_secs).to_degrees();
            total += vel_deg_s;
            count += 1;
        }

        if count == 0 {
            return None;
        }
        Some(total / count as f32)
    }

    /// Reset all state-machine fields *except* the sample window.
    fn reset_internal_state(&mut self) {
        self.state = ClassifierState::Unknown;
        self.fixation_start_time = None;
        self.fixation_sum_x = 0.0;
        self.fixation_sum_y = 0.0;
        self.fixation_sample_count = 0;
        self.saccade_start_x = 0.0;
        self.saccade_start_y = 0.0;
        self.saccade_start_time = None;
        self.saccade_peak_velocity = 0.0;
        self.low_velocity_start = None;
        self.confidence_drop_start = None;
    }

    /// Return the running centroid of the current fixation.
    fn fixation_centroid(&self) -> (f32, f32) {
        if self.fixation_sample_count == 0 {
            return (0.0, 0.0);
        }
        (
            self.fixation_sum_x / self.fixation_sample_count as f32,
            self.fixation_sum_y / self.fixation_sample_count as f32,
        )
    }

    // ------------------------------------------------------------------
    // State handlers
    // ------------------------------------------------------------------

    /// Handle a sample while in the `Unknown` state.
    fn handle_unknown(
        &mut self,
        velocity: f32,
        gaze_x: f32,
        gaze_y: f32,
        timestamp: Instant,
        events: &mut Vec<GazeEvent>,
    ) {
        if velocity < FIXATION_VELOCITY_THRESHOLD_DEG_S {
            // Start (or continue) the low-velocity dwell timer.
            if self.low_velocity_start.is_none() {
                self.low_velocity_start = Some(timestamp);
            }

            let dwell = timestamp
                .duration_since(self.low_velocity_start.unwrap());
            if dwell >= FIXATION_MIN_DURATION {
                // Sufficiently long low-velocity period → fixation.
                self.enter_fixation(gaze_x, gaze_y, timestamp, events);
            }
        } else {
            // Reset dwell – velocity is not low enough.
            self.low_velocity_start = None;

            if velocity > SACCADE_VELOCITY_THRESHOLD_DEG_S {
                // Rapid movement → saccade.
                self.enter_saccade(gaze_x, gaze_y, gaze_x, gaze_y, timestamp, velocity);
            }
        }
    }

    /// Handle a sample while in the `Fixation` state.
    fn handle_fixation(
        &mut self,
        velocity: f32,
        gaze_x: f32,
        gaze_y: f32,
        timestamp: Instant,
        events: &mut Vec<GazeEvent>,
    ) {
        if velocity > SACCADE_VELOCITY_THRESHOLD_DEG_S {
            // Ending fixation, starting a saccade.
            if let Some(fix_start) = self.fixation_start_time {
                let duration = timestamp.duration_since(fix_start);
                events.push(GazeEvent::FixationEnd {
                    centroid: self.fixation_centroid(),
                    duration,
                });
            }
            self.enter_saccade(self.fixation_centroid().0, self.fixation_centroid().1, gaze_x, gaze_y, timestamp, velocity);
        } else {
            // Still fixating – update the running centroid.
            self.fixation_sample_count += 1;
            self.fixation_sum_x += gaze_x;
            self.fixation_sum_y += gaze_y;
        }
    }

    /// Handle a sample while in the `Saccade` state.
    fn handle_saccade(
        &mut self,
        velocity: f32,
        gaze_x: f32,
        gaze_y: f32,
        timestamp: Instant,
        events: &mut Vec<GazeEvent>,
    ) {
        if velocity < FIXATION_VELOCITY_THRESHOLD_DEG_S {
            // Saccade has ended.
            if let Some(sacc_start) = self.saccade_start_time {
                let duration = timestamp.duration_since(sacc_start);
                if duration >= SACCADE_MIN_DURATION {
                    let amplitude = ((gaze_x - self.saccade_start_x).powi(2)
                        + (gaze_y - self.saccade_start_y).powi(2))
                    .sqrt();
                    events.push(GazeEvent::Saccade {
                        start: (self.saccade_start_x, self.saccade_start_y),
                        end: (gaze_x, gaze_y),
                        amplitude,
                        peak_velocity: self.saccade_peak_velocity,
                        duration,
                    });
                }
                // Saccades shorter than SACCADE_MIN_DURATION are silently
                // dropped (blink / noise rejection).
            }

            // Transition back to Unknown with a dwell timer so the
            // next few low-velocity samples can trigger a fixation.
            self.state = ClassifierState::Unknown;
            self.low_velocity_start = Some(timestamp);
        } else {
            // Still in the saccade – track peak velocity.
            if velocity > self.saccade_peak_velocity {
                self.saccade_peak_velocity = velocity;
            }
        }
    }

    /// Transition to the `Fixation` state and emit [`GazeEvent::FixationStart`].
    fn enter_fixation(
        &mut self,
        gaze_x: f32,
        gaze_y: f32,
        timestamp: Instant,
        events: &mut Vec<GazeEvent>,
    ) {
        self.state = ClassifierState::Fixation;
        self.fixation_start_time = Some(timestamp);
        self.fixation_sum_x = gaze_x;
        self.fixation_sum_y = gaze_y;
        self.fixation_sample_count = 1;

        events.push(GazeEvent::FixationStart {
            centroid: (gaze_x, gaze_y),
            timestamp,
        });
    }

    /// Transition to the `Saccade` state, recording the start position.
    ///
    /// `start_x/start_y` is where the eye was *before* the saccade began
    /// (usually the last fixation centroid).  `current_x/current_y` is the
    /// position that triggered the saccade detection.
    fn enter_saccade(
        &mut self,
        start_x: f32,
        start_y: f32,
        _current_x: f32,
        _current_y: f32,
        timestamp: Instant,
        velocity: f32,
    ) {
        self.state = ClassifierState::Saccade;
        self.saccade_start_x = start_x;
        self.saccade_start_y = start_y;
        self.saccade_start_time = Some(timestamp);
        self.saccade_peak_velocity = velocity;
        self.low_velocity_start = None;
    }
}

impl Default for GazeClassifier {
    fn default() -> Self {
        Self::new(DEFAULT_WINDOW_SIZE)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// After construction the classifier should report `Unknown` and
    /// `is_fixating() == false`.
    #[test]
    fn test_classifier_initial_state_is_unknown() {
        let classifier = GazeClassifier::new(10);
        assert_eq!(classifier.current_state(), GazeClassification::Unknown);
        assert!(!classifier.is_fixating());
    }

    /// When fed a stable gaze position over the dwell window the classifier
    /// should eventually emit a `FixationStart` and report `Fixation`.
    #[test]
    fn test_classifier_detects_fixation() {
        let mut classifier = GazeClassifier::new(5);
        let start = Instant::now();

        let mut fix_started = false;

        // Feed a stationary point with 20 ms spacing.
        for i in 0..20 {
            let t = start + Duration::from_millis(i as u64 * 20);
            let events = classifier.update(1.0, 0.0, t, 0.9);

            for ev in &events {
                if matches!(ev, GazeEvent::FixationStart { .. }) {
                    fix_started = true;
                }
            }
        }

        assert!(
            fix_started,
            "Should have started a fixation after ~400 ms of stable gaze",
        );
        assert!(
            classifier.is_fixating(),
            "Classifier should report fixating after stable gaze",
        );

        if let GazeClassification::Fixation { centroid, duration } = classifier.current_state() {
            assert!(
                duration >= Duration::from_millis(100),
                "Fixation duration should be at least 100 ms, got {:?}",
                duration,
            );
            assert!(
                (centroid.0 - 1.0).abs() < 0.5,
                "Fixation centroid x should be near 1.0, got {}",
                centroid.0,
            );
        } else {
            panic!("Expected Fixation classification, got {:?}", classifier.current_state());
        }
    }

    /// After establishing a fixation, a rapid position jump should trigger a
    /// saccade detection (emitting `FixationEnd` + eventually `Saccade`).
    #[test]
    fn test_classifier_detects_saccade() {
        let mut classifier = GazeClassifier::new(5);
        let start = Instant::now();

        // ── Phase 1: establish a fixation ──
        for i in 0..12 {
            let t = start + Duration::from_millis(i as u64 * 20);
            classifier.update(1.0, 0.0, t, 0.9);
        }
        assert!(
            classifier.is_fixating(),
            "Should be fixating after stable gaze",
        );

        // ── Phase 2: rapid jump ──
        let jump_time = start + Duration::from_millis(12 * 20);
        let events = classifier.update(-0.5, 0.5, jump_time, 0.9);

        let had_fixation_end = events.iter().any(|e| matches!(e, GazeEvent::FixationEnd { .. }));
        assert!(
            had_fixation_end,
            "Rapid position change should end the current fixation",
        );

        // ── Phase 3: let the window flush the high-velocity pair ──
        let mut saccade_emitted = false;
        for i in 0..15 {
            let t = jump_time + Duration::from_millis((i + 1) as u64 * 20);
            let evs = classifier.update(-0.5, 0.5, t, 0.9);
            for ev in &evs {
                if let GazeEvent::Saccade {
                    start,
                    end,
                    amplitude,
                    peak_velocity,
                    ..
                } = ev
                {
                    saccade_emitted = true;
                    // The amplitude should be non-trivial.
                    assert!(*amplitude > 0.5, "Saccade amplitude should be > 0.5, got {amplitude}");
                    assert!(*peak_velocity > 50.0, "Peak velocity should exceed 50°/s, got {peak_velocity}");
                    // Start/end should straddle the jump.
                    assert!((start.0 - 1.0).abs() < 0.1 || (end.0 - 1.0).abs() < 0.1);
                }
            }
        }

        assert!(
            saccade_emitted,
            "Should have emitted a Saccade event after the rapid position change",
        );
    }

    /// After `reset()` the classifier should be back in the `Unknown` state
    /// with an empty window.
    #[test]
    fn test_classifier_reset_clears_state() {
        let mut classifier = GazeClassifier::new(5);
        let start = Instant::now();

        // Feed enough samples to enter a fixation.
        for i in 0..15 {
            let t = start + Duration::from_millis(i as u64 * 20);
            classifier.update(1.0, 0.0, t, 0.9);
        }
        assert!(classifier.is_fixating());

        // Reset.
        classifier.reset();

        assert_eq!(classifier.current_state(), GazeClassification::Unknown);
        assert!(!classifier.is_fixating());
        assert!(
            classifier.samples.is_empty(),
            "Sample window should be empty after reset",
        );

        // After reset, classifier should start fresh.
        let t = start + Duration::from_millis(500);
        let events = classifier.update(1.0, 0.0, t, 0.9);
        assert!(
            events.is_empty(),
            "First post-reset update should not emit events (not enough samples yet)",
        );
    }

    /// Low-confidence samples followed by recovery within 300 ms should be
    /// treated as a blink and must NOT trigger a false saccade.
    #[test]
    fn test_blink_rejection() {
        let mut classifier = GazeClassifier::new(5);
        let start = Instant::now();

        // ── Phase 1: establish a fixation ──
        for i in 0..12 {
            let t = start + Duration::from_millis(i as u64 * 20);
            classifier.update(1.0, 0.0, t, 0.9);
        }
        assert!(classifier.is_fixating());

        // ── Phase 2: brief confidence drop (simulated blink) ──
        let blink_start = start + Duration::from_millis(12 * 20);
        // 3 samples with very low confidence, spaced 20 ms apart (= 60 ms < 300 ms)
        let mut blink_events = 0;
        for i in 0..3 {
            let t = blink_start + Duration::from_millis(i as u64 * 20);
            let evs = classifier.update(1.0, 0.0, t, 0.05);
            blink_events += evs.len();
        }
        assert_eq!(
            blink_events, 0,
            "No events should be emitted during low-confidence period",
        );

        // ── Phase 3: recovery (still within 300 ms blink window) ──
        let recovery_start = blink_start + Duration::from_millis(3 * 20); // 60 ms after blink
        let recovery_events = classifier.update(1.0, 0.0, recovery_start, 0.9);
        assert!(
            recovery_events.is_empty(),
            "Blink recovery should suppress all events (got {} events)",
            recovery_events.len(),
        );

        // ── Phase 4: confidence stays high – classifier should re-acquire ──
        for i in 0..15 {
            let t = recovery_start + Duration::from_millis((i + 1) as u64 * 20);
            let evs = classifier.update(1.0, 0.0, t, 0.9);
            // We should see a new FixationStart (not a saccade).
            for ev in &evs {
                assert!(
                    !matches!(ev, GazeEvent::Saccade { .. }),
                    "Blink recovery should not generate a saccade event",
                );
            }
        }

        // Should eventually re-enter fixation.
        assert!(
            classifier.is_fixating(),
            "Should re-enter fixation after blink recovery",
        );
    }
}
