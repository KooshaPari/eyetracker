from __future__ import annotations

from agent_user_status.gaze_evaluation import (
    REPEATED_GAZE_SAMPLE,
    STUCK_GAZE_SAMPLE,
    EvaluationCounters,
    GazeSampleStuckDetector,
)


def test_evaluation_counters_report_rejection_reasons_per_target() -> None:
    counters = EvaluationCounters()
    first = counters.begin_target(1, 100, 100)
    first.reject("settling")
    first.reject("settling")
    first.reject("low_confidence")
    first.accept((103.0, 104.0))

    second = counters.begin_target(2, 500, 400)
    second.reject("no_face_sample")
    second.accept((700.0, 400.0))

    summary = counters.summary(hold_threshold_px=120.0)

    assert summary["accepted_total"] == 2
    assert summary["rejected_total"] == 4
    assert summary["rejected_by_reason"] == {
        "low_confidence": 1,
        "no_face_sample": 1,
        "settling": 2,
    }
    assert summary["projection_hold_candidate_count"] == 1
    assert summary["targets"][0]["accepted"] == 1
    assert summary["targets"][0]["rejected"]["settling"] == 2


def test_stuck_detector_reports_repeated_then_stuck_derived_coordinates() -> None:
    detector = GazeSampleStuckDetector(repeated_threshold=2, stuck_threshold=4)

    assert detector.inspect((320.2, 240.4)) is None
    assert detector.inspect((320.2, 240.4)) == REPEATED_GAZE_SAMPLE
    assert detector.inspect((320.3, 240.1)) == REPEATED_GAZE_SAMPLE
    assert detector.inspect((320.4, 240.2)) == STUCK_GAZE_SAMPLE
    assert detector.summary() == {
        "repeated_gaze_sample_count": 2,
        "stuck_gaze_sample_count": 1,
    }


def test_evaluation_counters_include_sample_health_totals() -> None:
    counters = EvaluationCounters()
    target = counters.begin_target(1, 100, 100)
    for _ in range(4):
        reason = counters.inspect_observed_sample((100.0, 100.0))
        if reason:
            target.reject(reason)

    summary = counters.summary(hold_threshold_px=120.0)

    assert summary["repeated_gaze_sample_count"] == 2
    assert summary["stuck_gaze_sample_count"] == 1
    assert summary["rejected_by_reason"] == {
        REPEATED_GAZE_SAMPLE: 2,
        STUCK_GAZE_SAMPLE: 1,
    }


def test_stuck_detector_reset_starts_new_sample_sequence() -> None:
    detector = GazeSampleStuckDetector(repeated_threshold=2, stuck_threshold=3)

    assert detector.inspect((10.0, 10.0)) is None
    assert detector.inspect((10.0, 10.0)) == REPEATED_GAZE_SAMPLE
    detector.reset()

    assert detector.inspect((10.0, 10.0)) is None
    assert detector.summary() == {
        "repeated_gaze_sample_count": 1,
        "stuck_gaze_sample_count": 0,
    }


def test_stuck_detector_precision_buckets_nearby_derived_coordinates() -> None:
    detector = GazeSampleStuckDetector(
        repeated_threshold=2,
        stuck_threshold=4,
        coordinate_precision_px=5.0,
    )

    assert detector.inspect((100.1, 200.2)) is None
    assert detector.inspect((101.9, 199.7)) == REPEATED_GAZE_SAMPLE
    assert detector.inspect((107.0, 199.7)) is None
