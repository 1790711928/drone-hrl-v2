from src.env.termination import (
    EpisodeOutcome,
    TerminationConfig,
    TerminationState,
    evaluate_termination,
)


def test_escape_requires_k_steps_streak() -> None:
    cfg = TerminationConfig(k_escape=3, d_safe=10.0)
    tstate = TerminationState()

    out1 = evaluate_termination(
        distance=11.0,
        closing_speed=-0.1,
        los_escape_ok=True,
        evader_position=(0.0, 0.0, 5.0),
        step_count=1,
        cfg=cfg,
        tstate=tstate,
    )
    out2 = evaluate_termination(
        distance=11.5,
        closing_speed=-0.2,
        los_escape_ok=True,
        evader_position=(0.0, 0.0, 5.0),
        step_count=2,
        cfg=cfg,
        tstate=tstate,
    )
    out3 = evaluate_termination(
        distance=12.0,
        closing_speed=0.0,
        los_escape_ok=True,
        evader_position=(0.0, 0.0, 5.0),
        step_count=3,
        cfg=cfg,
        tstate=tstate,
    )

    assert out1 == EpisodeOutcome.RUNNING
    assert out2 == EpisodeOutcome.RUNNING
    assert out3 == EpisodeOutcome.ESCAPED


def test_capture_streak() -> None:
    cfg = TerminationConfig(k_capture=2, d_capture=2.5)
    tstate = TerminationState()

    out1 = evaluate_termination(
        distance=2.0,
        closing_speed=0.5,
        los_escape_ok=False,
        evader_position=(0.0, 0.0, 5.0),
        step_count=1,
        cfg=cfg,
        tstate=tstate,
    )
    out2 = evaluate_termination(
        distance=2.2,
        closing_speed=0.3,
        los_escape_ok=False,
        evader_position=(0.0, 0.0, 5.0),
        step_count=2,
        cfg=cfg,
        tstate=tstate,
    )

    assert out1 == EpisodeOutcome.RUNNING
    assert out2 == EpisodeOutcome.CAPTURED
