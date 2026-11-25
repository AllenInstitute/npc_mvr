import numpy as np
import upath

import npc_mvr


def _camera(num_exposures: int, num_frames_recorded: int, lost: int) -> npc_mvr.CameraFrameTimes:
    times = np.insert(np.arange(num_exposures - lost, dtype=float), 0, np.nan)
    return npc_mvr.CameraFrameTimes(
        camera_name="eye",
        video_path=upath.UPath("Eye_20230101T000000.mp4"),
        sync_line="eye_cam_exposing",
        frame_times=times,
        num_frames_in_video=num_frames_recorded + 1,
        num_frames_recorded=num_frames_recorded,
        num_lost_frames=lost,
        num_exposures=num_exposures,
        num_exposures_inserted=0,
        unknown_intervals=(),
    )


def test_clean_camera_is_padded_not_excluded() -> None:
    c = _camera(num_exposures=1000, num_frames_recorded=1020, lost=0)
    assert c.num_frames_cut == -20
    assert c.get_exclusion_reason() is None
    t = c.corrected_frame_times()
    assert len(t) == 1021 and np.isnan(t[-20:]).all() and np.isnan(t[0])


def test_unaccounted_exposures_exclude_camera() -> None:
    c = _camera(num_exposures=100_300, num_frames_recorded=100_000, lost=0)
    assert c.num_unaccounted_frames == c.num_frames_cut == 300
    assert "cut" in c.get_exclusion_reason()
    assert c.get_exclusion_reason(max_frames_cut=None) is None


def test_lost_frames_exclude_camera() -> None:
    c = _camera(num_exposures=100_000, num_frames_recorded=99_800, lost=200)
    assert c.num_frames_cut == 0
    assert "lost" in c.get_exclusion_reason()
    assert c.get_exclusion_reason(max_lost_frame_percentage=None) is None


def test_fill_unknown_intervals_keeps_positions() -> None:
    period = 1 / 60
    edges = np.arange(0, 10, period)
    unknown = (4.0, 4.5)
    recorded = edges[(edges < unknown[0]) | (edges > unknown[1])]
    filled, inserted = npc_mvr.fill_unknown_intervals(recorded, [unknown])
    assert len(filled) == len(edges)
    np.testing.assert_allclose(filled, edges)
    assert inserted.sum() == len(edges) - len(recorded)


def test_fill_unknown_intervals_ignores_idle_camera() -> None:
    edges = np.concatenate([np.arange(0, 1, 0.1), np.arange(100, 101, 0.1)])
    filled, inserted = npc_mvr.fill_unknown_intervals(edges, [(50, 60)])
    assert not inserted.any() and len(filled) == len(edges)


if __name__ == "__main__":
    import pytest

    pytest.main(["-v", __file__])
