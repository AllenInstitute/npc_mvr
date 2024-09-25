import numpy as np
import npc_mvr

def test_long_sync_sessions() -> None:
    """Sessions where sync + mvr were started up to 4 hours before the main
    experiment (with mvr being stopped/restarted partway through)
    - see https://github.com/AllenInstitute/npc_sessions/issues/114
    """
    for raw_data_path in (
        's3://aind-ephys-data/ecephys_660023_2023-08-08_07-58-13',
        's3://aind-ephys-data/ecephys_666986_2023-08-15_08-13-00',
        's3://aind-ephys-data/ecephys_644867_2023-02-23_12-14-29',
        's3://aind-ephys-data/ecephys_660023_2023-08-09_08-09-09',
    ):
        mvr = npc_mvr.MVRDataset(raw_data_path)
        # this will error if there are problems with processing
        frame_times = npc_mvr.get_video_frame_times(
            mvr.sync_data,
            *mvr.video_paths.values(),
            apply_correction=False,
        )
        frame_time_lengths = [len(ft) for ft in frame_times.values()]
        assert np.all(np.abs(frame_time_lengths - np.mean(frame_time_lengths) < 60 * 10)), f"frame time lengths different between cameras: {frame_time_lengths}"
    
if __name__ == '__main__':
    import pytest
    
    pytest.main(['-v', __file__])