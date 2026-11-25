# npc_mvr

Tools for reading raw video data from MindScope Neuropixels experiments, compatible with data in the cloud.

[![PyPI](https://img.shields.io/pypi/v/npc_mvr.svg?label=PyPI&color=blue)](https://pypi.org/project/npc_mvr/)
[![Python version](https://img.shields.io/pypi/pyversions/npc_mvr)](https://pypi.org/project/npc_mvr/)

[![Coverage](https://img.shields.io/codecov/c/github/AllenInstitute/npc_mvr?logo=codecov)](https://app.codecov.io/github/AllenInstitute/npc_mvr)
[![CI/CD](https://img.shields.io/github/actions/workflow/status/AllenInstitute/npc_mvr/publish.yml?label=CI/CD&logo=github)](https://github.com/AllenInstitute/npc_mvr/actions/workflows/publish.yml)
[![GitHub issues](https://img.shields.io/github/issues/AllenInstitute/npc_mvr?logo=github)](https://github.com/AllenInstitute/npc_mvr/issues)

# Usage
```bash
conda create -n npc_mvr python>=3.9
conda activate npc_mvr
pip install npc_mvr
```

## Python
```python
>>> import npc_mvr

>>> d = npc_mvr.MVRDataset('s3://aind-ephys-data/ecephys_670248_2023-08-03_12-04-15/behavior')

# get paths
>>> d.video_paths['behavior']
S3Path('s3://aind-ephys-data/ecephys_670248_2023-08-03_12-04-15/behavior/Behavior_20230803T120430.mp4')
>>> d.info_paths['behavior']
S3Path('s3://aind-ephys-data/ecephys_670248_2023-08-03_12-04-15/behavior/Behavior_20230803T120430.json')

# get data
>>> type(d.video_data['behavior'])
<class 'cv2.VideoCapture'>
>>> type(d.info_data['behavior'])
<class 'dict'>
>>> type(d.sync_data)
<class 'npc_sync.sync.SyncDataset'>

# get frame times for each camera on sync clock
# - nans correspond to frames not recorded on sync
# - first nan is metadata frame 
>>> d.frame_times['behavior']
array([       nan,   14.08409,   14.10075, ..., 5084.4582 , 5084.47487, 5084.49153]) 

>>> d.validate()
```

## Excluded cameras
`MVRDataset.video_paths` (and `frame_times`, `video_data` etc.) leaves out cameras whose
frame times can't be trusted, with a warning. The reasons are in `d.excluded_cameras`, and
all videos found are in `d.all_video_paths`:
- more than `max_frames_cut` (default 20) surplus exposures on sync that MVR doesn't account
  for (frames lost without being reported, so frame times drift)
- more than `max_lost_frame_percentage` (default 0.1%) frames lost
- the recording started inside an interval where sync events are unknown

Pass `None` for either limit to disable it. Counts for each camera are in
`d.camera_frame_times` (see `npc_mvr.CameraFrameTimes`).

## Unknown sync intervals
With a patched `npc_sync.SyncDataset` (or `on_invalid='mask'`), camera exposures inside
intervals where sync events are unknown (`SyncDataset.get_unknown_intervals`) are estimated
from the camera's period: their frame times are nan, and frames after them keep the right
times.

# Development
See instructions in https://github.com/AllenInstitute/npc_mvr/CONTRIBUTING.md and the original template: https://github.com/AllenInstitute/copier-pdm-npc/blob/main/README.md