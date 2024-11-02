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

# Development
See instructions in https://github.com/AllenInstitute/npc_mvr/CONTRIBUTING.md and the original template: https://github.com/AllenInstitute/copier-pdm-npc/blob/main/README.md