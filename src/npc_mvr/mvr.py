from __future__ import annotations

import datetime
import functools
import json
import logging
from collections.abc import Container, Iterable, Mapping
from typing import Any, Literal, TypeVar

import cv2
import npc_io
import npc_sync
import numpy as np
import numpy.typing as npt
import upath
from typing_extensions import TypeAlias

logger = logging.getLogger(__name__)


MVRInfoData: TypeAlias = Mapping[str, Any]
"""Contents of `RecordingReport` from a camera's info.json for an MVR
recording."""

CameraName: TypeAlias = Literal["eye", "face", "behavior"]
CameraNameOnSync: TypeAlias = Literal["eye", "face", "beh"]


class MVRDataset:
    """A collection of paths + data for processing the output from MVR for one
    session.

    Expectations:

    - 3 .mp4/.avi video file paths (eye, face, behavior)
    - 3 .json info file paths (eye, face, behavior)
    - the associated data as Python objects for each of the above (e.g mp3 -> CV2,
    json -> dict)

    - 1 sync file path (h5)
    - sync data as a SyncDataset object

    Assumptions:
    - all files live in the same directory (so we can initialize with a single path)
    - MVR was started after sync
    - MVR may have been stopped before sync

    >>> import npc_mvr

    >>> d = npc_mvr.MVRDataset('s3://aind-ephys-data/ecephys_670248_2023-08-03_12-04-15')

    # get paths
    >>> d.video_paths['behavior']
    S3Path('s3://aind-ephys-data/ecephys_670248_2023-08-03_12-04-15/behavior_videos/Behavior_20230803T120430.mp4')
    >>> d.info_paths['behavior']
    S3Path('s3://aind-ephys-data/ecephys_670248_2023-08-03_12-04-15/behavior_videos/Behavior_20230803T120430.json')
    >>> d.sync_path
    S3Path('s3://aind-ephys-data/ecephys_670248_2023-08-03_12-04-15/behavior/20230803T120415.h5')

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
    array([     nan, 14.08409, 14.10075, ...,      nan,      nan,      nan])
    >>> d.validate()
    """

    def __init__(
        self, session_dir: npc_io.PathLike, sync_path: npc_io.PathLike | None = None
    ) -> None:
        self.session_dir = npc_io.from_pathlike(session_dir)
        if sync_path is not None:
            self.sync_path = npc_io.from_pathlike(sync_path)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.session_dir})"

    @property
    def session_dir(self) -> upath.UPath:
        return self._session_dir

    @session_dir.setter
    def session_dir(self, value: npc_io.PathLike) -> None:
        path = npc_io.from_pathlike(value)
        if path.name in ("behavior", "behavior_videos", "behavior-videos"):
            path = path.parent
            logger.debug(
                f"Setting session directory as {path}: after March 2024 video and sync no longer stored together"
            )
        self._session_dir = path

    @npc_io.cached_property
    def is_cloud(self) -> bool:
        return self.session_dir.protocol not in ("file", "")

    @npc_io.cached_property
    def sync_dir(self) -> upath.UPath:
        if (path := self.session_dir / "behavior").exists():
            return path
        return self.session_dir

    @npc_io.cached_property
    def video_dir(self) -> upath.UPath:
        if not self.is_cloud:
            return self.session_dir
        for name in ("behavior_videos", "behavior-videos", "behavior"):
            if (path := self.session_dir / name).exists():
                return path
        return self.session_dir

    @npc_io.cached_property
    def frame_times(self) -> dict[CameraName, npt.NDArray[np.float64]]:
        """Returns frametimes (in seconds) as measured on sync clock for each
        camera.

        - see `get_video_frame_times` for more details
        """
        return {
            get_camera_name(p.stem): times
            for p, times in get_video_frame_times(
                self.sync_data, *self.video_paths.values()
            ).items()
        }

    @npc_io.cached_property
    def video_paths(self) -> dict[CameraName, upath.UPath]:
        return {
            get_camera_name(p.stem): p for p in get_video_file_paths(self.video_dir)
        }

    @npc_io.cached_property
    def video_data(self) -> npc_io.LazyDict[CameraName, cv2.VideoCapture]:
        return npc_io.LazyDict(
            (camera_name, (get_video_data, (path,), {}))
            for camera_name, path in self.video_paths.items()
        )

    @npc_io.cached_property
    def info_paths(self) -> dict[CameraName, upath.UPath]:
        return {
            get_camera_name(p.stem): p
            for p in get_video_info_file_paths(self.video_dir)
        }

    @npc_io.cached_property
    def info_data(self) -> dict[CameraName, MVRInfoData]:
        return {
            camera_name: get_video_info_data(path)
            for camera_name, path in self.info_paths.items()
        }

    @npc_io.cached_property
    def sync_path(self) -> upath.UPath:
        return npc_sync.get_single_sync_path(self.sync_dir)

    @npc_io.cached_property
    def sync_data(self) -> npc_sync.SyncDataset:
        return npc_sync.get_sync_data(self.sync_path)

    @npc_io.cached_property
    def video_start_times(self) -> dict[CameraName, datetime.datetime]:
        """Naive datetime of when the video recording started.
        - can be compared to `sync_data.start_time` to check if MVR was started
          after sync.
        """
        return {
            camera_name: datetime.datetime.fromisoformat(
                self.info_data[camera_name]["TimeStart"][:-1]
            )  # discard 'Z'
            for camera_name in self.info_data
        }

    @npc_io.cached_property
    def augmented_camera_info(self) -> dict[CameraName, dict[str, Any]]:
        cam_exposing_times = get_cam_exposing_times_on_sync(self.sync_data)
        cam_transfer_times = get_cam_transfer_times_on_sync(self.sync_data)
        cam_exposing_falling_edge_times = get_cam_exposing_falling_edge_times_on_sync(
            self.sync_data
        )
        augmented_camera_info = {}
        for camera_name, video_path in self.video_paths.items():
            camera_info = dict(self.info_data[camera_name])  # copy
            frames_lost = camera_info["FramesLostCount"]

            num_exposures = cam_exposing_times[camera_name].size
            num_transfers = cam_transfer_times[camera_name].size

            num_frames_in_video = get_total_frames_in_video(video_path)
            num_expected_from_sync = num_transfers - frames_lost + 1
            signature_exposures = (
                cam_exposing_falling_edge_times[camera_name][:10]
                - cam_exposing_times[camera_name][:10]
            )

            camera_info["num_frames_exposed"] = num_exposures
            camera_info["num_frames_transfered"] = num_transfers
            camera_info["num_frames_in_video"] = num_frames_in_video
            camera_info["num_frames_expected_from_sync"] = num_expected_from_sync
            camera_info["expected_minus_actual"] = (
                num_expected_from_sync - num_frames_in_video
            )
            camera_info["num_frames_from_sync"] = len(
                get_video_frame_times(
                    self.sync_path,
                    self.video_paths[camera_name],
                    apply_correction=False,
                )[self.video_paths[camera_name]]
            )
            camera_info["signature_exposure_duration"] = np.round(
                np.median(signature_exposures), 3
            )
            camera_info["lost_frame_percentage"] = (
                100 * camera_info["FramesLostCount"] / camera_info["FramesRecorded"]
            )
            augmented_camera_info[camera_name] = camera_info
        return augmented_camera_info

    @npc_io.cached_property
    def num_lost_frames_from_barcodes(self) -> dict[CameraName, int]:
        """Get the frame ID from the barcode in the last frame of the video: if no
        frames were lost, this should be equal to the number of frames in the
        video minus 1 (the frame ID is 0-indexed)."""
        cam_to_frames = {}
        for camera_name in self.video_data:
            video_data = self.video_data[camera_name]
            video_info = self.info_data[camera_name]
            actual_last_frame_index = int(video_data.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
            # get the last frame id from the video file
            try:
                last_frame_barcode_value: int = get_frame_number_from_barcode(video_data, video_info, frame_number=actual_last_frame_index)
            except ValueError as exc:
                raise AttributeError(f"Video file {self.video_paths[camera_name]} does not have barcodes in frames") from exc
            num_lost_frames = last_frame_barcode_value - actual_last_frame_index
            cam_to_frames[camera_name] = int(num_lost_frames)
        return cam_to_frames

    @npc_io.cached_property
    def lick_frames(self) -> npt.NDArray[np.intp]:
        if self.sync_path:
            lick_times = self.sync_data.get_rising_edges("lick_sensor", units="seconds")
            return np.array([
                np.nanargmin(np.abs(self.frame_times["behavior"] - time))
                for time in lick_times
            ])
        else:
            try:    
                return get_lick_frames_from_behavior_info(self.info_data['behavior'])
            except ValueError as exc:
                raise AttributeError("Lick frames not recorded in MVR in this session") from exc
        
    def validate(self) -> None:
        """Check all data required for processing is present and consistent. Check dropped frames
        count."""
        for camera in self.video_paths:
            video = self.video_data[camera]
            info_json = self.info_data[camera]
            augmented_info = self.augmented_camera_info[camera]
            times = self.frame_times[camera]

            if not times.any() or np.isnan(times).all():
                raise AssertionError(f"No frames recorded on sync for {camera}")
            if (a := video.get(cv2.CAP_PROP_FRAME_COUNT)) - (
                b := info_json["FramesRecorded"]
            ) > 1:
                # metadata frame is added to the video file, so the difference should be 1
                raise AssertionError(
                    f"Frame count from {camera} video file ({a}) does not match info.json ({b})"
                )
            if self.video_start_times[camera] < self.sync_data.start_time:
                raise AssertionError(
                    f"Video start time is before sync start time for {camera}"
                )
            if hasattr(self, "num_lost_frames_from_barcodes"):
                num_frames_lost_in_json = len(get_lost_frames_from_camera_info(info_json))
                if num_frames_lost_in_json != (b := self.num_lost_frames_from_barcodes[camera]):
                    raise AssertionError(
                        f"Lost frame count from frame barcodes ({b}) does not match `FramesLostCount` in info.json ({num_frames_lost_in_json}) for {camera=}"
                    )
            if not is_acceptable_frame_rate(info_json["FPS"]):
                raise AssertionError(f"Invalid frame rate: {info_json['FPS']=}")

            if not is_acceptable_lost_frame_percentage(
                augmented_info["lost_frame_percentage"]
            ):
                raise AssertionError(
                    f"Lost frame percentage too high: {augmented_info['lost_frame_percentage']=}"
                )

            if not is_acceptable_expected_minus_actual_frame_count(
                augmented_info["expected_minus_actual"]
            ):
                # if number of frame times on sync matches the number expected, this isn't a hard failure
                if (
                    augmented_info["num_frames_expected_from_sync"]
                    != augmented_info["num_frames_from_sync"]
                ):
                    raise AssertionError(
                        f"Expected minus actual frame count too high: {augmented_info['expected_minus_actual']=}"
                    )


def is_acceptable_frame_rate(frame_rate: float) -> bool:
    return abs(frame_rate - 60) <= 0.05


def is_acceptable_lost_frame_percentage(lost_frame_percentage: float) -> bool:
    return lost_frame_percentage < 0.05


def is_acceptable_expected_minus_actual_frame_count(
    expected_minus_actual: int | float,
) -> bool:
    return abs(expected_minus_actual) < 20


def get_camera_name(path: str) -> CameraName:
    names: dict[str, CameraName] = {
        "eye": "eye",
        "face": "face",
        "beh": "behavior",
    }
    try:
        return names[next(n for n in names if n in str(path).lower())]
    except StopIteration as exc:
        raise ValueError(f"Could not extract camera name from {path}") from exc

def get_camera_name_on_sync(sync_line: str) -> CameraNameOnSync:
    """Camera name as used in sync line labels (`beh`, `eye`, `face`)."""
    name = get_camera_name(sync_line)
    return 'beh' if name == 'behavior' else name

@functools.cache
def get_camera_sync_line_name_mapping(
    sync_path_or_dataset: npc_io.PathLike | npc_sync.SyncDataset,
    *video_paths: npc_io.PathLike,
) -> dict[CameraName, CameraNameOnSync]:
    """Detects if cameras are plugged into sync correctly and returns a mapping
    of camera names to the camera name on sync that actually corresponds, so that this function can
    be used to wrap any access of line data.
    
    >>> m = MVRDataset('s3://aind-private-data-prod-o5171v/ecephys_703333_2024-04-09_13-06-44')
    >>> get_camera_sync_line_name_mapping(m.sync_path, *m.video_paths.values())
    {'behavior': 'beh', 'face': 'eye', 'eye': 'face'}
    """
    sync_data = npc_sync.get_sync_data(sync_path_or_dataset)
    jsons = get_video_info_file_paths(*video_paths)
    camera_to_json_data = {
        get_camera_name(path.stem): get_video_info_data(path) for path in jsons
    }
    camera_names_on_sync = ('beh', 'face', 'eye')
    def get_exposure_fingerprint_durations_from_jsons() -> dict[str, int]:
        """Nominally expected exposure time in milliseconds for each camera, as
        recorded in info jsons."""
        return {
            f"{camera_name}_cam_exposing": camera_to_json_data[get_camera_name(camera_name)]['CustomInitialExposureTime']
            for camera_name in camera_names_on_sync
        }
        
    def get_exposure_fingerprint_durations_from_sync() -> dict[str, int]:
        """Initial fingerpring exposure time in milliseconds for each camera, as recorded on sync clock."""
        return {
            (n := f"{camera_name}_cam_exposing"): round(
                (
                    sync_data.get_falling_edges(n, units="seconds")[:8]
                    - sync_data.get_rising_edges(n, units="seconds")[:8]
                ).mean()*1000
            )
            for camera_name in camera_names_on_sync
        }

    def get_start_times_on_sync() -> dict[str, float]:
        return {
            f"{camera_name}{line_suffix}": sync_data.get_rising_edges(f"{camera_name}{line_suffix}", units="seconds")[0]
            for camera_name in camera_names_on_sync
            for line_suffix in ('_cam_exposing', '_cam_frame_readout')
        }
    start_times_on_sync = get_start_times_on_sync()
    lines_sorted_by_start_time = tuple(sorted(start_times_on_sync, key=start_times_on_sync.get))
    expected_exposure_fingerprint_durations = get_exposure_fingerprint_durations_from_jsons()
    actual_exposure_fingerprint_durations = get_exposure_fingerprint_durations_from_sync()
    expected_to_actual_line_mapping: dict[CameraName, CameraNameOnSync] = {}
    for sync_camera_name in camera_names_on_sync:
        exposing_line = f"{sync_camera_name}_cam_exposing"
        expected_duration = expected_exposure_fingerprint_durations[exposing_line]
        actual_line = min(
            actual_exposure_fingerprint_durations,
            key=lambda line: abs(expected_duration - actual_exposure_fingerprint_durations[line])
        )
        expected_to_actual_line_mapping[get_camera_name(sync_camera_name)] = get_camera_name_on_sync(actual_line)
        readout_line = f"{sync_camera_name}_cam_frame_readout"
        assert (a := lines_sorted_by_start_time.index(start_times_on_sync[exposing_line])) + 1 == (b := lines_sorted_by_start_time.index(start_times_on_sync[readout_line])), (
            f"Expected {readout_line} (start index {a}) to start immediately after {exposing_line} (start index {b}) - assumption is incorrect (are lines connected to sync separately?)"
        )
    return expected_to_actual_line_mapping
        
def get_video_frame_times(
    sync_path_or_dataset: npc_io.PathLike | npc_sync.SyncDataset,
    *video_paths: npc_io.PathLike,
    apply_correction: bool = True,
) -> dict[upath.UPath, npt.NDArray[np.float64]]:
    """Returns frametimes as measured on sync clock for each video file.

    If a single directory is passed, video files in that directory will be
    found. If multiple paths are passed, the video files will be filtered out.

    - keys are video file paths
    - values are arrays of frame times in seconds
    - the first frametime will be a nan value (corresponding to a metadata frame)
    - frames at the end may also be nan values:
        MVR previously ceased all TTL pulses before the recording was
        stopped, resulting in frames in the video that weren't registered
        in sync. MVR was fixed July 2023 after Corbett discovered the issue.

        (only applied if `apply_correction` is True)

    - frametimes from sync may be cut to match the number of frames in the video:
        after July 2023, we started seeing video files that had fewer frames than
        timestamps in sync file.

        (only applied if `apply_correction` is True)

    >>> sync_path = 's3://aind-private-data-prod-o5171v/ecephys_708019_2024-03-22_15-33-01/behavior/20240322T153301.h5'
    >>> video_path = 's3://aind-private-data-prod-o5171v/ecephys_708019_2024-03-22_15-33-01/behavior-videos'
    >>> frame_times = get_video_frame_times(sync_path, video_path)
    >>> [len(frames) for frames in frame_times.values()]
    [103418, 103396, 103406]
    >>> sync_path = 's3://aind-ephys-data/ecephys_670248_2023-08-03_12-04-15/behavior/20230803T120415.h5'
    >>> video_path = 's3://aind-ephys-data/ecephys_670248_2023-08-03_12-04-15/behavior_videos'
    >>> frame_times = get_video_frame_times(sync_path, video_path)
    >>> [len(frames) for frames in frame_times.values()]
    [304233, 304240, 304236]
    """
    videos = get_video_file_paths(*video_paths)
    jsons = get_video_info_file_paths(*video_paths)
    camera_to_video_path = {get_camera_name(path.stem): path for path in videos}
    camera_to_json_data = {
        get_camera_name(path.stem): get_video_info_data(path) for path in jsons
    }
    correct_sync_line_names = get_camera_sync_line_name_mapping(sync_path_or_dataset, *videos)
    if tuple(correct_sync_line_names.keys()) != tuple(correct_sync_line_names.values()):
        logger.warning(f"Camera lines are plugged into sync incorrectly - we'll accommodate for this, but if this is a recent session check the rig: {correct_sync_line_names}")
    camera_exposing_times = get_cam_exposing_times_on_sync(sync_path_or_dataset)
    camera_exposing_times = {
        camera: camera_exposing_times[get_camera_name(correct_sync_line_names[camera])]
        for camera in camera_exposing_times
    }
    frame_times: dict[upath.UPath, npt.NDArray[np.floating]] = {}
    for camera in camera_exposing_times:
        if camera in camera_to_video_path:
            num_frames_in_video = get_total_frames_in_video(
                camera_to_video_path[camera]
            )
            camera_frame_times = remove_lost_frame_times(
                camera_exposing_times[camera],
                get_lost_frames_from_camera_info(camera_to_json_data[camera]),
            )
            # Insert a nan frame time at the beginning to account for metadata frame
            camera_frame_times = np.insert(camera_frame_times, 0, np.nan)
            # append nan frametimes for frames in the video file but are
            # unnaccounted for on sync:
            if (
                apply_correction
                and (
                    frames_missing_from_sync := num_frames_in_video
                    - len(camera_frame_times)
                )
                > 0
            ):
                camera_frame_times = np.append(
                    camera_frame_times,
                    np.full(frames_missing_from_sync, np.nan),
                )
            # cut times of sync events that don't correspond to frames in the video:
            elif apply_correction and (len(camera_frame_times) > num_frames_in_video):
                camera_frame_times = camera_frame_times[:num_frames_in_video]
            if apply_correction:
                assert len(camera_frame_times) == num_frames_in_video, (
                    f"Expected {num_frames_in_video} frame times, got {len(camera_frame_times)} "
                    f"for {camera_to_video_path[camera]}"
                    f"{'' if apply_correction else ' (try getting frametimes with `apply_correction=True`)'}"
                )
            frame_times[camera_to_video_path[camera]] = camera_frame_times
    return frame_times


def get_cam_line_times_on_sync(
    sync_path_or_dataset: npc_io.PathLike | npc_sync.SyncDataset,
    sync_line_suffix: str,
    edge_type: Literal["rising", "falling"] = "rising",
) -> dict[Literal["behavior", "eye", "face"], npt.NDArray[np.float64]]:
    sync_data = npc_sync.get_sync_data(sync_path_or_dataset)

    edge_getter = (
        sync_data.get_rising_edges
        if edge_type == "rising"
        else sync_data.get_falling_edges
    )

    line_times = {}
    for line in (line for line in sync_data.line_labels if sync_line_suffix in line):
        camera_name = get_camera_name(line)
        line_times[camera_name] = edge_getter(line, units="seconds")
    return line_times


def get_cam_exposing_times_on_sync(
    sync_path_or_dataset: npc_io.PathLike | npc_sync.SyncDataset,
) -> dict[Literal["behavior", "eye", "face"], npt.NDArray[np.float64]]:
    return get_cam_line_times_on_sync(sync_path_or_dataset, "_cam_exposing")


def get_cam_exposing_falling_edge_times_on_sync(
    sync_path_or_dataset: npc_io.PathLike | npc_sync.SyncDataset,
) -> dict[Literal["behavior", "eye", "face"], npt.NDArray[np.float64]]:
    return get_cam_line_times_on_sync(sync_path_or_dataset, "_cam_exposing", "falling")


def get_cam_transfer_times_on_sync(
    sync_path_or_dataset: npc_io.PathLike | npc_sync.SyncDataset,
) -> dict[Literal["behavior", "eye", "face"], npt.NDArray[np.float64]]:
    return get_cam_line_times_on_sync(sync_path_or_dataset, "_cam_frame_readout")


def get_lost_frames_from_camera_info(
    info_path_or_data: MVRInfoData | npc_io.PathLike,
) -> npt.NDArray[np.int32]:
    """
    >>> get_lost_frames_from_camera_info({'LostFrames': ['1-2,4-5,7']})
    array([0, 1, 3, 4, 6])
    """
    info = get_video_info_data(info_path_or_data)

    if info.get("FramesLostCount") == 0:
        return np.array([])

    assert isinstance(_lost_frames := info["LostFrames"], list)
    lost_frame_spans: list[str] = _lost_frames[0].split(",")

    lost_frames: list[int] = []
    for span in lost_frame_spans:
        start_end = span.split("-")
        if len(start_end) == 1:
            lost_frames.append(int(start_end[0]))
        else:
            lost_frames.extend(np.arange(int(start_end[0]), int(start_end[1]) + 1))

    return np.subtract(lost_frames, 1)  # lost frames in info are 1-indexed


def get_total_frames_from_camera_info(
    info_path_or_data: MVRInfoData | npc_io.PathLike,
) -> int:
    """`FramesRecorded` in info.json plus 1 (for metadata frame)."""
    info = get_video_info_data(info_path_or_data)
    assert isinstance((reported := info.get("FramesRecorded")), int)
    return reported + 1


NumericT = TypeVar("NumericT", bound=np.generic, covariant=True)


def remove_lost_frame_times(
    frame_times: Iterable[NumericT], lost_frame_idx: Container[int]
) -> npt.NDArray[NumericT]:
    """
    >>> remove_lost_frame_times([1., 2., 3., 4., 5.], [1, 3])
    array([1., 3., 5.])
    """
    return np.array(
        [t for idx, t in enumerate(frame_times) if idx not in lost_frame_idx]
    )


def get_video_file_paths(*paths: npc_io.PathLike) -> tuple[upath.UPath, ...]:
    if len(paths) == 1 and npc_io.from_pathlike(paths[0]).is_dir():
        upaths = tuple(npc_io.from_pathlike(paths[0]).iterdir())
    else:
        upaths = tuple(npc_io.from_pathlike(p) for p in paths)
    return tuple(
        p
        for p in upaths
        if p.suffix in (".avi", ".mp4")
        and any(label in p.stem.lower() for label in ("eye", "face", "beh"))
    )


def get_video_info_file_paths(*paths: npc_io.PathLike) -> tuple[upath.UPath, ...]:
    return tuple(
        p.with_suffix(".json").with_stem(p.stem.replace(".mp4", "").replace(".avi", ""))
        for p in get_video_file_paths(*paths)
    )


def get_video_info_data(path_or_info_data: npc_io.PathLike | Mapping) -> MVRInfoData:
    if isinstance(path_or_info_data, Mapping):
        if "RecordingReport" in path_or_info_data:
            return path_or_info_data["RecordingReport"]
        return path_or_info_data
    return json.loads(npc_io.from_pathlike(path_or_info_data).read_bytes())[
        "RecordingReport"
    ]


def get_video_data(
    video_or_video_path: cv2.VideoCapture | npc_io.PathLike,
) -> cv2.VideoCapture:
    """
    >>> path = 's3://aind-ephys-data/ecephys_660023_2023-08-08_07-58-13/behavior_videos/Behavior_20230808T130057.mp4'
    >>> v = get_video_data(path)
    >>> assert isinstance(v, cv2.VideoCapture)
    >>> assert v.get(cv2.CAP_PROP_FRAME_COUNT) != 0
    """
    if isinstance(video_or_video_path, cv2.VideoCapture):
        return video_or_video_path

    video_path = npc_io.from_pathlike(video_or_video_path)
    # check if this is a local or cloud path
    is_local = video_path.protocol in ("file", "")
    if is_local:
        path = video_path.as_posix()
    else:
        path = npc_io.get_presigned_url(video_path)
    return cv2.VideoCapture(path)

def get_barcode_image(
    frame: npt.NDArray[np.uint8], 
    coordinates: dict[Literal["xOffset", "yOffset", "width", "height"], int],
) -> npt.NDArray[np.uint8]:
    """
    Image box contains a series of grey vertical divider lines (1 per exponent; 1-pix wide): 
    the binary value for each exponent is the value to the right of the grey
    line - either black (0) or white (1)
    """
    return frame[
        coordinates["yOffset"] + 1 : coordinates["yOffset"] + coordinates["height"],
        coordinates["xOffset"] : coordinates["xOffset"] + coordinates["width"] + 3, # specification in json seems to be incorrect (perhaps does not include border pixels)
    ]

def get_barcode_value(
    barcode_image: npt.NDArray[np.uint8],
): 
    border = 1 # either side of each "value"
    value_size = 4
    num_values_per_group = 4
    group_size = num_values_per_group * (value_size + border * 2)
    group_separator = 3
    num_groups = 5
    # express values in barcode image as [black, grey, white] -> [-1, 0, 1]:
    values = []
    for group_idx in range(num_groups):
        group_start = group_idx * (group_size + group_separator)
        group_end = group_start + group_size
        group_image = barcode_image[:, group_start: group_end]
        for value_idx in range(num_values_per_group):
            value_start = (value_size + border) * value_idx + (value_idx + 1) * border
            value_end = value_start + value_size
            value_image = group_image[:, value_start : value_end]
            mean_value = np.mean(value_image)
            norm_mean = np.round((mean_value / 255) * 2 - 1) # [black, grey, white] -> [-1, 0, 1]
            values.append(norm_mean)
    exponent_values = tuple(values[::-1])
    """
    plt.subplot(4,1,1)
    plt.imshow(get_barcode_image(frame))
    plt.subplot(4,1,2)
    plt.imshow([get_barcode_image(frame)[0, :, 0] / 255 * 2 - 1])
    plt.subplot(4,1,4)
    plt.imshow(frame[0:10, 0:150, :])
    plt.title(str(values))
    """
    if all(x == 1 for x in exponent_values) and round(np.mean(barcode_image)) > 250:
        # whole barcode area in frame is white: likely metadata frame
        return 0
    value = 0
    for exponent, exponent_value in enumerate(exponent_values):
        if exponent_value == 1:
            value += 2 ** exponent
    return value

def get_barcode_value_from_frame(video_data: cv2.VideoCapture, frame_number: int, barcode_image_coordinates: dict[str, int]) -> int:
    """
    value is the binary value extracted from the barcode in the corner of the
    image
    - there's no barcode on the metadata frame (frame 0)
    - the first proper barcode starts with a value of 1
    """
    video_data.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
    frame: npt.NDArray[np.uint8] = video_data.read()[1] # type: ignore
        
    barcode_image = get_barcode_image(frame, coordinates=barcode_image_coordinates)[:, :, 0]
    value = get_barcode_value(barcode_image)
    if value == 0:
        assert frame_number == 0
    return value

def get_barcode_image_coordinates(video_info: MVRInfoData) -> dict[str, int]:
    default_coordinates = {"xOffset":"0","yOffset":"0","width":"129","height":"3"}
    coordinates: dict[str | Any, int] = {k: int(v) for k, v in video_info.get("BarcodeCoordinates", default_coordinates).items()}
    return coordinates

def get_frame_number_from_barcode(
    video_or_video_path: cv2.VideoCapture | npc_io.PathLike,
    info_path_or_data: MVRInfoData | npc_io.PathLike,
    frame_number: int,
) -> int:
    """
    Extract barcode from encoded ID in image frame.
    
    - barcodes start at 1: presumably to account for metadata frame at 0
    
    >>> path = 's3://aind-private-data-prod-o5171v/ecephys_703333_2024-04-09_13-06-44'
    >>> m = MVRDataset(path)
    >>> video_data = m.video_data['behavior']
    >>> video_info = m.info_data['behavior']
    >>> frame_number = 0
    >>> get_frame_number_from_barcodes(video_data, frame_number=0, video_info=video_info) # metadata frame
    0
    >>> get_frame_number_from_barcodes(video_data, frame_number=1, video_info=video_info)
    1
    """
    video_info = get_video_info_data(info_path_or_data)
    if not video_info.get("FrameID imprint enabled", False) == "true":
        raise ValueError("FrameID imprint not enabled in video")
    video_data = get_video_data(video_or_video_path)
    coordinates = get_barcode_image_coordinates(video_info)
    return get_barcode_value_from_frame(video_data, frame_number, coordinates)

@functools.cache
def get_total_frames_in_video(
    video_path: npc_io.PathLike,
) -> int:
    v = get_video_data(video_path)
    num_frames = v.get(cv2.CAP_PROP_FRAME_COUNT)

    return int(num_frames)

def get_closest_index(target: npt.ArrayLike, values: int) -> int:
    return int(np.nanargmin(np.abs(target - values)))

def get_lick_frames_from_behavior_info(
    info_path_or_data: MVRInfoData | npc_io.PathLike,
):
    if (camera_input := get_video_info_data(info_path_or_data).get("CameraInput", ["1,0"])) == ["1,0"]:
        raise ValueError("Lick frames not recorded in MVR in this session")
    def parse_camera_input(camera_input: str) -> tuple[int, ...]:
        """
        >>> camera_input = ["1,0,105847,1,105849,0,105936,1,105940,0,105945,1,105952,0,105962,1,105966,1,398682,0"]
        >>> parse_camera_input(camera_input)
        (105847, 105849, 105936, 105940, 105945, 105952, 105962, 105966, 398682)
        """
        camera_input: str = camera_input[0]
        return tuple(int(x.strip()) for x in re.findall(r"(\d+)(?=,1,)", camera_input))
    return parse_camera_input(camera_input)

def get_frame(video_data: cv2.VideoCapture, frame_number: int) -> npt.NDArray[np.uint8]:
    video_data.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    return video_data.read()[1] # type: ignore
    
if __name__ == "__main__":
    from npc_mvr import testmod

    testmod()
