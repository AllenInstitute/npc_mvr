from __future__ import annotations

import dataclasses
import datetime
import functools
import json
import logging
import re
from collections.abc import Iterable, Mapping
from typing import Any, Literal, TypeVar

import cv2
import matplotlib.figure
import npc_io
import npc_sync
import numpy as np
import numpy.typing as npt
import upath
from matplotlib import pyplot as plt
from typing_extensions import TypeAlias

logger = logging.getLogger(__name__)


MVRInfoData: TypeAlias = Mapping[str, Any]
"""Contents of `RecordingReport` from a camera's info.json for an MVR
recording."""

CameraName: TypeAlias = Literal["eye", "face", "behavior", "head", "nose"]
CameraNameOnSync: TypeAlias = Literal["eye", "face", "beh", "head", "nose"]

DEFAULT_MAX_FRAMES_CUT: int = 20
"""Cameras with more exposures on sync than MVR accounts for (`FramesRecorded` +
`FramesLostCount`) are excluded by `MVRDataset` if the surplus timestamps that would be
cut from the end of the frame times exceeds this. A surplus means frames were lost without
MVR reporting them (frame-ID barcodes agree with `LostFrames`, so they're missing from both),
and it isn't known where: frame times after them are early, by up to the surplus (see
https://github.com/AllenInstitute/npc_mvr/issues/2).

From a scan of 372 sessions (1,169 cameras, 2022-2026): most cameras need 0-36 frames
padded or cut 0-1; with < 0.1% lost frames the surplus was at most 9 frames in 2022-2024,
but eye cameras in 2025 reach 41 (with ~0.02% lost)."""

DEFAULT_MAX_LOST_FRAME_PERCENTAGE: float = 0.1
"""Cameras with more lost frames than this (% of `FramesRecorded`, from the info json)
are excluded by `MVRDataset`."""


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

    Cameras whose frame times can't be trusted are excluded from `video_paths` (and
    everything derived from it), with a warning: see `excluded_cameras`,
    `max_frames_cut` and `max_lost_frame_percentage`. All videos found are in
    `all_video_paths`.

    >>> import npc_mvr

    >>> d = npc_mvr.MVRDataset('s3://aind-ephys-data/ecephys_670248_2023-08-03_12-04-15')

    # get paths
    >>> d.video_paths['behavior']
    S3Path('s3://aind-ephys-data/ecephys_670248_2023-08-03_12-04-15/behavior_videos/Behavior_20230803T120430.mp4')
    >>> d.info_paths['behavior']
    S3Path('s3://aind-ephys-data/ecephys_670248_2023-08-03_12-04-15/behavior_videos/Behavior_20230803T120430.json')

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
    >>> print(d.frame_times['behavior'])
    [     nan 14.08409 14.10075 ...      nan      nan      nan]
    >>> d.validate()
    """

    def __init__(
        self,
        session_dir: npc_io.PathLike,
        sync_path: npc_io.PathLike | npc_sync.SyncDataset | None = None,
        video_name_filter: str | None = None,
        task_data_or_path: Any = None,
        max_frames_cut: int | None = DEFAULT_MAX_FRAMES_CUT,
        max_lost_frame_percentage: float | None = DEFAULT_MAX_LOST_FRAME_PERCENTAGE,
    ) -> None:
        """
        - `max_frames_cut`: exclude cameras with more surplus exposures on sync than this
          (frames lost without MVR reporting them, so frame times drift). None to disable.
        - `max_lost_frame_percentage`: exclude cameras with more lost frames than this (% of
          `FramesRecorded`). None to disable.
        """
        self.max_frames_cut = max_frames_cut
        self.max_lost_frame_percentage = max_lost_frame_percentage
        self.session_dir = npc_io.from_pathlike(session_dir)
        self._sync_data: npc_sync.SyncDataset | None
        if isinstance(sync_path, npc_sync.SyncDataset):
            self._sync_data = sync_path
        elif sync_path is not None:
            self._sync_data = npc_sync.get_sync_data(sync_path)
        else:
            try:
                sync_path = npc_sync.get_single_sync_path(self.sync_dir)
            except (ValueError, FileNotFoundError):
                self._sync_data = None
            else:
                self._sync_data = npc_sync.get_sync_data(sync_path)
        self._video_name_filter = video_name_filter or ""
        self.task_data_or_path = task_data_or_path  # for behavior box sessions

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
    def is_npexp_format(self) -> bool:
        for name in ("behavior_videos", "behavior-videos", "behavior"):
            if (self.session_dir / name).exists():
                return False
        return True

    @npc_io.cached_property
    def is_behavior_box(self) -> bool:
        if not self.all_video_paths:
            raise NotImplementedError(
                "No video files found in session directory - cannot determine if this is a behavior box session"
            )
        return self.all_video_paths.keys() == {"behavior"} and not self.is_sync

    @npc_io.cached_property
    def is_sync(self) -> bool:
        return self._sync_data is not None

    @npc_io.cached_property
    def sync_dir(self) -> upath.UPath:
        if (path := self.session_dir / "behavior").exists():
            return path
        return self.session_dir

    @npc_io.cached_property
    def video_dir(self) -> upath.UPath:
        if self.is_npexp_format:
            return self.session_dir
        for name in ("behavior_videos", "behavior-videos", "behavior"):
            if (path := self.session_dir / name).exists():
                return path
        return self.session_dir

    @npc_io.cached_property
    def frame_times(self) -> dict[CameraName, npt.NDArray[np.float64]]:
        """Returns frametimes (in seconds) for each camera.

        - metadata frame is at index 0 with a nan value
        - times are according to the sync clock, if sync data is available
        - times are according to the task control clock for behavior box sessions

        - see `get_video_frame_times` for more details
        """
        if self.is_sync:
            return {
                camera_name: self.camera_frame_times[camera_name].corrected_frame_times()
                for camera_name in self.video_paths
            }
        if self.is_behavior_box:
            if "behavior" not in self.video_paths:
                return {}  # excluded: see `excluded_cameras`
            return {
                get_camera_name("behavior"): get_video_frame_times_for_behavior_session(
                    self,
                    self.task_data_or_path,
                )
            }
        else:
            raise AttributeError(
                f"No sync data available for {self.session_dir}. If this is a behavior box session,"
                "use `npc_mvr.get_video_frame_times_for_behavior_session` with a DynRoutData object"
            )

    @npc_io.cached_property
    def all_video_paths(self) -> dict[CameraName, upath.UPath]:
        """All videos found, including cameras in `excluded_cameras`."""
        return {
            get_camera_name(p.stem): p
            for p in get_video_file_paths(self.video_dir)
            if self._video_name_filter in p.stem
        }

    @npc_io.cached_property
    def video_paths(self) -> dict[CameraName, upath.UPath]:
        """Videos whose frame times can be used: `all_video_paths` minus
        `excluded_cameras`."""
        return {
            camera_name: path
            for camera_name, path in self.all_video_paths.items()
            if camera_name not in self.excluded_cameras
        }

    @npc_io.cached_property
    def camera_frame_times(self) -> dict[CameraName, CameraFrameTimes]:
        """Uncorrected frame times and frame counts for all cameras with sync data (see
        `get_camera_frame_times`)."""
        return get_camera_frame_times(self.sync_data, *self.all_video_paths.values())

    @npc_io.cached_property
    def excluded_cameras(self) -> dict[CameraName, str]:
        """Cameras excluded from `video_paths`, with the reason.

        - sync sessions: see `CameraFrameTimes.get_exclusion_reason` (too many frame times
          cut from the end, too many lost frames, or a recording that started where sync
          events are unknown)
        - behavior box sessions: too many lost frames
        """
        excluded: dict[CameraName, str] = {}
        if self.is_sync:
            for camera_name in self.all_video_paths:
                if camera_name not in self.camera_frame_times:
                    excluded[camera_name] = "no exposing line on sync for this camera"
                    continue
                reason = self.camera_frame_times[camera_name].get_exclusion_reason(
                    max_frames_cut=self.max_frames_cut,
                    max_lost_frame_percentage=self.max_lost_frame_percentage,
                )
                if reason is not None:
                    excluded[camera_name] = reason
        elif self.max_lost_frame_percentage is not None:
            for camera_name in self.all_video_paths:
                info = get_video_info_data(
                    get_video_info_file_paths(self.all_video_paths[camera_name])[0]
                )
                lost = 100 * len(get_lost_frames_from_camera_info(info)) / max(
                    info["FramesRecorded"], 1
                )
                if lost > self.max_lost_frame_percentage:
                    excluded[camera_name] = (
                        f"{lost:.3f}% frames lost (limit {self.max_lost_frame_percentage}%)"
                    )
        for camera_name, reason in excluded.items():
            logger.warning(
                f"Excluding {camera_name} camera ({self.all_video_paths[camera_name].as_posix()}): {reason}"
            )
        return excluded

    @npc_io.cached_property
    def video_data(self) -> npc_io.LazyDict[CameraName, cv2.VideoCapture]:
        return npc_io.LazyDict(
            (camera_name, (get_video_data, (path,), {}))
            for camera_name, path in self.all_video_paths.items()
        )

    @npc_io.cached_property
    def info_paths(self) -> dict[CameraName, upath.UPath]:
        return {
            get_camera_name(p.stem): p
            for p in get_video_info_file_paths(self.video_dir)
            if self._video_name_filter in p.stem
        }

    @npc_io.cached_property
    def info_data(self) -> dict[CameraName, MVRInfoData]:
        return {
            camera_name: get_video_info_data(path)
            for camera_name, path in self.info_paths.items()
        }

    @npc_io.cached_property
    def sync_data(self) -> npc_sync.SyncDataset:
        if self._sync_data is not None:
            return self._sync_data
        else:
            raise AttributeError(
                f"No sync data available for {self.session_dir}: this may be a behavior box session"
            )

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
        """Info json contents plus frame counts from sync, for all cameras (including
        those in `excluded_cameras`, whose reason is in `excluded_reason`)."""
        cam_exposing_times = get_cam_exposing_times_on_sync(self.sync_data)
        cam_exposing_falling_edge_times = get_cam_exposing_falling_edge_times_on_sync(
            self.sync_data
        )
        augmented_camera_info = {}
        for camera_name in self.all_video_paths:
            camera_info = dict(self.info_data[camera_name])  # copy
            c = self.camera_frame_times.get(camera_name)
            camera_info["excluded_reason"] = self.excluded_cameras.get(camera_name)
            if c is None:
                augmented_camera_info[camera_name] = camera_info
                continue
            sync_camera_name = get_camera_name(c.sync_line)
            signature_exposures = (
                cam_exposing_falling_edge_times[sync_camera_name][:10]
                - cam_exposing_times[sync_camera_name][:10]
            )
            # exposures in this recording on sync (including those estimated in intervals
            # where sync events are unknown), minus MVR's lost frames, plus the metadata
            # frame. Not from the readout lines: they can miss pulses and count every
            # recording on the line.
            num_expected_from_sync = len(c.frame_times)
            camera_info["num_frames_exposed"] = c.num_exposures
            camera_info["num_frames_exposed_inserted"] = c.num_exposures_inserted
            camera_info["num_frames_in_video"] = c.num_frames_in_video
            camera_info["num_frames_expected_from_sync"] = num_expected_from_sync
            camera_info["expected_minus_actual"] = (
                num_expected_from_sync - c.num_frames_in_video
            )
            camera_info["num_unaccounted_frames"] = c.num_unaccounted_frames
            camera_info["num_frames_from_sync"] = len(c.corrected_frame_times())
            camera_info["signature_exposure_duration"] = np.round(
                np.median(signature_exposures), 3
            )
            camera_info["lost_frame_percentage"] = c.lost_frame_percentage
            camera_info["unknown_sync_intervals"] = c.unknown_intervals
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
                last_frame_barcode_value: int = get_frame_number_from_barcode(
                    video_data, video_info, frame_number=actual_last_frame_index
                )
            except ValueError as exc:
                raise AttributeError(
                    f"Video file {self.video_paths[camera_name]} does not have barcodes in frames"
                ) from exc
            num_lost_frames = last_frame_barcode_value - actual_last_frame_index
            cam_to_frames[camera_name] = int(num_lost_frames)
        return cam_to_frames

    @npc_io.cached_property
    def lick_frames(self) -> npt.NDArray[np.intp]:
        if self.is_sync:
            lick_times = self.sync_data.get_rising_edges("lick_sensor", units="seconds")
            return np.array(
                [
                    get_closest_index(self.frame_times["behavior"], lick_time)
                    for lick_time in lick_times
                ]
            )
        else:
            try:
                return np.array(
                    get_lick_frames_from_behavior_info(self.info_data["behavior"])
                )
            except ValueError as exc:
                raise AttributeError(
                    "Lick frames not recorded in MVR in this session"
                ) from exc

    def plot_synced_frames(self, time: float | None = None) -> matplotlib.figure.Figure:
        check_barcode_matches_frame_number = False
        if time is None:
            if hasattr(self, "lick_frames"):
                time = np.random.choice(self.lick_frames)
            else:
                time = np.random.randint(
                    0, max(len(times) for times in self.frame_times.values())
                )
        fig = plt.figure(figsize=(12, 6))
        ax_idx = 0
        for camera_name in ("face", "behavior"):
            frame_times = self.frame_times[camera_name]
            ax_idx += 1
            closest_frame = get_closest_index(frame_times, time)
            v = self.video_data[camera_name]
            v.set(cv2.CAP_PROP_POS_FRAMES, 0)
            v.set(cv2.CAP_PROP_POS_FRAMES, float(closest_frame))

            if check_barcode_matches_frame_number:
                # because this method of getting the frame via cv2 is known to be
                # unreliable (wrong fame fetched) with variable frame rate videos,
                # verify from the barcode that the frame is correct:
                frame_barcode: int = get_frame_number_from_barcode(v, self.info_data[camera_name], closest_frame)
                if frame_barcode != closest_frame:
                    raise LookupError(
                        f"Frame number from barcode {frame_barcode} does not match expected requested frame number {closest_frame} for {camera_name}"
                    )
            plt.subplot(1, 2, ax_idx)
            frame = v.read()[1]
            if camera_name == "behavior":
                frame = frame[: frame.shape[0] // 2, : frame.shape[1] // 2, :]
            plt.imshow(frame)
            plt.axis("off")
            plt.title(f"{camera_name} @ {time:.2f} s experiment time")

        return fig

    def validate(self) -> None:
        """Check all data required for processing is present and consistent. Check dropped frames
        count."""
        for camera, reason in self.excluded_cameras.items():
            raise AssertionError(f"{camera} camera excluded: {reason}")
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
                num_frames_lost_in_json = len(
                    get_lost_frames_from_camera_info(info_json)
                )
                if num_frames_lost_in_json != (
                    b := self.num_lost_frames_from_barcodes[camera]
                ):
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
    return lost_frame_percentage <= DEFAULT_MAX_LOST_FRAME_PERCENTAGE


def is_acceptable_expected_minus_actual_frame_count(
    expected_minus_actual: int | float,
) -> bool:
    return abs(expected_minus_actual) < 20


def get_camera_name(path: str) -> CameraName:
    """Camera name according to MVR (`behavior`, `eye`, `face`)."""
    names: dict[str, CameraName] = {
        "eye": "eye",
        "face": "face",
        "beh": "behavior",
        "_box_": "behavior",
        "nose": "nose",
    }
    try:
        return names[next(n for n in names if n in str(path).lower())]
    except StopIteration as exc:
        raise ValueError(f"Could not extract camera name from {path}") from exc


def get_camera_name_on_sync(sync_line: str) -> CameraNameOnSync:
    """Camera name as used in sync line labels (`beh`, `eye`, `face`)."""
    name = get_camera_name(sync_line)
    return "beh" if name == "behavior" else name


def get_camera_sync_line_name_mapping(
    sync_path_or_dataset: npc_io.PathLike | npc_sync.SyncDataset,
    *video_paths: npc_io.PathLike,
) -> dict[CameraName, CameraNameOnSync]:
    """Detects if cameras are plugged into sync correctly and returns a mapping
    of camera names to the camera name on sync that actually corresponds, so that this function can
    be used to wrap any access of line data.

    >>> m = MVRDataset('s3://aind-private-data-prod-o5171v/ecephys_703333_2024-04-09_13-06-44')
    >>> get_camera_sync_line_name_mapping(m.sync_data, *m.video_paths.values())
    {'behavior': 'beh', 'face': 'eye', 'eye': 'face'}
    """
    if len(video_paths) == 1:
        raise ValueError("Need to pass all video paths to get camera sync line mapping")
    sync_data = npc_sync.get_sync_data(sync_path_or_dataset)
    jsons = get_video_info_file_paths(*video_paths)
    camera_to_json_data = {
        get_camera_name(path.stem): get_video_info_data(path) for path in jsons
    }
    camera_names_on_sync = [
        name
        for name in ("beh", "face", "eye", "nose")
        if f"{name}_cam_exposing" in sync_data.line_labels
    ]

    def get_exposure_fingerprint_durations_from_jsons() -> dict[str, int]:
        """Nominally expected exposure time in milliseconds for each camera, as
        recorded in info jsons."""
        return {
            f"{camera_name}_cam_exposing": camera_to_json_data[
                get_camera_name(camera_name)
            ]["CustomInitialExposureTime"]
            for camera_name in camera_names_on_sync
            if get_camera_name(camera_name) in camera_to_json_data
        }

    def get_exposure_fingerprint_durations_from_sync() -> dict[str, int]:
        """Initial fingerpring exposure time in milliseconds for each camera, as recorded on sync clock."""
        return {
            (n := f"{camera_name}_cam_exposing"): round(
                (
                    sync_data.get_falling_edges(n, units="seconds")[:8]
                    - sync_data.get_rising_edges(n, units="seconds")[:8]
                ).mean()
                * 1000
            )
            for camera_name in camera_names_on_sync
        }

    def get_start_times_on_sync() -> dict[str, float]:
        return {
            f"{camera_name}{line_suffix}": sync_data.get_rising_edges(
                f"{camera_name}{line_suffix}", units="seconds"
            )[0]
            for camera_name in camera_names_on_sync
            for line_suffix in ("_cam_exposing", "_cam_frame_readout")
            if f"{camera_name}{line_suffix}" in sync_data.line_labels
        }

    start_times_on_sync = get_start_times_on_sync()
    lines_sorted_by_start_time: tuple[float, ...] = tuple(sorted(start_times_on_sync, key=start_times_on_sync.get))  # type: ignore
    expected_exposure_fingerprint_durations = (
        get_exposure_fingerprint_durations_from_jsons()
    )
    actual_exposure_fingerprint_durations = (
        get_exposure_fingerprint_durations_from_sync()
    )
    expected_to_actual_line_mapping: dict[CameraName, CameraNameOnSync] = {}
    for sync_camera_name in camera_names_on_sync:
        exposing_line = f"{sync_camera_name}_cam_exposing"
        if exposing_line not in expected_exposure_fingerprint_durations:
            continue
        expected_duration = expected_exposure_fingerprint_durations[exposing_line]
        actual_line = min(
            actual_exposure_fingerprint_durations,
            key=lambda line: abs(
                expected_duration - actual_exposure_fingerprint_durations[line]
            ),
        )
        expected_to_actual_line_mapping[get_camera_name(sync_camera_name)] = (
            get_camera_name_on_sync(actual_line)
        )
        # readout line is coupled to exposing line (ie. they share the same plug) so the same mapping applies to both
    return expected_to_actual_line_mapping


@dataclasses.dataclass
class CameraFrameTimes:
    """Frame times for one camera's video, before correction to the video's frame count,
    plus the counts needed to judge whether they can be trusted. See
    `get_camera_frame_times`."""

    camera_name: CameraName
    video_path: upath.UPath
    sync_line: str
    """Exposing line on sync that corresponds to the camera (lines may be plugged in
    incorrectly)."""
    frame_times: npt.NDArray[np.float64]
    """One per exposure in the recording on sync, minus the frames MVR reports lost, plus a
    nan for the metadata frame at index 0. Exposures inside intervals where sync events
    are unknown are nan."""
    num_frames_in_video: int
    num_frames_recorded: int
    """`FramesRecorded` in the info json (excludes the metadata frame)."""
    num_lost_frames: int
    """Number of frames listed in `LostFrames` in the info json."""
    num_exposures: int
    """Exposures in the recording on sync, including those inserted in unknown intervals."""
    num_exposures_inserted: int
    """Exposures estimated (from the camera's period) inside intervals where the sync
    line's events are unknown: their frame times are nan."""
    unknown_intervals: tuple[tuple[float, float], ...]
    """Intervals (s) where the sync line's events are unknown (see
    `npc_sync.SyncDataset.get_unknown_intervals`)."""
    is_start_unknown: bool = False
    """The recording may have started inside an unknown interval, so frames can't be
    matched to exposures."""
    ambiguous_intervals: tuple[tuple[float, float], ...] = ()
    """Unknown intervals inside the recording where the number of exposures is ambiguous,
    so frames after them can't be matched to exposures."""

    @property
    def num_frames_cut(self) -> int:
        """Frame times without a frame in the video, which are cut from the end by
        `apply_correction`. Positive means more exposures on sync than MVR accounts for."""
        return len(self.frame_times) - self.num_frames_in_video

    @property
    def num_unaccounted_frames(self) -> int:
        """Exposures on sync that are neither in the video nor listed as lost by MVR."""
        return self.num_exposures - (self.num_frames_recorded + self.num_lost_frames)

    @property
    def lost_frame_percentage(self) -> float:
        return 100 * self.num_lost_frames / max(self.num_frames_recorded, 1)

    def corrected_frame_times(self) -> npt.NDArray[np.float64]:
        """Frame times matched to the video's frame count: nans appended for frames that
        weren't recorded on sync (sync stopped before MVR), or surplus times cut from the
        end."""
        times = self.frame_times
        if (n := self.num_frames_in_video - len(times)) > 0:
            times = np.append(times, np.full(n, np.nan))
        else:
            times = times[: self.num_frames_in_video]
        assert len(times) == self.num_frames_in_video
        return times

    def get_exclusion_reason(
        self,
        max_frames_cut: int | None = DEFAULT_MAX_FRAMES_CUT,
        max_lost_frame_percentage: float | None = DEFAULT_MAX_LOST_FRAME_PERCENTAGE,
    ) -> str | None:
        """Why the frame times shouldn't be used, or None. Limits of None aren't
        checked."""
        if self.is_start_unknown:
            return (
                "recording may have started inside an interval where sync events are "
                f"unknown {self.unknown_intervals}"
            )
        if self.ambiguous_intervals:
            return (
                "number of exposures is ambiguous in intervals where sync events are "
                f"unknown {self.ambiguous_intervals}"
            )
        if max_frames_cut is not None and self.num_frames_cut > max_frames_cut:
            return (
                f"{self.num_frames_cut} frame times would be cut from the end (limit "
                f"{max_frames_cut}): {self.num_unaccounted_frames} exposures on sync are "
                f"neither in the video nor in `LostFrames`, so frame times may drift"
            )
        if (
            max_lost_frame_percentage is not None
            and self.lost_frame_percentage > max_lost_frame_percentage
        ):
            return (
                f"{self.lost_frame_percentage:.3f}% frames lost (limit "
                f"{max_lost_frame_percentage}%)"
            )
        return None

    def summary(self) -> dict[str, Any]:
        return {
            "sync_line": self.sync_line,
            "num_frames_in_video": self.num_frames_in_video,
            "num_exposures": self.num_exposures,
            "num_exposures_inserted": self.num_exposures_inserted,
            "num_lost_frames": self.num_lost_frames,
            "num_unaccounted_frames": self.num_unaccounted_frames,
            "num_frames_cut": self.num_frames_cut,
            "lost_frame_percentage": self.lost_frame_percentage,
            "unknown_intervals": self.unknown_intervals,
            "is_start_unknown": self.is_start_unknown,
            "ambiguous_intervals": self.ambiguous_intervals,
        }


def fill_unknown_intervals(
    edges: npt.ArrayLike,
    unknown_intervals: Iterable[tuple[float, float]],
    period: float | None = None,
    max_gap_periods: float = 3,
    max_fraction: float = npc_sync.patch.PERIODIC_MAX_FRACTION,
) -> tuple[
    npt.NDArray[np.float64], npt.NDArray[np.bool_], tuple[tuple[float, float], ...]
]:
    """Insert edges of a periodic signal in intervals where its events are unknown, so
    edges keep their positions in the sequence. Returns (edges, mask of inserted edges,
    intervals whose count is ambiguous).

    - only intervals with a recorded edge within `max_gap_periods` periods of each side are
      filled (the signal was running throughout); others are left as they are
    - the number inserted is the gap between the edges either side, in periods, minus 1
    - if the gap isn't within `max_fraction` of a whole number of periods (the same gate
      as `npc_sync`'s periodic reconstruction), the count is ambiguous: nothing is inserted
      and the interval is returned in the third element

    A `SyncPatch` already reconstructs camera exposures wherever the count is clear, so with a
    patched `SyncDataset` this only fills intervals that fail its other checks (e.g. rising
    and falling edges disagree); it mainly matters for `on_invalid='mask'`.

    >>> edges, inserted, ambiguous = fill_unknown_intervals([0., 1., 2., 6., 7.], [(2.5, 5.5)])
    >>> edges
    array([0., 1., 2., 3., 4., 5., 6., 7.])
    >>> inserted.nonzero()[0]
    array([3, 4, 5])
    >>> bool(fill_unknown_intervals([0., 1., 2., 106., 107.], [(10., 105.5)])[1].any())  # camera stopped before
    False
    >>> fill_unknown_intervals([0., 1., 2., 6.5, 7.5], [(2.5, 6.)])[2]  # 4.5 periods
    ((2.5, 6.0),)
    """
    edges = np.asarray(edges, dtype=np.float64)
    inserted = np.zeros(edges.shape, dtype=bool)
    ambiguous: list[tuple[float, float]] = []
    if edges.size < 2:
        return edges, inserted, ()
    if period is None:
        period = float(np.median(np.diff(edges)))
    for start, stop in sorted(unknown_intervals, reverse=True):  # insert from the end
        idx = int(np.searchsorted(edges, start))
        if idx == 0 or idx == edges.size:
            continue
        before, after = edges[idx - 1], edges[idx]
        if after <= stop and after >= start:
            continue  # edges inside: not unknown for this signal
        if start - before > max_gap_periods * period or after - stop > max_gap_periods * period:
            continue
        n_periods = (after - before) / period
        n = round(n_periods) - 1
        if abs(n_periods - round(n_periods)) > max_fraction:
            logger.warning(
                f"Gap of {after - before:.4f} s around unknown interval {start}-{stop} s is "
                f"{n_periods:.2f} periods: the number of edges inside is ambiguous"
            )
            ambiguous.append((float(start), float(stop)))
            continue
        if n < 1:
            continue
        new = np.linspace(before, after, n + 2)[1:-1]
        edges = np.insert(edges, idx, new)
        inserted = np.insert(inserted, idx, np.ones(n, dtype=bool))
    return edges, inserted, tuple(sorted(ambiguous))


def get_camera_frame_times(
    sync_path_or_dataset: npc_io.PathLike | npc_sync.SyncDataset,
    *video_paths: npc_io.PathLike,
) -> dict[CameraName, CameraFrameTimes]:
    """Frame times on the sync clock for each video, before correction to the video's frame
    count, with the counts used to check them. See `get_video_frame_times`.

    - pass all of a session's videos: they're needed to work out which sync line
      corresponds to each camera
    - if the sync data has intervals where a camera's exposing events are unknown
      (`npc_sync.SyncDataset.get_unknown_intervals`, e.g. replayed DAQ buffers that couldn't
      be reconstructed by a `SyncPatch`), the missing exposures are estimated from the
      camera's period and their frame times are nan, so frames after them keep the right
      times
    """
    sync_data = npc_sync.get_sync_data(sync_path_or_dataset)
    videos = get_video_file_paths(*video_paths)
    jsons = get_video_info_file_paths(*video_paths)
    camera_to_video_path = {get_camera_name(path.stem): path for path in videos}
    camera_to_json_data = {
        get_camera_name(path.stem): get_video_info_data(path) for path in jsons
    }
    correct_sync_line_name_mapping = get_camera_sync_line_name_mapping(
        sync_data, *videos
    )
    if tuple(
        get_camera_name_on_sync(c) for c in correct_sync_line_name_mapping
    ) != tuple(correct_sync_line_name_mapping.values()):
        logger.warning(
            f"Camera lines are plugged into sync incorrectly - we'll accommodate for this, but if this is a recent session check the rig: {correct_sync_line_name_mapping}"
        )
    results: dict[CameraName, CameraFrameTimes] = {}
    for camera, sync_camera in correct_sync_line_name_mapping.items():
        if camera not in camera_to_video_path:
            continue
        line = f"{sync_camera}_cam_exposing"
        video_path = camera_to_video_path[camera]
        info = camera_to_json_data[camera]
        exposing_times = sync_data.get_rising_edges(line, units="seconds")
        unknown_intervals = tuple(
            (float(a), float(b)) for a, b in sync_data.get_unknown_intervals(line)
        )
        exposing_times, is_inserted, ambiguous_intervals = fill_unknown_intervals(
            exposing_times, unknown_intervals
        )
        if is_inserted.any():
            logger.warning(
                f"Sync events on {line} are unknown in {unknown_intervals}: inserted "
                f"{is_inserted.sum()} exposures with nan frame times for {video_path.as_posix()}"
            )
        num_frames_in_video = get_total_frames_in_video(video_path)
        json_start_time = datetime.datetime.fromisoformat(info["TimeStart"].strip("Z"))
        sync_start_time = sync_data.start_time
        # check that exposing time in results is close to video start time in metadata:
        assert (
            sync_start_time < json_start_time
        ), f"Video start time from json info {json_start_time} is before sync start time {sync_start_time} for {camera}: cannot align frames if first exposure not captured on sync"

        estimated_start_time_on_sync = (json_start_time - sync_start_time).seconds
        # if sync + MVR started long before experiment (ie. pretest that wasn't
        # stopped) sync will have extra exposing times at start that we need to ignore.
        # outlier long exposing intervals help us identify MVR recordings being
        # stopped then restarted for the experiment. We expect one, but could be
        # multiple. Split on long intervals and get the block that starts closest to the
        # video's start time.
        # unknown intervals are filled above, so they aren't mistaken for breaks between
        # recordings
        intervals = np.diff(exposing_times)
        interval_z = np.abs(intervals - np.median(intervals)) / np.std(intervals)
        long_interval_idx = np.where(interval_z > 50)[0]
        # breaks between recordings typically 100s or 1000s of seconds, but for
        # 644867_2023-02-23 it's only 0.36 s (zscore > 500)
        if long_interval_idx.size:
            exposing_time_blocks = np.split(exposing_times, long_interval_idx + 1)
            inserted_blocks = np.split(is_inserted, long_interval_idx + 1)
            logger.warning(
                f"Long exposure times detected for {video_path.as_posix()}, suggesting multiple videos captured on sync: {len(exposing_time_blocks)=}"
            )
            assert np.all(
                np.diff([e[0] for e in exposing_time_blocks]) > 4 * np.median(intervals)
            ), f"Exposing times not split correctly for {camera}"
            block_idx = min(
                range(len(exposing_time_blocks)),
                key=lambda i: abs(
                    exposing_time_blocks[i][0] - estimated_start_time_on_sync
                ),
            )
            exposing_times = exposing_time_blocks[block_idx]
            is_inserted = inserted_blocks[block_idx]
        _threshold = 10  # the allowable difference in seconds between the system time on sync computer and the system time on the vidmon computer
        assert (
            abs(estimated_start_time_on_sync - exposing_times[0]) < _threshold
        ), f"First exposing time {exposing_times[0]} s isn't close to estimated video start time {estimated_start_time_on_sync} s: check method for dividing exposing times into blocks"

        # the recording's first exposures may be lost if it started in an unknown interval
        # (not filled, as the camera wasn't running before it):
        period = float(np.median(np.diff(exposing_times)))
        is_start_unknown = any(
            start <= exposing_times[0] <= stop + 3 * period
            for start, stop in unknown_intervals
        )
        if is_start_unknown:
            logger.warning(
                f"Recording for {video_path.as_posix()} may have started inside an "
                f"interval where events on {line} are unknown {unknown_intervals}: frames "
                "can't be matched to exposures"
            )
        # an ambiguous count inside the recording shifts every later frame by an unknown amount
        ambiguous_intervals = tuple(
            (a, b)
            for a, b in ambiguous_intervals
            if exposing_times[0] <= a and b <= exposing_times[-1]
        )
        lost_frames = get_lost_frames_from_camera_info(info)
        keep = np.ones(exposing_times.size, dtype=bool)
        keep[lost_frames[lost_frames < exposing_times.size].astype(int)] = False
        camera_frame_times = np.where(is_inserted, np.nan, exposing_times)[keep]
        # Insert a nan frame time at the beginning to account for metadata frame
        camera_frame_times = np.insert(camera_frame_times, 0, np.nan)
        results[camera] = CameraFrameTimes(
            camera_name=camera,
            video_path=video_path,
            sync_line=line,
            frame_times=camera_frame_times,
            num_frames_in_video=num_frames_in_video,
            num_frames_recorded=int(info["FramesRecorded"]),
            num_lost_frames=len(lost_frames),
            num_exposures=int(exposing_times.size),
            num_exposures_inserted=int(is_inserted.sum()),
            unknown_intervals=unknown_intervals,
            is_start_unknown=is_start_unknown,
            ambiguous_intervals=ambiguous_intervals,
        )
    return results


def get_video_frame_times(
    sync_path_or_dataset: npc_io.PathLike | npc_sync.SyncDataset,
    *video_paths: npc_io.PathLike,
    apply_correction: bool = True,
    max_frames_cut: int | None = None,
) -> dict[upath.UPath, npt.NDArray[np.float64]]:
    """Returns frametimes as measured on sync clock for each video file.

    If a single directory is passed, video files in that directory will be
    found. If multiple paths are passed, the video files will be filtered out.

    - keys are video file paths
    - values are arrays of frame times in seconds
    - the first frametime will be a nan value (corresponding to a metadata frame)
    - frames whose exposures fall in intervals where sync events are unknown are nan (see
      `get_camera_frame_times`)
    - frames at the end may also be nan values:
        MVR previously ceased all TTL pulses before the recording was
        stopped, resulting in frames in the video that weren't registered
        in sync. MVR was fixed July 2023 after Corbett discovered the issue.

        (only applied if `apply_correction` is True)

    - frametimes from sync may be cut to match the number of frames in the video:
        after July 2023, we started seeing video files that had fewer frames than
        timestamps in sync file.

        (only applied if `apply_correction` is True)

        Many surplus timestamps mean frames were lost without MVR reporting them, so the
        frame times drift. If `max_frames_cut` is not None, a ValueError is raised when
        more than this would be cut. `MVRDataset` excludes such cameras instead (see
        `DEFAULT_MAX_FRAMES_CUT`).

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
    frame_times: dict[upath.UPath, npt.NDArray[np.float64]] = {}
    for camera, c in get_camera_frame_times(sync_path_or_dataset, *video_paths).items():
        if not apply_correction:
            frame_times[c.video_path] = c.frame_times
            continue
        if max_frames_cut is not None and c.num_frames_cut > max_frames_cut:
            raise ValueError(
                f"{c.video_path}: {c.get_exclusion_reason(max_frames_cut, None)}"
            )
        frame_times[c.video_path] = c.corrected_frame_times()
    return frame_times


def get_cam_line_times_on_sync(
    sync_path_or_dataset: npc_io.PathLike | npc_sync.SyncDataset,
    sync_line_suffix: str,
    edge_type: Literal["rising", "falling"] = "rising",
) -> dict[CameraName, npt.NDArray[np.float64]]:
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
) -> dict[CameraName, npt.NDArray[np.float64]]:
    return get_cam_line_times_on_sync(sync_path_or_dataset, "_cam_exposing")


def get_cam_exposing_falling_edge_times_on_sync(
    sync_path_or_dataset: npc_io.PathLike | npc_sync.SyncDataset,
) -> dict[CameraName, npt.NDArray[np.float64]]:
    return get_cam_line_times_on_sync(sync_path_or_dataset, "_cam_exposing", "falling")


def get_cam_transfer_times_on_sync(
    sync_path_or_dataset: npc_io.PathLike | npc_sync.SyncDataset,
) -> dict[CameraName, npt.NDArray[np.float64]]:
    return get_cam_line_times_on_sync(sync_path_or_dataset, "_cam_frame_readout")


def get_lost_frames_from_camera_info(
    info_path_or_data: MVRInfoData | npc_io.PathLike,
) -> npt.NDArray[np.int32]:
    """
    LostFrames are 0-indexed (including metadata frame if present), so a value of 4 means the 5th
    frame in the actual video.

    >>> get_lost_frames_from_camera_info({'LostFrames': ['1-2,4-5,7']})
    array([1, 2, 4, 5, 7])
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

    return np.array(
        lost_frames
    )  # lost frames in info are 0-indexed (where frame 0 may be metadata frame)


def get_total_frames_from_camera_info(
    info_path_or_data: MVRInfoData | npc_io.PathLike,
) -> int:
    """`FramesRecorded` in info.json plus 1 (for metadata frame)."""
    info = get_video_info_data(info_path_or_data)
    assert isinstance((reported := info.get("FramesRecorded")), int)
    return reported + 1


NumericT = TypeVar("NumericT", bound=np.generic, covariant=True)


def remove_lost_frame_idx(
    frame_times: Iterable[NumericT], lost_frame_idx: Iterable[int]
) -> npt.NDArray[NumericT]:
    """
    >>> remove_lost_frame_idx([1., 2., 3., 4., 5.], [1, 3])
    array([1., 3., 5.])
    """
    times = np.asarray(list(frame_times))
    idx = np.asarray(list(lost_frame_idx), dtype=int)
    keep = np.ones(times.shape[0], dtype=bool)
    keep[idx[(idx >= 0) & (idx < times.shape[0])]] = False
    return times[keep]


def get_video_file_paths(*paths: npc_io.PathLike) -> tuple[upath.UPath, ...]:
    if len(paths) == 1 and npc_io.from_pathlike(paths[0]).is_dir():
        upaths = tuple(npc_io.from_pathlike(paths[0]).iterdir())
    else:
        upaths = tuple(npc_io.from_pathlike(p) for p in paths)
    return tuple(
        p
        for p in upaths
        if p.suffix in (".avi", ".mp4")
        and any(
            label in p.stem.lower() for label in ("eye", "face", "beh", "_box_", "nose")
        )
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
        coordinates["xOffset"] : coordinates["xOffset"]
        + coordinates["width"]
        + 3,  # specification in json seems to be incorrect (perhaps does not include border pixels)
    ]


def get_barcode_value(
    barcode_image: npt.NDArray[np.uint8],
):
    border = 1  # either side of each "value"
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
        group_image = barcode_image[:, group_start:group_end]
        for value_idx in range(num_values_per_group):
            value_start = (value_size + border) * value_idx + (value_idx + 1) * border
            value_end = value_start + value_size
            value_image = group_image[:, value_start:value_end]
            mean_value = np.mean(value_image)
            norm_mean = np.round(
                (mean_value / 255) * 2 - 1
            )  # [black, grey, white] -> [-1, 0, 1]
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
            value += 2**exponent
    return value


def get_barcode_value_from_frame(
    video_data: cv2.VideoCapture,
    frame_number: int,
    barcode_image_coordinates: dict[
        Literal["xOffset", "yOffset", "width", "height"], int
    ],
) -> int:
    """
    value is the binary value extracted from the barcode in the corner of the
    image
    - there's no barcode on the metadata frame (frame 0)
    - the first proper barcode starts with a value of 1
    """
    video_data.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
    frame: npt.NDArray[np.uint8] = video_data.read()[1]  # type: ignore

    barcode_image = get_barcode_image(frame, coordinates=barcode_image_coordinates)[
        :, :, 0
    ]
    value = get_barcode_value(barcode_image)
    if value == 0:
        assert frame_number == 0
    return value


def get_barcode_image_coordinates(
    video_info: MVRInfoData,
) -> dict[Literal["xOffset", "yOffset", "width", "height"], int]:
    default_coordinates = {
        "xOffset": "0",
        "yOffset": "0",
        "width": "129",
        "height": "3",
    }
    coordinates: dict[Literal["xOffset", "yOffset", "width", "height"], int] = {
        k: int(v)
        for k, v in video_info.get("BarcodeCoordinates", default_coordinates).items()
    }
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
    >>> get_frame_number_from_barcode(video_data, info_path_or_data=video_info, frame_number=0) # metadata frame
    0
    >>> get_frame_number_from_barcode(video_data, info_path_or_data=video_info, frame_number=1)
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


def get_closest_index(arr: npt.ArrayLike, value: int | float) -> int:
    return int(np.nanargmin(np.abs(arr - value)))  # type: ignore


def get_lick_frames_from_behavior_info(
    info_path_or_data: MVRInfoData | npc_io.PathLike,
) -> tuple[int, ...]:
    def parse_camera_input(camera_input: str) -> tuple[int, ...]:
        """
        CameraInput values are 0-indexed (including metadata frame if present), so a value of 105847 means the 105848th
        frame in the actual video.

        >>> camera_input = ["1,0,105847,1,105849,0,105936,1,105940,0,105945,1,105952,0,105962,1,105966,1,398682,0"]
        >>> parse_camera_input(camera_input)
        (105847, 105849, 105936, 105940, 105945, 105952, 105962, 105966, 398682)
        """
        return tuple(int(x.strip()) for x in re.findall(r"(\d+)(?=,1,)", camera_input))

    if (
        camera_input := get_video_info_data(info_path_or_data).get(
            "CameraInput", ["1,0"]
        )[0]
    ) == "1,0":
        raise ValueError("Lick frames not recorded in MVR in this session")
    return parse_camera_input(camera_input)


def get_frame(video_data: cv2.VideoCapture, frame_number: int) -> npt.NDArray[np.uint8]:
    video_data.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    return video_data.read()[1]  # type: ignore


def get_frametimes_from_behavior_session_original(mvr_dataset, task_data_or_path):
    try:
        import npc_samstim
    except ImportError:
        raise ImportError("`npc_samstim` required to get frametimes for behavior boxes")
    import npc_stim  # installed with npc_samstim

    behavData = npc_samstim.get_sam(task_data_or_path)
    # override filtered lick times
    behavData.lickFrames = npc_stim.get_stim_data(task_data_or_path)["lickFrames"][:]
    behavData.lickTimes = behavData.frameTimes[behavData.lickFrames]

    videoData = mvr_dataset.info_data["behavior"]
    lostFrames = videoData["LostFrames"]
    assert (
        len(lostFrames) < 1
    )  # haven't found any videos with lost frames, so not dealing with them yet

    numVideoFrames = videoData["FramesRecorded"]
    videoFrameRate = videoData["FPS"]

    cameraInput = np.array([int(v) for v in videoData["CameraInput"][0].split(",")])
    videoLickFrames = cameraInput[0::2][cameraInput[1::2].astype(bool)]
    assert videoLickFrames.size == videoData["CameraInputCount"]

    # remove video licks before or after behavior session
    if videoLickFrames.size > behavData.lickFrames.size:
        videoLickIntervals = np.diff(videoLickFrames) * 2
        behavLickIntervals = np.diff(behavData.lickFrames)
        c = np.correlate(videoLickIntervals, behavLickIntervals)
        peak = np.argmax(c)
        if peak > videoLickIntervals.size:
            i = peak - c.size
            videoLickFrames = videoLickFrames[i - behavData.lickFrames.size : i]
        else:
            i = peak
            videoLickFrames = videoLickFrames[i : i + behavData.lickFrames.size]
    assert videoLickFrames.size == behavData.lickFrames.size

    # get video frame times aligned to behavior session
    videoFrameTimes = []  # list of arrays of frame times between licks

    # get frame times to first lick using time of first lick and video frame rate
    videoStartTime = (
        behavData.lickTimes[0] - videoLickFrames[0] / videoFrameRate
    )  # relative to start of behavior session
    videoFrameTimes.append(
        np.linspace(videoStartTime, behavData.lickTimes[0], videoLickFrames[0])[:-1]
    )

    # get frame times between licks using frame interval implied by interval between licks and number of video frames
    for i in range(videoLickFrames.size - 1):
        nFrames = videoLickFrames[i + 1] - videoLickFrames[i]
        videoFrameTimes.append(
            np.linspace(
                behavData.lickTimes[i], behavData.lickTimes[i + 1], nFrames + 1
            )[:-1]
        )

    # get frame times after last lick
    nFrames = numVideoFrames - videoLickFrames[-1]
    videoStopTime = behavData.lickTimes[-1] + nFrames / videoFrameRate
    videoFrameTimes.append(
        np.linspace(behavData.lickTimes[-1], videoStopTime, nFrames + 1)
    )

    # concatenate all the frame times
    videoFrameTimes = np.concatenate(videoFrameTimes)
    assert videoFrameTimes.size == numVideoFrames
    return videoFrameTimes


def get_video_frame_times_for_behavior_session(
    mvr_dataset,
    task_data_or_path,
):
    """Returns predicted vis stim frametimes, for use with behavior file.

    from https://github.com/samgale/DynamicRoutingTask/blob/dbf6df70fc0edc8aae51eceb409e0bb6c84cc981/Analysis/alignBehavVideo.py
    """
    try:
        import npc_samstim
    except ImportError:
        raise ImportError("`npc_samstim` required to get frametimes for behavior boxes")
    import npc_stim  # installed with npc_samstim

    behavData = npc_samstim.get_sam(task_data_or_path)
    # override filtered lick times
    behavData.lickFrames = npc_stim.get_stim_data(task_data_or_path)["lickFrames"][:]
    behavData.lickTimes = behavData.frameTimes[behavData.lickFrames]

    videoPath = mvr_dataset.video_paths["behavior"]
    videoInfo = mvr_dataset.info_data["behavior"]
    videoData = mvr_dataset.video_data["behavior"]

    numVideoFrames = get_total_frames_in_video(videoData)
    assert numVideoFrames == videoInfo["FramesRecorded"] + 1  # +1 for metadata frame
    lostFrames = get_lost_frames_from_camera_info(videoInfo)
    videoFrameRate = videoInfo["FPS"]
    videoLickFrames = np.array(get_lick_frames_from_behavior_info(videoInfo))
    assert videoLickFrames.size == videoInfo["CameraInputCount"]
    # lost and lick frames from camera json are 0-indexed. If metadata frame is present, it is frame 0

    # remove video licks before or after behavior session
    if videoLickFrames.size > behavData.lickFrames.size:
        videoLickIntervals = np.diff(videoLickFrames) * (
            behavData.frameRate / videoFrameRate
        )
        behavLickIntervals = np.diff(behavData.lickFrames)
        c = np.correlate(videoLickIntervals, behavLickIntervals)
        peak = np.argmax(c)
        assert peak <= videoLickIntervals.size
        if peak > videoLickIntervals.size:  # shouldn't be possible?
            videoLickFrames = videoLickFrames[peak - behavData.lickFrames.size : peak]
        else:
            videoLickFrames = videoLickFrames[peak : peak + behavData.lickFrames.size]
    # elif videoLickFrames.size < behavData.lickFrames.size:
    #     c = np.correlate(behavLickIntervals, videoLickIntervals)
    #     peak = np.argmax(c)
    #     nans = np.full(behavData.lickFrames.size - videoLickFrames.size, np.nan)
    #     if peak == 0:
    #         videoLickFrames = np.concatenate([videoLickFrames, nans])
    #     else:
    #         videoLickFrames = np.concatenate([nans, videoLickFrames])

    assert videoLickFrames.size <= behavData.lickFrames.size == behavData.lickTimes.size
    if videoLickFrames.size > behavData.lickFrames.size:
        raise NotImplementedError(
            f"Video licks {videoLickFrames.size} > behavior licks {behavData.lickFrames.size} - cannot handle this currently"
        )
    # get video frame times aligned to behavior session
    videoFrameTimes = []  # list of arrays of frame times between licks
    videoFrameTimes.append(
        np.array([np.nan])
    )  # nan first frame time for metadata frame

    def get_lost_frames_between(start_frame_idx, end_frame_idx):
        return lostFrames[
            (lostFrames >= start_frame_idx) & (lostFrames < end_frame_idx)
        ]

    def append_frame_times(start_frame_idx, end_frame_idx, start_time, end_time):
        assert end_frame_idx % 1 == start_frame_idx % 1 == 0
        nFrames = int(end_frame_idx - start_frame_idx)
        lost_frames_between_licks = get_lost_frames_between(
            start_frame_idx, end_frame_idx
        )
        n_lost_frames = len(lost_frames_between_licks)
        if n_lost_frames:
            print(f"Adjusting for lost frames: {lost_frames_between_licks}")
        frame_idx = np.linspace(
            start_frame_idx, end_frame_idx, nFrames + n_lost_frames, endpoint=False
        ).astype(int)
        frame_times = np.linspace(
            start_time, end_time, nFrames + n_lost_frames, endpoint=False
        )  # (..., nFrames+1)[:-1] or endpoint=false
        assert frame_times.size == frame_idx.size
        assert np.all(np.diff(frame_times) > 0)
        if np.abs(
            (frame_times[-1] - frame_times[0]) / len(frame_times) - (1 / videoFrameRate)
        ) > (2 / videoFrameRate):
            raise AssertionError(
                f"Frame interval is not close to 1 / videoFrameRate: {frame_times[-1] - frame_times[0]} / {len(frame_times)} != 1 / {videoFrameRate}"
            )
        videoFrameTimes.append(
            frame_times[
                np.in1d(
                    frame_idx,
                    lost_frames_between_licks,
                    assume_unique=True,
                    invert=True,
                )
            ]
        )

    # get frame times to first lick using time of first lick and video frame rate
    videoStartTime = (
        behavData.lickTimes[0] - videoLickFrames[0] / videoFrameRate
    )  # relative to start of behavior session
    append_frame_times(
        start_frame_idx=1,
        end_frame_idx=videoLickFrames[0],
        start_time=videoStartTime,
        end_time=behavData.lickTimes[0],
    )

    # get frame times between licks using frame interval implied by interval between licks and
    # number of video frames
    i_vid = -1
    i_behav = -1
    for i in range(videoLickFrames.size - 1):
        i_vid += 1
        i_behav += 1
        assert i_vid == i_behav
        # TODO: handle mismatches in lick intervals (at some point, a lick is missing in the video frames)
        # Example case: 750092_Box_2_20241008T103718.avi
        start_frame_idx = videoLickFrames[i_vid]
        end_frame_idx = videoLickFrames[i_vid + 1]
        start_time = behavData.lickTimes[i_behav]
        end_time = behavData.lickTimes[i_behav + 1]
        append_frame_times(
            start_frame_idx=start_frame_idx,
            end_frame_idx=end_frame_idx,
            start_time=start_time,
            end_time=end_time,
        )

    # get frame times after last lick
    nFrames = numVideoFrames - videoLickFrames[-1]
    videoStopTime = behavData.lickTimes[-1] + nFrames / videoFrameRate
    append_frame_times(
        start_frame_idx=videoLickFrames[-1],
        end_frame_idx=numVideoFrames,
        start_time=behavData.lickTimes[-1],
        end_time=videoStopTime,
    )

    # concatenate all the frame times
    videoFrameTimes = np.concatenate(videoFrameTimes)
    assert videoFrameTimes.size == numVideoFrames
    return videoFrameTimes


def validate_stim_frame_times_for_behavior_session(
    videoFrameTimes,
    mvr_dataset,
    task_data_or_path,
):
    """Returns diff between predicted vis stim onset frame (from videoFrameTimes) and actual
    onset frame (from sam_obj.stimStartTimes).

    from https://github.com/samgale/DynamicRoutingTask/blob/dbf6df70fc0edc8aae51eceb409e0bb6c84cc981/Analysis/alignBehavVideo.py
    """
    try:
        import npc_samstim
    except ImportError:
        raise ImportError("`npc_samstim` required to get frametimes for behavior boxes")
    import npc_stim  # installed with npc_samstim

    task_data = npc_stim.get_stim_data(task_data_or_path)
    behavData = npc_samstim.get_sam(task_data_or_path)

    # find video frames corresponding to visual stimulus onset times in behavior file
    visOnsetTimes = behavData.stimStartTimes[
        np.in1d(behavData.trialStim, ("vis1", "vis2"))
    ]
    predictedVisOnsetFrames = np.searchsorted(
        videoFrameTimes, visOnsetTimes
    )  # first video frame after vis stim onset

    # find visual stimulus onset frames in video by thresholding roi over stimulus location
    videoIn = get_video_data(mvr_dataset.video_paths["behavior"])

    stimRoiIntensity = []
    # don't skip first frame (metadata)
    while True:
        isFrame, videoFrame = videoIn.read()
        if isFrame:
            videoFrame = cv2.cvtColor(videoFrame, cv2.COLOR_BGR2GRAY)
            stimRoiIntensity.append(videoFrame[:60, :30].mean())
        else:
            break
    videoIn.release()

    stimRoiIntensity = np.array(stimRoiIntensity)
    assert (
        stimRoiIntensity.size == mvr_dataset.info_data["behavior"]["FramesRecorded"] + 1
    )  # +1 for metadata frame

    # find roi intensity changes
    m = np.median(stimRoiIntensity)
    thresh = 0.05 * m
    aboveThresh = (stimRoiIntensity < m - thresh) | (stimRoiIntensity > m + thresh)
    threshFrames = np.where(aboveThresh)[0]

    # find onsets; remove first and last (start and end of session)
    d = np.concatenate(([0], np.diff(threshFrames)))
    onsetFrames = threshFrames[d > 30][:-1]

    # remove timeouts
    visOnsetFrames = np.array(
        [i for i in onsetFrames if aboveThresh[i : i + 20].sum() < 19]
    )

    # remove onset of non-completed trial if present
    if not hasattr(behavData, "endsWithNonCompletedTrial"):
        behavData.endsWithNonCompletedTrial = (
            task_data["trialStartFrame"].size > task_data["trialEndFrame"].size
        )
    if (
        behavData.endsWithNonCompletedTrial
        and (visOnsetFrames.size - predictedVisOnsetFrames.size) == 1
    ):
        visOnsetFrames = visOnsetFrames[:-1]

    assert visOnsetFrames.size == predictedVisOnsetFrames.size > 0

    # get difference between predicted and actual vis stim onset frames
    return np.subtract(predictedVisOnsetFrames, visOnsetFrames)


if __name__ == "__main__":
    from npc_mvr import testmod

    testmod()
