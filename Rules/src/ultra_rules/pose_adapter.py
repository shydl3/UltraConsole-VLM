"""Video semantic analysis pose adapter with an independent tracker per file."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from ultralytics import YOLO

from .contracts import Keypoint, PersonObservation


@dataclass
class PoseAdapter:
    model_path: str
    device: str = "cpu"
    conf: float = 0.25
    iou: float = 0.5
    tracker_yaml: str = "bytetrack.yaml"
    class_name: str = "person"
    # PT0-1：>1 时走批量前向 + 按帧序 BYTETracker 关联（infer_batch）；
    # =1 保持原有单帧 model.track 路径，语义与缓存行为不变。
    batch_size: int = 1
    _model: Any = None
    _model_hash: str | None = None
    _batch_tracker: Any = None

    # ------------------------------------------------------------------
    def load(self) -> None:
        if self._model is None:
            self._model = YOLO(self.model_path)
            self._model_hash = _sha1(self.model_path)

    @property
    def model_hash(self) -> str | None:
        if self._model_hash is None:
            self._model_hash = _sha1(self.model_path)
        return self._model_hash

    # ------------------------------------------------------------------
    def infer(self, image_bgr: np.ndarray, *, track: bool = True) -> list[PersonObservation]:
        """单帧推理。image_bgr: HxWx3 uint8（OpenCV BGR）。"""
        self.load()
        kwargs: dict[str, Any] = dict(
            verbose=False,
            device=self.device,
            conf=self.conf,
            iou=self.iou,
        )
        if track:
            results = self._model.track(image_bgr, persist=True, tracker=self.tracker_yaml, **kwargs)
        else:
            results = self._model.predict(image_bgr, **kwargs)
        if not results:
            return []
        return self._persons_from_result(results[0], track=track)

    # ------------------------------------------------------------------
    def infer_batch(self, images_bgr: list[np.ndarray]) -> list[list[PersonObservation]]:
        """批量前向 + 按帧序关联（PT0-1）。输入须按时间序排列。

        检测用一次 batch predict；关联逐帧喂给独立 BYTETracker，精确复刻
        ultralytics `on_predict_start / on_predict_postprocess_end`（persist=True、
        非 stream 模式全局单 tracker、frame_rate=30、永不 reset），跟踪语义与
        单帧 model.track 路径一致；仅前向改为批量。调用方不得混用 infer(track=True)
        与本方法（两套 tracker 状态独立）。
        """
        if not images_bgr:
            return []
        import torch

        self.load()
        results = self._model.predict(images_bgr, verbose=False, device=self.device,
                                      conf=self.conf, iou=self.iou)
        tracker = self._bytetrack()
        out: list[list[PersonObservation]] = []
        for r0 in results:
            det = r0.boxes.cpu().numpy()
            tracks = tracker.update(det, r0.orig_img)
            if len(tracks):
                idx = tracks[:, -1].astype(int)
                r0 = r0[idx]
                r0.update(boxes=torch.as_tensor(tracks[:, :-1]))
            out.append(self._persons_from_result(r0, track=True))
        return out

    def _bytetrack(self):
        if self._batch_tracker is None:
            from ultralytics.trackers.byte_tracker import BYTETracker
            from ultralytics.utils import YAML, IterableSimpleNamespace
            from ultralytics.utils.checks import check_yaml

            cfg = IterableSimpleNamespace(**YAML.load(check_yaml(self.tracker_yaml)))
            if getattr(cfg, "tracker_type", "bytetrack") != "bytetrack":
                raise ValueError("Batch pose inference only supports bytetrack")
            self._batch_tracker = BYTETracker(args=cfg, frame_rate=30)
        return self._batch_tracker

    def _persons_from_result(self, r0, *, track: bool) -> list[PersonObservation]:
        persons: list[PersonObservation] = []
        boxes = getattr(r0, "boxes", None)
        if boxes is None:
            return persons
        xyxy = _np(boxes.xyxy)
        confs = _np(boxes.conf)
        clss = _np(boxes.cls)
        ids = _np(getattr(boxes, "id", None)) if track else None
        kps_obj = getattr(r0, "keypoints", None)
        kp_xy = _np(kps_obj.xy) if kps_obj is not None else None
        kp_conf = _np(getattr(kps_obj, "conf", None)) if kps_obj is not None else None

        names = getattr(self._model, "names", {}) or {}
        for i in range(len(xyxy)):
            cls_id = int(clss[i])
            cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
            if cls_name != self.class_name:
                continue
            x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
            tid: int | None = None
            if ids is not None and i < len(ids):
                try:
                    tid = int(ids[i])
                except Exception:
                    tid = None
            keypoints: list[Keypoint] = []
            if kp_xy is not None and i < len(kp_xy):
                joints = kp_xy[i]
                confs_j = kp_conf[i] if kp_conf is not None and i < len(kp_conf) else None
                for j, (jx, jy) in enumerate(joints):
                    jc = float(confs_j[j]) if confs_j is not None and j < len(confs_j) else None
                    keypoints.append(Keypoint(x=float(jx), y=float(jy), conf=jc))
            persons.append(
                PersonObservation(
                    track_id=tid,
                    bbox=[x1, y1, x2, y2],
                    bbox_conf=float(confs[i]),
                    cls=cls_name,
                    keypoints=keypoints,
                )
            )
        return persons

    # ------------------------------------------------------------------
    def close(self) -> None:
        self._model = None


def _np(t):
    if t is None:
        return None
    if hasattr(t, "cpu"):
        return t.cpu().numpy()
    return np.asarray(t)


def _sha1(path: str, chunk: int = 1 << 20) -> str:
    if not os.path.isfile(path):
        return ""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:16]
