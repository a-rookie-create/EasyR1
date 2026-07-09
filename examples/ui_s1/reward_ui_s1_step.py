"""Step-level Android GUI reward for EasyR1 UI-S1 experiments."""

import json
import math
import re
from typing import Any


REWARD_NAME = "ui_s1_step"
REWARD_TYPE = "batch"

SCREEN_WIDTH = 1080.0
SCREEN_HEIGHT = 2400.0


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def parse_action(text: Any) -> tuple[dict[str, Any] | None, float]:
    if isinstance(text, dict):
        return text, 1.0 if "action" in text else 0.0

    raw = str(text or "").strip()
    if not raw:
        return None, 0.0

    candidates = [raw]
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        candidates.insert(0, match.group(0))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "action" in parsed:
            return parsed, 1.0

    return None, 0.0


def as_point(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None


def as_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in value[:4]]
    except (TypeError, ValueError):
        return None
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def point_from_action(action: dict[str, Any]) -> tuple[float, float] | None:
    point = as_point(action.get("coordinate"))
    if point is not None:
        return point

    bbox = as_bbox(action.get("bbox"))
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    return None


def point_in_bbox(point: tuple[float, float], bbox: tuple[float, float, float, float]) -> bool:
    x, y = point
    x1, y1, x2, y2 = bbox
    pad_x = max(8.0, (x2 - x1) * 0.15)
    pad_y = max(8.0, (y2 - y1) * 0.15)
    return (x1 - pad_x) <= x <= (x2 + pad_x) and (y1 - pad_y) <= y <= (y2 + pad_y)


def normalized_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    dx = (a[0] - b[0]) / SCREEN_WIDTH
    dy = (a[1] - b[1]) / SCREEN_HEIGHT
    return math.sqrt(dx * dx + dy * dy)


def score_point_action(pred: dict[str, Any], gt: dict[str, Any]) -> float:
    pred_point = point_from_action(pred)
    gt_point = point_from_action(gt)
    if pred_point is None or gt_point is None:
        return 0.0

    gt_bbox = as_bbox(gt.get("bbox"))
    if gt_bbox is not None:
        return 1.0 if point_in_bbox(pred_point, gt_bbox) else 0.0

    return 1.0 if normalized_distance(pred_point, gt_point) <= 0.08 else 0.0


def vector(start: tuple[float, float], end: tuple[float, float]) -> tuple[float, float]:
    return end[0] - start[0], end[1] - start[1]


def score_swipe(pred: dict[str, Any], gt: dict[str, Any]) -> float:
    pred_start = as_point(pred.get("coordinate"))
    pred_end = as_point(pred.get("coordinate2"))
    gt_start = as_point(gt.get("coordinate"))
    gt_end = as_point(gt.get("coordinate2"))
    if None in (pred_start, pred_end, gt_start, gt_end):
        return 0.0

    pred_vec = vector(pred_start, pred_end)
    gt_vec = vector(gt_start, gt_end)
    pred_norm = math.hypot(*pred_vec)
    gt_norm = math.hypot(*gt_vec)
    if pred_norm == 0.0 or gt_norm == 0.0:
        return 0.0

    cosine = (pred_vec[0] * gt_vec[0] + pred_vec[1] * gt_vec[1]) / (pred_norm * gt_norm)
    start_ok = normalized_distance(pred_start, gt_start) <= 0.25
    end_ok = normalized_distance(pred_end, gt_end) <= 0.25
    return 1.0 if cosine >= 0.7 and (start_ok or end_ok) else 0.0


def score_text_field(pred: dict[str, Any], gt: dict[str, Any], key: str) -> float:
    return 1.0 if normalize_text(pred.get(key)) == normalize_text(gt.get(key)) else 0.0


def score_action(pred: dict[str, Any], gt: dict[str, Any]) -> float:
    action_type = normalize_text(gt.get("action"))
    if action_type in {"click", "long_press"}:
        return score_point_action(pred, gt)
    if action_type == "swipe":
        return score_swipe(pred, gt)
    if action_type in {"type", "open"}:
        return score_text_field(pred, gt, "text")
    if action_type == "system_button":
        return score_text_field(pred, gt, "button")
    if action_type == "terminate":
        if "status" not in gt:
            return 1.0
        return score_text_field(pred, gt, "status")
    if action_type == "wait":
        return 1.0
    return 1.0


def score_one(response: Any, ground_truth: Any) -> dict[str, float]:
    pred, format_score = parse_action(response)
    gt, gt_format = parse_action(ground_truth)
    if pred is None or gt is None or gt_format == 0.0:
        return {"overall": 0.0, "format": format_score, "type": 0.0, "accuracy": 0.0}

    type_score = 1.0 if normalize_text(pred.get("action")) == normalize_text(gt.get("action")) else 0.0
    action_score = score_action(pred, gt) if type_score == 1.0 else 0.0
    overall = 0.1 * format_score + 0.4 * type_score + 0.5 * action_score
    return {
        "overall": float(overall),
        "format": float(format_score),
        "type": float(type_score),
        "accuracy": float(action_score),
    }


def compute_score(reward_inputs: list[dict[str, Any]]) -> list[dict[str, float]]:
    scores = []
    for reward_input in reward_inputs:
        scores.append(score_one(reward_input.get("response", ""), reward_input.get("ground_truth", "")))
    return scores


if __name__ == "__main__":
    test_cases = [
        {
            "response": '{"action":"click","coordinate":[540,390]}',
            "ground_truth": '{"action":"click","coordinate":[540,389.8],"bbox":[360,327,695,466]}',
        },
        {
            "response": '{"action":"open","text":"Zoho Meeting"}',
            "ground_truth": '{"action":"open","text":"Zoho Meeting"}',
        },
        {
            "response": '{"action":"click","coordinate":[10,10]}',
            "ground_truth": '{"action":"click","coordinate":[540,389.8],"bbox":[360,327,695,466]}',
        },
        {
            "response": "not json",
            "ground_truth": '{"action":"wait","time":2}',
        },
    ]
    for score in compute_score(test_cases):
        print(score)
