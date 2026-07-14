"""Step-level Android GUI reward for EasyR1 UI-S1 experiments."""

import json
import math
import re
from typing import Any


REWARD_NAME = "ui_s1_step"
REWARD_TYPE = "batch"

ACTION_ARGUMENTS = {
    "open": ("text",),
    "click": ("coordinate",),
    "long_press": ("coordinate",),
    "swipe": ("coordinate", "coordinate2"),
    "type": ("text",),
    "system_button": ("button",),
    "wait": ("time",),
    "terminate": ("status",),
}
OPTIONAL_ACTION_ARGUMENTS = {
    "long_press": ("time",),
}
TERMINATE_STATUSES = {"success", "failure"}
SYSTEM_BUTTONS = {"back", "home", "enter"}
MODEL_RESPONSE_PATTERN = re.compile(
    r"\s*<thinking>(?P<thinking>.*?)</thinking>\s*"
    r"<tool_call>\s*(?P<tool_call>\{.*\})\s*</tool_call>\s*",
    flags=re.DOTALL,
)

def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def is_valid_action_schema(action: Any, allow_reward_metadata: bool = False) -> bool:
    if not isinstance(action, dict):
        return False
    action_name = action.get("action")
    required = ACTION_ARGUMENTS.get(action_name)
    if required is None or any(field not in action for field in required):
        return False
    allowed_fields = {"action", *required, *OPTIONAL_ACTION_ARGUMENTS.get(action_name, ())}
    if allow_reward_metadata:
        # AMEX labels retain these fields for coordinate-tolerance scoring.
        allowed_fields.update({"bbox", "device_dim"})
    if set(action) - allowed_fields:
        return False
    if action_name in {"click", "long_press"} and not is_json_point(action["coordinate"]):
        return False
    if action_name == "swipe" and (not is_json_point(action["coordinate"]) or not is_json_point(action["coordinate2"])):
        return False
    if action_name in {"open", "type"} and not isinstance(action["text"], str):
        return False
    if action_name == "wait" or (action_name == "long_press" and "time" in action):
        try:
            if not math.isfinite(float(action["time"])):
                return False
        except (TypeError, ValueError):
            return False
    if action_name == "system_button":
        if not isinstance(action["button"], str) or normalize_text(action["button"]) not in SYSTEM_BUTTONS:
            return False
    if action_name == "terminate":
        if not isinstance(action["status"], str) or normalize_text(action["status"]) not in TERMINATE_STATUSES:
            return False
    return True


def is_json_point(value: Any) -> bool:
    point = as_point(value)
    return isinstance(value, list) and len(value) == 2 and point is not None and all(math.isfinite(item) for item in point)


def unwrap_mobile_use_tool_call(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict) or set(value) != {"name", "arguments"}:
        return None
    if value["name"] != "mobile_use" or not isinstance(value["arguments"], dict):
        return None
    return value["arguments"]


def parse_action(text: Any, require_tool_call: bool = False) -> tuple[dict[str, Any] | None, float]:
    if isinstance(text, dict):
        if "name" in text or "arguments" in text:
            text = unwrap_mobile_use_tool_call(text)
        return text, 1.0 if not require_tool_call and is_valid_action_schema(text, allow_reward_metadata=True) else 0.0

    raw = str(text or "").strip()
    if not raw:
        return None, 0.0

    response_match = MODEL_RESPONSE_PATTERN.fullmatch(raw)
    candidates = []
    if response_match and response_match.group("thinking").strip():
        candidates.append((response_match.group("tool_call"), 1.0, True))
    if not require_tool_call:
        candidates.append((raw, 1.0, False))
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match:
            candidates.append((match.group(0), 1.0, False))

    for candidate, format_score, is_tool_call in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        if is_tool_call:
            parsed = unwrap_mobile_use_tool_call(parsed)
        if is_valid_action_schema(parsed, allow_reward_metadata=not require_tool_call):
            return parsed, format_score

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


def screen_size(action: dict[str, Any]) -> tuple[float, float]:
    dim = action.get("device_dim")
    if isinstance(dim, (list, tuple)) and len(dim) >= 2:
        try:
            width = max(float(dim[0]), 1.0)
            height = max(float(dim[1]), 1.0)
            return width, height
        except (TypeError, ValueError):
            pass
    return 1080.0, 2400.0


def normalized_distance(a: tuple[float, float], b: tuple[float, float], size: tuple[float, float]) -> float:
    width, height = size
    dx = (a[0] - b[0]) / width
    dy = (a[1] - b[1]) / height
    return math.sqrt(dx * dx + dy * dy)


def score_point_action(pred: dict[str, Any], gt: dict[str, Any]) -> float:
    pred_point = point_from_action(pred)
    gt_point = point_from_action(gt)
    if pred_point is None or gt_point is None:
        return 0.0

    gt_bbox = as_bbox(gt.get("bbox"))
    if gt_bbox is not None:
        return 1.0 if point_in_bbox(pred_point, gt_bbox) else 0.0

    return 1.0 if normalized_distance(pred_point, gt_point, screen_size(gt)) <= 0.08 else 0.0


def score_time_field(pred: dict[str, Any], gt: dict[str, Any]) -> float:
    try:
        return 1.0 if math.isclose(float(pred["time"]), float(gt["time"]), abs_tol=0.25) else 0.0
    except (KeyError, TypeError, ValueError):
        return 0.0


def swipe_direction(start: tuple[float, float], end: tuple[float, float]) -> str | None:
    delta_x, delta_y = end[0] - start[0], end[1] - start[1]
    if delta_x == 0.0 and delta_y == 0.0:
        return None
    if abs(delta_x) > abs(delta_y):
        return "right" if delta_x > 0.0 else "left"
    return "down" if delta_y > 0.0 else "up"


def score_swipe(pred: dict[str, Any], gt: dict[str, Any]) -> float:
    pred_start = as_point(pred.get("coordinate"))
    pred_end = as_point(pred.get("coordinate2"))
    gt_start = as_point(gt.get("coordinate"))
    gt_end = as_point(gt.get("coordinate2"))
    if None in (pred_start, pred_end, gt_start, gt_end):
        return 0.0

    return 1.0 if swipe_direction(pred_start, pred_end) == swipe_direction(gt_start, gt_end) else 0.0


def score_text_field(pred: dict[str, Any], gt: dict[str, Any], key: str) -> float:
    return 1.0 if normalize_text(pred.get(key)) == normalize_text(gt.get(key)) else 0.0


def score_action(pred: dict[str, Any], gt: dict[str, Any]) -> float:
    action_type = normalize_text(gt.get("action"))
    if action_type == "click":
        return score_point_action(pred, gt)
    if action_type == "long_press":
        point_score = score_point_action(pred, gt)
        return point_score * score_time_field(pred, gt) if "time" in gt else point_score
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
        # Wait duration is syntactic metadata, not an expert-action match target.
        return 1.0
    return 0.0


def score_one(response: Any, ground_truth: Any) -> dict[str, float]:
    pred, format_score = parse_action(response, require_tool_call=True)
    gt, gt_format = parse_action(ground_truth)
    if pred is None or gt is None or gt_format == 0.0:
        return {"overall": 0.0, "format": format_score, "type": 0.0, "accuracy": 0.0}

    type_score = 1.0 if format_score == 1.0 and normalize_text(pred.get("action")) == normalize_text(gt.get("action")) else 0.0
    action_score = score_action(pred, gt) if type_score == 1.0 else 0.0
    overall = 0.1 * format_score + 0.4 * format_score * type_score + 0.5 * format_score * type_score * action_score
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
            "response": '<thinking>click the target</thinking>\n<tool_call>\n{"name":"mobile_use","arguments":{"action":"click","coordinate":[540,390]}}\n</tool_call>',
            "ground_truth": '{"action":"click","coordinate":[540,389.8],"bbox":[360,327,695,466]}',
        },
        {
            "response": '<thinking>open the app</thinking>\n<tool_call>\n{"name":"mobile_use","arguments":{"action":"open","text":"Zoho Meeting"}}\n</tool_call>',
            "ground_truth": '{"action":"open","text":"Zoho Meeting"}',
        },
        {
            "response": '<tool_call>\n{"name":"mobile_use","arguments":{"action":"click","coordinate":[10,10]}}\n</tool_call>',
            "ground_truth": '{"action":"click","coordinate":[540,389.8],"bbox":[360,327,695,466]}',
        },
        {
            "response": "not json",
            "ground_truth": '{"action":"wait","time":2}',
        },
    ]
    for score in compute_score(test_cases):
        print(score)
