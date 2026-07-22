import math

from examples.ui_s1.reward_ui_s1_step import extract_action, score_one, transform_action_coordinates


def test_ui_s1_reward_requires_tagged_mobile_use_format():
    response = """<thinking>
Open the target application.
</thinking>
<tool_call>
{"name":"mobile_use","arguments":{"action":"open","text":"Calendar"}}
</tool_call>"""
    ground_truth = '{"action":"open","text":"Calendar"}'
    assert score_one(response, ground_truth)["overall"] == 1.0

    direct_response = """<thinking>Open the target application.</thinking>
- {"name":"mobile_use","arguments":{"action":"open","text":"Calendar"}}"""
    assert score_one(direct_response, ground_truth)["format"] == 0.0

    missing_thinking = '{"name":"mobile_use","arguments":{"action":"open","text":"Calendar"}}'
    assert score_one(missing_thinking, ground_truth)["format"] == 0.0

    invalid_status = """<thinking>x</thinking>
<tool_call>
{"name":"mobile_use","arguments":{"action":"terminate","status":"impossible"}}
</tool_call>"""
    assert score_one(invalid_status, '{"action":"terminate","status":"failure"}')["format"] == 0.0


def test_amex_reward_metadata_is_accepted_only_for_ground_truth():
    response = """<thinking>Tap the target.</thinking>
<tool_call>
{"name":"mobile_use","arguments":{"action":"click","coordinate":[100,200]}}
</tool_call>"""
    annotated_ground_truth = '{"action":"click","coordinate":[100,200],"bbox":[90,190,110,210],"device_dim":[1080,2400]}'
    assert score_one(response, annotated_ground_truth)["overall"] == 1.0


def test_model_response_requires_exact_json_wrapper_and_no_extra_text():
    ground_truth = '{"action":"system_button","button":"Back"}'
    valid = """<thinking>Go back.</thinking>
<tool_call>{"name":"mobile_use","arguments":{"action":"system_button","button":"Back"}}</tool_call>"""
    assert score_one(valid, ground_truth)["format"] == 1.0

    malformed_json = """<thinking>Go back.</thinking>
<tool_call>{action:"system_button","button":"Back"}</tool_call>"""
    assert score_one(malformed_json, ground_truth)["format"] == 0.0

    extra_wrapper_field = """<thinking>Go back.</thinking>
<tool_call>{"name":"mobile_use","arguments":{"action":"system_button","button":"Back"},"id":"1"}</tool_call>"""
    assert score_one(extra_wrapper_field, ground_truth)["format"] == 0.0

    extra_text = "note\n" + valid
    assert score_one(extra_text, ground_truth)["format"] == 0.0


def test_android_control_long_press_time_matches_actual_sft_schema():
    response = """<thinking>Hold the target.</thinking>
<tool_call>{"name":"mobile_use","arguments":{"action":"long_press","coordinate":[100,200]}}</tool_call>"""
    ground_truth = '{"action":"long_press","coordinate":[100,200]}'
    assert score_one(response, ground_truth)["overall"] == 1.0


def test_system_button_and_terminate_values_are_case_insensitive_but_enumerated():
    button_response = """<thinking>Go back.</thinking>
<tool_call>{"name":"mobile_use","arguments":{"action":"system_button","button":"back"}}</tool_call>"""
    assert score_one(button_response, '{"action":"system_button","button":"Back"}')["overall"] == 1.0

    terminate_response = """<thinking>The task cannot be completed.</thinking>
<tool_call>{"name":"mobile_use","arguments":{"action":"terminate","status":"FAILURE"}}</tool_call>"""
    assert score_one(terminate_response, '{"action":"terminate","status":"failure"}')["overall"] == 1.0

    invalid_button = """<thinking>Use a system button.</thinking>
<tool_call>{"name":"mobile_use","arguments":{"action":"system_button","button":"Menu"}}</tool_call>"""
    assert score_one(invalid_button, '{"action":"system_button","button":"Back"}')["format"] == 0.0


def test_wait_duration_is_validated_but_not_compared_and_swipe_matches_direction_only():
    wait_response = """<thinking>Wait for the screen.</thinking>
<tool_call>{"name":"mobile_use","arguments":{"action":"wait","time":0.1}}</tool_call>"""
    assert score_one(wait_response, '{"action":"wait","time":10}')["overall"] == 1.0

    swipe_response = """<thinking>Scroll upward.</thinking>
<tool_call>{"name":"mobile_use","arguments":{"action":"swipe","coordinate":[10,2300],"coordinate2":[1000,100]}}</tool_call>"""
    swipe_ground_truth = '{"action":"swipe","coordinate":[540,1800],"coordinate2":[540,600]}'
    assert score_one(swipe_response, swipe_ground_truth)["overall"] == 1.0


def test_reward_inverse_transforms_qwen_coordinates_with_independent_xy_scales():
    transform = {
        "original_width": 1440,
        "original_height": 3120,
        "model_width": 980,
        "model_height": 2128,
    }
    ground_truth = '{"action":"click","coordinate":[240,1589],"device_dim":[1440,3120]}'
    model_coordinate = [240 * 980 / 1440, 1589 * 2128 / 3120]
    response = (
        "<thinking>Tap the target.</thinking><tool_call>"
        f'{{"name":"mobile_use","arguments":{{"action":"click","coordinate":{model_coordinate}}}}}'
        "</tool_call>"
    )

    assert score_one(response, ground_truth)["accuracy"] == 0.0
    assert score_one(response, ground_truth, transform)["overall"] == 1.0
    extracted_coordinate = extract_action(response, transform)["coordinate"]
    assert math.isclose(extracted_coordinate[0], 240.0)
    assert math.isclose(extracted_coordinate[1], 1589.0)


def test_coordinate_transform_scales_both_swipe_endpoints_without_mutating_action():
    transform = {
        "original_width": 1440,
        "original_height": 3120,
        "model_width": 980,
        "model_height": 2128,
    }
    action = {"action": "swipe", "coordinate": [240, 1589], "coordinate2": [1200, 3000]}
    scaled = transform_action_coordinates(action, transform)
    restored = transform_action_coordinates(scaled, transform, inverse=True)

    assert action["coordinate"] == [240, 1589]
    assert all(
        math.isclose(actual, expected)
        for actual, expected in zip(scaled["coordinate"], [240 * 980 / 1440, 1589 * 2128 / 3120])
    )
    assert all(
        math.isclose(actual, expected)
        for actual, expected in zip(scaled["coordinate2"], [1200 * 980 / 1440, 3000 * 2128 / 3120])
    )
    assert all(math.isclose(actual, expected) for actual, expected in zip(restored["coordinate"], [240.0, 1589.0]))
    assert all(math.isclose(actual, expected) for actual, expected in zip(restored["coordinate2"], [1200.0, 3000.0]))
