from examples.ui_s1.reward_ui_s1_step import score_one


def test_ui_s1_reward_requires_thinking_tool_call_and_mobile_use_schema():
    response = """<thinking>
Open the target application.
</thinking>
<tool_call>
{"name":"mobile_use","arguments":{"action":"open","text":"Calendar"}}
</tool_call>"""
    ground_truth = '{"action":"open","text":"Calendar"}'
    assert score_one(response, ground_truth)["overall"] == 1.0

    missing_thinking = '<tool_call>\n{"name":"mobile_use","arguments":{"action":"open","text":"Calendar"}}\n</tool_call>'
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
