from examples.ui_s1.monitor_gpu_memory import parse_nvidia_smi_rows


def test_parse_nvidia_smi_rows_handles_numeric_and_na_utilization():
    rows = parse_nvidia_smi_rows("0, 24576, 12288, 90\n1, 24576, 10, N/A\n")

    assert rows == [
        {"index": 0, "total_mib": 24576, "used_mib": 12288, "utilization_percent": 90},
        {"index": 1, "total_mib": 24576, "used_mib": 10, "utilization_percent": None},
    ]
