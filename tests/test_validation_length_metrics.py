from verl.trainer.metrics import reduce_length_metric_samples


def test_validation_length_statistics_use_global_sample_maximum_and_minimum():
    metrics = reduce_length_metric_samples(
        {
            "response_length": [50.0, 80.0, 100.0],
            "prompt_length": [3000.0, 3200.0, 3400.0],
            "response_length/clip": [0.0, 0.0, 1.0],
            "prompt_length/clip": [0.0, 0.0, 0.0],
        }
    )

    assert metrics == {
        "response_length/mean": 230.0 / 3.0,
        "response_length/max": 100.0,
        "response_length/min": 50.0,
        "response_length/clip_ratio": 1.0 / 3.0,
        "prompt_length/mean": 3200.0,
        "prompt_length/max": 3400.0,
        "prompt_length/min": 3000.0,
        "prompt_length/clip_ratio": 0.0,
    }
