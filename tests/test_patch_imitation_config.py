import json
import unittest

from omegaconf import OmegaConf

from verl.trainer.config import (
    AlgorithmConfig,
    PatchImitationConfig,
    PPOConfig,
    TrainerConfig,
    compute_patch_imitation_lambda,
)


def _enabled_config(**overrides) -> PatchImitationConfig:
    values = {
        "enabled": True,
        "lambda_initial": 0.8,
        "lambda_decay": 0.5,
        "lambda_min": 0.2,
        "target_mode": "action_only",
        "history_mode": "keep_model_thinking",
    }
    values.update(overrides)
    config = PatchImitationConfig(**values)
    config.post_init()
    return config


class PatchImitationConfigTest(unittest.TestCase):
    def test_defaults_are_disabled_and_json_serializable(self):
        config = PPOConfig()
        config.deep_post_init()

        patch_config = config.algorithm.patch_imitation
        self.assertFalse(patch_config.enabled)
        self.assertEqual(patch_config.lambda_initial, 0.0)
        self.assertEqual(patch_config.lambda_decay, 1.0)
        self.assertEqual(patch_config.lambda_min, 0.0)
        self.assertEqual(patch_config.lambda_cutoff_step, 0)
        self.assertEqual(patch_config.target_mode, "action_only")
        self.assertEqual(patch_config.history_mode, "keep_model_thinking")
        serialized = json.loads(json.dumps(config.to_dict()))
        self.assertFalse(serialized["algorithm"]["patch_imitation"]["enabled"])
        self.assertTrue(serialized["trainer"]["val_after_train"])

    def test_final_validation_can_be_disabled_independently(self):
        base = OmegaConf.structured(PPOConfig())
        cli = OmegaConf.from_dotlist(["trainer.val_after_train=false"])

        config = OmegaConf.to_object(OmegaConf.merge(base, cli))
        config.deep_post_init()

        self.assertFalse(config.trainer.val_after_train)
        self.assertFalse(config.trainer.val_only)

    def test_structured_cli_override_accepts_scientific_notation(self):
        base = OmegaConf.structured(PPOConfig())
        cli = OmegaConf.from_dotlist(
            [
                "algorithm.semi_online=true",
                "algorithm.patch_imitation.enabled=true",
                "algorithm.patch_imitation.lambda_initial=1e-2",
                "algorithm.patch_imitation.lambda_decay=9.5e-1",
                "algorithm.patch_imitation.lambda_min=1e-3",
                "algorithm.patch_imitation.lambda_cutoff_step=80",
                "algorithm.patch_imitation.target_mode=action_only",
                "algorithm.patch_imitation.history_mode=keep_model_thinking",
            ]
        )

        config = OmegaConf.to_object(OmegaConf.merge(base, cli))
        config.deep_post_init()

        self.assertTrue(config.algorithm.patch_imitation.enabled)
        self.assertAlmostEqual(config.algorithm.patch_imitation.lambda_initial, 0.01)
        self.assertAlmostEqual(config.algorithm.patch_imitation.lambda_decay, 0.95)
        self.assertAlmostEqual(config.algorithm.patch_imitation.lambda_min, 0.001)
        self.assertEqual(config.algorithm.patch_imitation.lambda_cutoff_step, 80)

    def test_lambda_is_one_based_and_respects_floor(self):
        config = _enabled_config()

        self.assertAlmostEqual(compute_patch_imitation_lambda(config, 1), 0.8)
        self.assertAlmostEqual(compute_patch_imitation_lambda(config, 2), 0.4)
        self.assertAlmostEqual(compute_patch_imitation_lambda(config, 3), 0.2)
        self.assertAlmostEqual(compute_patch_imitation_lambda(config, 10), 0.2)

    def test_lambda_resume_sequence_has_no_step_offset(self):
        config = _enabled_config(lambda_initial=1.0, lambda_decay=0.9, lambda_min=0.1)
        continuous = [compute_patch_imitation_lambda(config, step) for step in range(1, 9)]

        checkpoint_step = 5
        before_checkpoint = [compute_patch_imitation_lambda(config, step) for step in range(1, checkpoint_step + 1)]
        after_resume = [compute_patch_imitation_lambda(config, step) for step in range(checkpoint_step + 1, 9)]

        self.assertEqual(before_checkpoint + after_resume, continuous)
        self.assertAlmostEqual(
            after_resume[0],
            config.lambda_initial * config.lambda_decay**checkpoint_step,
        )

    def test_disabled_configuration_has_zero_effective_lambda(self):
        config = PatchImitationConfig()
        config.post_init()

        self.assertEqual(compute_patch_imitation_lambda(config, 1), 0.0)
        self.assertEqual(compute_patch_imitation_lambda(config, 100), 0.0)

    def test_lambda_cutoff_zeros_and_skips_imitation_after_its_last_step(self):
        config = _enabled_config(
            lambda_initial=1.0,
            lambda_decay=0.9433732216,
            lambda_min=0.0,
            lambda_cutoff_step=80,
        )

        self.assertAlmostEqual(compute_patch_imitation_lambda(config, 50), 0.057476944, places=9)
        self.assertAlmostEqual(compute_patch_imitation_lambda(config, 80), 0.01, places=9)
        self.assertEqual(compute_patch_imitation_lambda(config, 81), 0.0)

    def test_lambda_requires_positive_integer_step(self):
        for global_step in (0, -1, 1.5, True):
            with self.subTest(global_step=global_step):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    compute_patch_imitation_lambda(_enabled_config(), global_step)

    def test_invalid_lambda_configuration_fails_fast(self):
        invalid_cases = [
            ({"lambda_initial": -0.1}, "lambda_initial must be non-negative"),
            ({"enabled": True, "lambda_initial": 0.0}, "lambda_initial must be positive"),
            ({"lambda_decay": 0.0}, "lambda_decay must be in"),
            ({"lambda_decay": 1.1}, "lambda_decay must be in"),
            ({"lambda_min": -0.1}, "lambda_min must be non-negative"),
            ({"lambda_initial": 0.2, "lambda_min": 0.3}, "lambda_min must not exceed"),
            ({"lambda_cutoff_step": -1}, "lambda_cutoff_step must be a non-negative integer"),
            ({"lambda_cutoff_step": 1.5}, "lambda_cutoff_step must be a non-negative integer"),
            ({"lambda_initial": float("nan")}, "must be a finite number"),
            ({"lambda_decay": float("inf")}, "must be a finite number"),
        ]
        defaults = {
            "enabled": False,
            "lambda_initial": 0.0,
            "lambda_decay": 1.0,
            "lambda_min": 0.0,
        }

        for overrides, message in invalid_cases:
            with self.subTest(overrides=overrides):
                values = {**defaults, **overrides}
                with self.assertRaisesRegex(ValueError, message):
                    PatchImitationConfig(**values).post_init()

    def test_future_modes_fail_fast(self):
        invalid_modes = [
            ("target_mode", "thinking_and_action", "currently supports only `action_only`"),
            (
                "history_mode",
                "replace_with_expert_thinking",
                "currently supports only `keep_model_thinking`",
            ),
        ]
        for field, value, message in invalid_modes:
            with self.subTest(field=field, value=value):
                with self.assertRaisesRegex(ValueError, message):
                    PatchImitationConfig(**{field: value}).post_init()

    def test_patch_imitation_requires_semi_online_rollout(self):
        algorithm = AlgorithmConfig(semi_online=False, patch_imitation=_enabled_config())

        with self.assertRaisesRegex(ValueError, "requires algorithm.semi_online=true"):
            algorithm.post_init()

    def test_explicit_missing_resume_checkpoint_fails_fast(self):
        config = TrainerConfig(load_checkpoint_path="/definitely/missing/global_step_1")

        with self.assertRaisesRegex(ValueError, "load_checkpoint_path does not exist"):
            config.post_init()


if __name__ == "__main__":
    unittest.main()
