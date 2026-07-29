import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "examples" / "ui_s1" / "checkpoint_lineage.py"
SPEC = importlib.util.spec_from_file_location("checkpoint_lineage", MODULE_PATH)
lineage = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(lineage)


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


class CheckpointLineageTest(unittest.TestCase):
    def make_source_run(self, root: Path) -> tuple[Path, Path]:
        run_dir = root / "source"
        checkpoint = run_dir / "global_step_99"
        adapter = checkpoint / "actor" / "lora_adapter"
        adapter.mkdir(parents=True)
        (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
        (checkpoint / "dataloader.pt").write_bytes(b"state")
        (run_dir / "experiment_config.json").write_text(
            json.dumps(
                {
                    "data": {
                        "train_files": "/data/train.jsonl",
                        "shuffle": True,
                        "seed": 1,
                        "rollout_batch_size": 4,
                        "mini_rollout_batch_size": None,
                    }
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "training_progress.log").write_text(
            "RUN START\n"
            "STEPLESS\n"
            "2026 | STEP 99 | CHECKPOINT_SAVE | END\n"
            "2026 | STEP 99 | STEP | END\n"
            "2026 | STEP 100 | STEP | START\n"
            "RUN RESUME\n"
            "MODEL_PROCESSOR | START\n"
            "2026 | STEP 0 | TRAINING_LOOP | START\n"
            "2026 | STEP 99 | CHECKPOINT_LOAD | END\n",
            encoding="utf-8",
        )
        write_jsonl(run_dir / "experiment_log.jsonl", [{"step": 99}, {"step": 100}])
        write_jsonl(run_dir / "semi_online_rollouts.jsonl", [{"global_step": 99}, {"global_step": 100}])
        return run_dir, checkpoint

    def test_warm_start_copies_only_history_and_lineage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, checkpoint = self.make_source_run(root)
            target = root / "warm"
            lineage.prepare_warm_start(
                checkpoint,
                target,
                "inherit",
                {
                    "train_files": "/data/train.jsonl",
                    "shuffle": True,
                    "seed": 1,
                    "rollout_batch_size": 4,
                    "mini_rollout_batch_size": None,
                },
            )
            self.assertEqual([json.loads(line)["step"] for line in (target / "experiment_log.jsonl").read_text().splitlines()], [99])
            self.assertEqual(
                [json.loads(line)["global_step"] for line in (target / "semi_online_rollouts.jsonl").read_text().splitlines()],
                [99],
            )
            progress = (target / "training_progress.log").read_text(encoding="utf-8")
            self.assertIn("STEP 99", progress)
            self.assertNotIn("STEP 100", progress)
            self.assertNotIn("RUN RESUME", progress)
            self.assertNotIn("STEP 0", progress)
            self.assertTrue(progress.rstrip().endswith("| STEP 99 | STEP | END"))
            self.assertTrue((target / "warm_start" / "source_dataloader.pt").is_file())
            manifest = json.loads((target / "warm_start.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["source_global_step"], 99)

    def test_warm_start_rejects_incompatible_inherited_data(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, checkpoint = self.make_source_run(root)
            with self.assertRaisesRegex(ValueError, "WARM_START_DATALOADER=inherit"):
                lineage.prepare_warm_start(
                    checkpoint,
                    root / "warm",
                    "inherit",
                    {
                        "train_files": "/data/other.jsonl",
                        "shuffle": True,
                        "seed": 1,
                        "rollout_batch_size": 4,
                        "mini_rollout_batch_size": None,
                    },
                )

    def test_resume_rollback_truncates_logs_tracker_and_checkpoints(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run_dir, checkpoint = self.make_source_run(root)
            (run_dir / "global_step_100").mkdir()
            (run_dir / "global_step_104").mkdir()
            (run_dir / "checkpoint_tracker.json").write_text(
                json.dumps({"last_global_step": 104, "best_global_step": 104, "best_val_reward_score": 1.0}),
                encoding="utf-8",
            )
            (run_dir / "train.log").write_text("opaque", encoding="utf-8")
            lineage.prepare_resume_rollback(run_dir, checkpoint)
            self.assertFalse((run_dir / "global_step_100").exists())
            self.assertFalse((run_dir / "global_step_104").exists())
            self.assertEqual([json.loads(line)["step"] for line in (run_dir / "experiment_log.jsonl").read_text().splitlines()], [99])
            tracker = json.loads((run_dir / "checkpoint_tracker.json").read_text())
            self.assertEqual(tracker["last_global_step"], 99)
            self.assertEqual(tracker["best_global_step"], 0)
            archive = run_dir / "rollback_archive" / "before_global_step_99"
            self.assertTrue((archive / "train.log").is_file())
            manifest = json.loads((archive / "rollback.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["history_records"]["experiment_log.jsonl"], {"before": 2, "after": 1, "removed": 1})
            self.assertEqual(manifest["removed_checkpoints"], ["global_step_100", "global_step_104"])


if __name__ == "__main__":
    unittest.main()
