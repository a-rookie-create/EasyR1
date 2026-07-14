import torch

from examples.ui_s1.advantage_ui_s1 import compute_ui_s1_advantages, replace_nearest_rollout, rollout_score_std


def test_ui_s1_advantages_use_full_episode_and_natural_step_segments():
    # Deliberately shuffle rows to verify grouping does not rely on batch order.
    rewards = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    mask = torch.tensor([[1, 0], [1, 0], [1, 0], [1, 0]], dtype=torch.bool)
    result = compute_ui_s1_advantages(
        token_level_rewards=rewards,
        response_mask=mask,
        task_ids=["task", "task", "task", "task"],
        trajectory_ids=["rollout_b", "rollout_a", "rollout_a", "rollout_b"],
        step_ids=[1, 0, 1, 0],
        extract_matches=[True, True, False, True],
        gamma=0.5,
        normalize_by_std=False,
    )

    # rollout_a: rewards [1, 0], rollout_b: rewards [1, 1]
    assert torch.allclose(result.episode_returns, torch.tensor([2.0, 1.0, 1.0, 2.0]))
    # The mismatch at rollout_a step 1 prevents its reward from reaching step 0.
    assert torch.allclose(result.step_returns, torch.tensor([1.0, 1.0, 0.0, 1.5]))
    assert torch.allclose(result.episode_advantages, torch.tensor([0.5, -0.5, -0.5, 0.5]))
    assert torch.allclose(result.step_advantages, torch.tensor([0.5, -0.25, -0.5, 0.25]))

    expected = torch.tensor([[1.0, 0.0], [-0.75, 0.0], [-1.0, 0.0], [0.75, 0.0]])
    assert torch.allclose(result.advantages, expected)
    assert torch.equal(result.advantages, result.returns)


def test_ui_s1_singleton_groups_have_zero_normalized_advantage():
    result = compute_ui_s1_advantages(
        token_level_rewards=torch.tensor([[0.5]]),
        response_mask=torch.tensor([[True]]),
        task_ids=["only-task"],
        trajectory_ids=["only-rollout"],
        step_ids=[0],
        extract_matches=[True],
    )

    assert torch.allclose(result.advantages, torch.zeros((1, 1)))


def test_ui_s1_step_return_continues_between_two_patches():
    result = compute_ui_s1_advantages(
        token_level_rewards=torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]]),
        response_mask=torch.ones((5, 1), dtype=torch.bool),
        task_ids=["task"] * 5,
        trajectory_ids=["rollout"] * 5,
        step_ids=[0, 1, 2, 3, 4],
        extract_matches=[False, True, True, False, True],
        gamma=0.5,
        normalize_by_std=False,
    )

    # Steps 1 and 2 are between two patches, so step 1 receives step 2's reward.
    assert torch.allclose(result.step_returns, torch.tensor([1.0, 3.5, 3.0, 4.0, 5.0]))


def test_diversity_selection_replaces_only_a_near_mean_rollout():
    selected_scores = [-0.8, -0.1, 0.0, 0.2]
    replace, index, candidate_distance, nearest_distance = replace_nearest_rollout(selected_scores, 1.4)
    assert replace
    assert index == 3
    assert candidate_distance > nearest_distance

    replace, index, _, _ = replace_nearest_rollout(selected_scores, -0.13)
    assert not replace
    assert index == 1
    assert rollout_score_std(selected_scores) > 0.0
