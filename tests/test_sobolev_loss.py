from __future__ import annotations

import torch

from models.mlp.training import RelativeL2Loss, SobolevLoss, compute_wall_distance_weights


def _build_knn_indices(pos: torch.Tensor, k: int) -> torch.Tensor:
    batch_size, num_points, _ = pos.shape
    all_indices = torch.empty((batch_size, num_points, k), dtype=torch.long)
    for batch_idx in range(batch_size):
        dist = torch.cdist(pos[batch_idx], pos[batch_idx])
        dist.fill_diagonal_(float("inf"))
        all_indices[batch_idx] = torch.topk(dist, k=k, largest=False, dim=1).indices
    return all_indices


def test_sobolev_loss_matches_rl2_without_spatial_context() -> None:
    torch.manual_seed(0)
    pred = torch.randn((2, 3, 5, 2), dtype=torch.float32)
    target = torch.randn((2, 3, 5, 2), dtype=torch.float32)

    sobolev = SobolevLoss(grad_weight=0.5)
    rl2 = RelativeL2Loss()

    assert torch.allclose(sobolev(pred, target), rl2(pred, target))


def test_sobolev_loss_decompose_zero_for_identical_inputs() -> None:
    torch.manual_seed(1)
    pos = torch.rand((1, 6, 3), dtype=torch.float32)
    field = torch.rand((1, 2, 6, 3), dtype=torch.float32)
    knn_indices = _build_knn_indices(pos, k=2)

    sobolev = SobolevLoss(grad_weight=0.5)
    rl2_term, grad_term = sobolev.decompose(field, field, knn_indices=knn_indices, pos=pos)
    loss = sobolev(field, field, knn_indices=knn_indices, pos=pos)

    assert rl2_term.item() == 0.0
    assert grad_term.item() == 0.0
    assert loss.item() == 0.0


def test_sobolev_loss_gradient_term_is_scale_invariant() -> None:
    torch.manual_seed(2)
    pos = torch.rand((1, 8, 3), dtype=torch.float32)
    target = torch.rand((1, 2, 8, 3), dtype=torch.float32) + 0.5
    pred = target.clone()
    pred[:, :, :, 0] *= 1.15
    knn_indices = _build_knn_indices(pos, k=3)

    sobolev = SobolevLoss(grad_weight=0.5)
    rl2_term, grad_term = sobolev.decompose(pred, target, knn_indices=knn_indices, pos=pos)

    scale = 7.0
    scaled_rl2_term, scaled_grad_term = sobolev.decompose(
        pred * scale,
        target * scale,
        knn_indices=knn_indices,
        pos=pos,
    )

    assert grad_term.item() > 0.0
    assert torch.allclose(rl2_term, scaled_rl2_term, atol=1e-6, rtol=1e-5)
    assert torch.allclose(grad_term, scaled_grad_term, atol=1e-6, rtol=1e-5)
    assert torch.allclose(
        sobolev(pred, target, knn_indices=knn_indices, pos=pos),
        rl2_term + (sobolev.grad_weight * grad_term),
    )


def test_wall_distance_weights_shape_and_normalisation() -> None:
    torch.manual_seed(3)
    pos = torch.rand((2, 10, 3), dtype=torch.float32)
    idcs = [torch.tensor([0, 1]), torch.tensor([3, 4, 5])]

    weights = compute_wall_distance_weights(pos, idcs, alpha=1.0)
    assert weights.shape == (2, 1, 10, 1)
    # Mean of weights should be ~1 per sample (normalised)
    for b in range(2):
        assert abs(weights[b, 0, :, 0].mean().item() - 1.0) < 1e-5


def test_wall_distance_weights_upweight_near_boundary() -> None:
    torch.manual_seed(4)
    pos = torch.rand((1, 10, 3), dtype=torch.float32)
    idcs = [torch.tensor([0])]

    weights = compute_wall_distance_weights(pos, idcs, alpha=1.0)
    # The airfoil point itself (index 0) should have high weight (dist ~ 0)
    airfoil_weight = weights[0, 0, 0, 0].item()
    far_weights = weights[0, 0, 1:, 0]
    assert airfoil_weight > far_weights.mean().item()


def test_rl2_with_wall_weights_differs_from_unweighted() -> None:
    torch.manual_seed(5)
    pred = torch.randn((1, 2, 6, 3), dtype=torch.float32)
    target = torch.randn((1, 2, 6, 3), dtype=torch.float32)
    pos = torch.rand((1, 6, 3), dtype=torch.float32)
    idcs = [torch.tensor([0, 1])]

    wall_weights = compute_wall_distance_weights(pos, idcs, alpha=1.0)
    rl2 = RelativeL2Loss()

    unweighted = rl2(pred, target)
    weighted = rl2(pred, target, wall_weights=wall_weights)
    # Weighted and unweighted should generally differ
    assert not torch.allclose(unweighted, weighted, atol=1e-6)


def test_sobolev_with_wall_weights_changes_loss() -> None:
    torch.manual_seed(6)
    pos = torch.rand((1, 8, 3), dtype=torch.float32)
    target = torch.rand((1, 2, 8, 3), dtype=torch.float32) + 0.5
    pred = target.clone()
    pred[:, :, :, 0] *= 1.15
    knn_indices = _build_knn_indices(pos, k=3)
    idcs = [torch.tensor([0, 1])]

    wall_weights = compute_wall_distance_weights(pos, idcs, alpha=1.0)
    sobolev = SobolevLoss(grad_weight=0.5)

    loss_no_w = sobolev(pred, target, knn_indices=knn_indices, pos=pos)
    loss_w = sobolev(pred, target, knn_indices=knn_indices, pos=pos, wall_weights=wall_weights)
    # The weighted loss should differ from the unweighted loss
    assert not torch.allclose(loss_no_w, loss_w, atol=1e-6)


def test_wall_distance_weights_precomputed_matches_on_the_fly() -> None:
    torch.manual_seed(7)
    pos = torch.rand((2, 10, 3), dtype=torch.float32)
    idcs = [torch.tensor([0, 1]), torch.tensor([3, 4, 5])]

    # Precompute wall distance the same way the dataset does
    wall_distance = torch.zeros((2, 10), dtype=torch.float32)
    for b, idx in enumerate(idcs):
        airfoil_pos = pos[b, idx.long()]
        dists = torch.cdist(pos[b].unsqueeze(0), airfoil_pos.unsqueeze(0)).squeeze(0)
        wall_distance[b] = dists.min(dim=-1).values

    weights_fly = compute_wall_distance_weights(pos, idcs, alpha=1.5)
    weights_pre = compute_wall_distance_weights(pos, idcs, alpha=1.5, wall_distance=wall_distance)
    assert torch.allclose(weights_fly, weights_pre, atol=1e-6)