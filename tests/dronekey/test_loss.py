import math

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from dronepose.keypoints.dronekey.loss import (KeypointLoss, gaussian_terms, keypoint_covariance,  # noqa: E402
                                               scale_schedule)


def _cov_from(offsets):
    y = 0.5 + torch.tensor([offsets], dtype=torch.float64)  # [1, 4, 2]
    return keypoint_covariance(y)[0]


def test_covariance_fig3_cases():
    sq = torch.tensor([[0., 0.], [1., 0.], [1., 1.], [0., 1.]], dtype=torch.float64)[None]
    assert torch.allclose(keypoint_covariance(sq)[0], torch.eye(2, dtype=torch.float64) * 0.25, atol=1e-6)
    for u2, v2, diag, off in ((0.2, 0.02, 0.11, 0.09), (0.005, 0.199, 0.102, -0.097)):
        u, v = math.sqrt(u2), math.sqrt(v2)
        c = _cov_from([[u, u], [-u, -u], [v, -v], [-v, v]])
        assert torch.allclose(c, torch.tensor([[diag, off], [off, diag]], dtype=torch.float64), atol=1e-6)


@pytest.mark.parametrize("e,expected", list(zip((1, 5, 10, 20, 40, 80, 100),
                                                (1.000, 0.819, 0.638, 0.387, 0.142, 0.019, 0.007))))
def test_scale_schedule_matches_draft_table(e, expected):
    assert scale_schedule(e - 1) / 10 == pytest.approx(expected, abs=1e-3)


def test_scale_schedule_compressed():
    assert scale_schedule(19, epochs=20) == pytest.approx(scale_schedule(99))


def test_gaussian_terms_draft_loss_values():
    delta = torch.tensor([[[0.1, 0.0]]])
    for s, expected in zip((1, 0.5, 0.1, 0.05, 0.001), (0.045, 0.088, 0.372, 0.605, 1.0)):
        cov_t = (s ** 2 * torch.diag(torch.tensor([1.1563, 1.0])))[None]
        assert gaussian_terms(delta, cov_t).item() == pytest.approx(expected, abs=0.005)


def test_identity_cov_is_euclidean_distance():
    torch.manual_seed(0)
    delta = torch.randn(3, 4, 2)
    cov = torch.eye(2).expand(3, 2, 2)
    out = gaussian_terms(delta, cov)
    assert torch.allclose(out, 1 - torch.exp(-0.5 * delta.norm(dim=-1)), atol=1e-6)


def test_exact_prediction_zero_loss_finite_grad():
    torch.manual_seed(5)
    t = torch.rand(3, 8)
    for kind in ("gaussian", "pose_adaptive"):
        pred = t.clone().requires_grad_(True)
        crit = KeypointLoss(kind)
        crit.set_epoch(50)
        out = crit(pred, t)
        out.backward()
        assert out.item() == pytest.approx(0.0, abs=1e-5) and torch.isfinite(pred.grad).all()


def test_rotation_invariance():
    torch.manual_seed(1)
    gt = torch.rand(5, 4, 2)
    err = 0.05 * torch.randn(5, 4, 2)
    a = 0.7
    R = torch.tensor([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    crit = KeypointLoss("pose_adaptive")
    crit.set_epoch(30)
    l0 = crit((gt + err).reshape(5, 8), gt.reshape(5, 8))
    gt_r, err_r = gt @ R.T, err @ R.T
    l1 = crit((gt_r + err_r).reshape(5, 8), gt_r.reshape(5, 8))
    assert l0.item() == pytest.approx(l1.item(), abs=1e-5)


def test_loss_finite_for_targets_outside_crop():
    target = torch.tensor([[-0.1, 0.3, 1.2, 0.3, 1.2, 0.9, -0.1, 0.9]])
    crit = KeypointLoss("pose_adaptive")
    crit.set_epoch(10)
    out = crit(target + 0.02, target)
    assert out.dtype == torch.float32 and torch.isfinite(out)


def test_loss_collinear_keypoints_finite_grad():
    target = torch.tensor([[0.1, 0.1, 0.3, 0.3, 0.5, 0.5, 0.7, 0.7]]).repeat(2, 1)
    pred = (target + 0.01).half().requires_grad_(True)
    crit = KeypointLoss("pose_adaptive")
    crit.set_epoch(99)
    out = crit(pred, target)
    out.backward()
    assert torch.isfinite(out) and torch.isfinite(pred.grad).all()


def test_mse_kind():
    torch.manual_seed(2)
    p, t = torch.rand(6, 8), torch.rand(6, 8)
    assert KeypointLoss("mse")(p, t).item() == pytest.approx(F.mse_loss(p, t).item(), abs=1e-7)


def test_gaussian_kind_uses_identity_with_same_schedule():
    torch.manual_seed(3)
    t = torch.rand(4, 8)
    p = t + 0.05
    crit = KeypointLoss("gaussian", alpha=5, D=10, eps=1e-4)
    crit.set_epoch(7)
    s = scale_schedule(7, 5.0, 10.0, 100, 100)
    d2 = ((p - t).reshape(4, 4, 2) ** 2).sum(-1)
    expected = (1 - torch.exp(-0.5 * torch.sqrt(d2 / (s + 1e-4)))).mean()
    assert crit(p, t).item() == pytest.approx(expected.item(), abs=1e-6)


def test_loss_grows_as_scale_shrinks_for_fixed_error():
    # loss grows as S(t) shrinks (spec §2.3): same error, later epoch -> larger loss
    torch.manual_seed(4)
    t = torch.rand(4, 8)
    p = t + 0.03
    crit = KeypointLoss("pose_adaptive")
    crit.set_epoch(0)
    early = crit(p, t).item()
    crit.set_epoch(90)
    assert crit(p, t).item() > early


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        KeypointLoss("huber")
