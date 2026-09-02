# tests/test_sbi.py
"""Tests for the transfit.sbi subpackage.

All tests use pytest.importorskip("sbi") so they are skipped gracefully
when the sbi package is not installed.
"""
from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sbi = pytest.importorskip("sbi")

import transfit as tf
from transfit.sbi.prior import TransFitPrior
from transfit.sbi.simulator import make_bolometric_simulator, make_multiband_simulator
from transfit.sbi.embedding import SetSummaryNet, MLPEmbeddingNet, encode_observations, encode_batch
from transfit.sbi.training import generate_training_data, _filter_nans
from transfit.sbi.posterior import SBIPosterior
from transfit.sbi.io import save_posterior, load_posterior, _reconstruct_embedding_net
from transfit.sbi import _build_density_estimator
from transfit.priors import build_bounds, MixedBoundsPrior
from transfit.api import _split_prior_specs, _apply_log10_priors, _split_sampling


# ---- Fixtures ----

@pytest.fixture
def nickel_prior():
    """Build a MixedBoundsPrior for the nickel model with 3 free params."""
    priors_lin, priors_log10 = _split_prior_specs({
        "M_ej": (1.0, 8.0),
        "v_ej": (0.3, 2.0),
        "M_ni": (0.01, 0.5),
    })
    names_all, bounds_all = build_bounds("nickel", priors=priors_lin, include_t_shift=True)
    bounds_all, log_set_all = _apply_log10_priors(names_all, bounds_all, priors_log10)
    fixed = {"f_ni": 0.5, "kappa_gamma": 0.03, "kappa": 0.2, "R_0": 10.0, "E_Th_in": 0.0, "T_floor": 5000.0, "t_shift": 0.0, "delta": 0.0, "n": 10.0}
    names_samp, bounds_samp, fixed_dict = _split_sampling(names_all, bounds_all, fixed=fixed)
    log_flags_samp = [n in log_set_all for n in names_samp]
    return MixedBoundsPrior(bounds=bounds_samp, param_names=names_samp, log_flags=log_flags_samp), names_samp, names_all, fixed_dict


@pytest.fixture
def tf_prior_nickel(nickel_prior):
    mp, names_samp, names_all, fixed = nickel_prior
    return TransFitPrior(mp), names_samp, names_all, fixed


# ---- Test: Prior Adapter ----

class TestPriorAdapter:

    def test_sample_shape(self, tf_prior_nickel):
        prior, names_samp, _, _ = tf_prior_nickel
        samples = prior.sample((100,))
        assert samples.shape == (100, len(names_samp))

    def test_sample_within_bounds(self, tf_prior_nickel, nickel_prior):
        tf_prior, _, _, _ = tf_prior_nickel
        mp, _, _, _ = nickel_prior
        samples = tf_prior.sample((500,))
        lo = mp.bounds[:, 0]
        hi = mp.bounds[:, 1]
        samples_np = np.asarray(samples)
        assert np.all(samples_np < hi)
        assert np.all(lo < samples_np)

    def test_log_prob_in_bounds(self, tf_prior_nickel):
        prior, _, _, _ = tf_prior_nickel
        # Sample a valid point
        theta = prior.sample((10,))
        lp = prior.log_prob(theta)
        assert lp.shape == (10,)
        assert torch.all(torch.isfinite(lp))

    def test_log_prob_out_of_bounds(self, tf_prior_nickel, nickel_prior):
        prior, _, _, _ = tf_prior_nickel
        mp, _, _, _ = nickel_prior
        # Point outside bounds
        bad = torch.tensor([[-999.0] * mp.bounds.shape[0]], dtype=torch.float32)
        lp = prior.log_prob(bad)
        assert lp[0].item() == float("-inf")


# ---- Test: Simulator ----

class TestSimulator:

    def test_bolometric_output_shape(self, tf_prior_nickel, nickel_prior):
        prior, names_samp, names_all, fixed = tf_prior_nickel
        t_days = np.array([10.0, 20.0, 30.0, 50.0])
        sim = make_bolometric_simulator(
            model="nickel", z=0.01, t_days=t_days,
            param_names=names_samp, names_all=names_all, fixed=fixed,
            Nx=10, Ny=20,
        )
        theta = prior.sample((5,))
        x = sim(theta)
        assert x.shape == (5, 4)
        assert x.dtype == torch.float32

    def test_bolometric_nan_handling(self, tf_prior_nickel, nickel_prior):
        prior, names_samp, names_all, fixed = tf_prior_nickel
        t_days = np.array([10.0, 30.0])
        sim = make_bolometric_simulator(
            model="nickel", z=0.01, t_days=t_days,
            param_names=names_samp, names_all=names_all, fixed=fixed,
            Nx=10, Ny=20,
        )
        # Very extreme parameters should produce NaN
        bad_theta = torch.tensor([[1e10, 1e10, 1e10]], dtype=torch.float32)
        x = sim(bad_theta)
        # Should be NaN (simulator catches exceptions)
        assert x.shape == (1, 2)

    def test_multiband_output_shape(self, tf_prior_nickel, nickel_prior):
        prior, names_samp, names_all, fixed = tf_prior_nickel
        t_days = np.array([10.0, 20.0, 30.0, 50.0])
        band = np.array(["B", "B", "V", "V"], dtype=object)
        sim = make_multiband_simulator(
            model="nickel", z=0.01, distance_modulus=30.0,
            filters={"B": "johnson_cousins.B", "V": "johnson_cousins.V"},
            t_days=t_days, band=band, y_kind="mag",
            param_names=names_samp, names_all=names_all, fixed=fixed,
            Nx=10, Ny=20,
        )
        theta = prior.sample((3,))
        x = sim(theta)
        assert x.shape == (3, 4)
        assert x.dtype == torch.float32


# ---- Test: Embedding Networks ----

class TestEmbedding:

    def test_set_summary_output_dim(self):
        net = SetSummaryNet(feature_dim=4, hidden_features=32, output_dim=16)
        # feature_dim=4 → phi sees 3 features, last col is validity indicator
        x = torch.randn(5, 10, 4)
        # Set validity column to 1.0 so mask is inferred correctly
        x[..., -1] = 1.0
        out = net(x)
        assert out.shape == (5, 16)

    def test_set_summary_with_mask(self):
        net = SetSummaryNet(feature_dim=3, hidden_features=32, output_dim=8)
        x = torch.randn(3, 8, 3)
        mask = torch.ones(3, 8, dtype=torch.bool)
        mask[0, 5:] = False
        mask[1, 3:] = False
        out = net(x, mask)
        assert out.shape == (3, 8)

    def test_feature_normalization_excludes_mask_and_padding(self):
        x = torch.tensor(
            [
                [[5.0, 10.0, 1.0], [5.0, 20.0, 1.0], [999.0, 999.0, 0.0]],
                [[5.0, 30.0, 1.0], [5.0, 40.0, 1.0], [5.0, 50.0, 1.0]],
            ],
            dtype=torch.float32,
        )
        original_validity = x[..., -1].clone()
        valid = x[..., -1] != 0.0
        expected = x[..., :-1][valid]

        net = SetSummaryNet(feature_dim=3, hidden_features=8, output_dim=4)
        net.fit_normalization(x)

        torch.testing.assert_close(net.feature_mean, expected.mean(dim=0))
        expected_scale = expected.std(dim=0, unbiased=False)
        expected_scale = torch.where(
            expected_scale > net.normalization_eps,
            expected_scale,
            torch.ones_like(expected_scale),
        )
        torch.testing.assert_close(net.feature_scale, expected_scale)
        torch.testing.assert_close(x[..., -1], original_validity)

    def test_fixed_cadence_mask_survives_normalization(self):
        torch.manual_seed(0)
        x = torch.tensor(
            [
                [[0.0, 10.0, 1.0], [0.5, 10.0, 1.0], [1.0, 10.0, 1.0]],
                [[0.0, 20.0, 1.0], [0.5, 20.0, 1.0], [1.0, 20.0, 1.0]],
            ],
            dtype=torch.float32,
        )
        explicit_mask = torch.ones(2, 3, dtype=torch.bool)
        net = SetSummaryNet(feature_dim=3, hidden_features=8, output_dim=4)
        net.fit_normalization(x)

        inferred = net(x)
        explicit = net(x, explicit_mask)

        torch.testing.assert_close(inferred, explicit)
        assert not torch.allclose(inferred[0], inferred[1])

    def test_density_estimator_disables_external_x_standardization(self, monkeypatch):
        import sbi.neural_nets as sbi_neural_nets

        captured = {}
        marker = object()

        def fake_posterior_nn(**kwargs):
            captured.update(kwargs)
            return marker

        monkeypatch.setattr(sbi_neural_nets, "posterior_nn", fake_posterior_nn)
        emb = SetSummaryNet(feature_dim=3, hidden_features=8, output_dim=4)

        result = _build_density_estimator(
            embedding_net=emb,
            hidden_features=8,
            num_transforms=2,
        )

        assert result is marker
        assert captured["embedding_net"] is emb
        assert captured["z_score_x"] == "none"

    def test_legacy_unpickled_embedding_still_forwards(self):
        emb = SetSummaryNet(feature_dim=3, hidden_features=8, output_dim=4)
        del emb.feature_mean
        del emb.feature_scale
        del emb.feature_dim
        del emb.normalize_features
        del emb.normalization_eps
        x = torch.tensor(
            [[[0.0, 10.0, 1.0], [1.0, 20.0, 1.0]]],
            dtype=torch.float32,
        )

        out = emb(x)

        assert out.shape == (1, 4)

    def test_mlp_embedding(self):
        net = MLPEmbeddingNet(input_dim=10, hidden_features=32, output_dim=8)
        x = torch.randn(4, 10)
        out = net(x)
        assert out.shape == (4, 8)

    def test_encode_observations_multiband(self):
        y = np.array([15.0, 16.0, 14.5])
        t = np.array([10.0, 20.0, 30.0])
        band = np.array(["B", "V", "B"], dtype=object)
        vocab = ["B", "V"]

        features, mask = encode_observations(
            y, t_days=t, band=band, band_vocabulary=vocab,
            t_range=(0.0, 100.0),
        )
        assert features.shape == (3, 4)  # t_norm + 2 onehot + y
        assert mask.shape == (3,)
        assert mask.all()
        # Check B one-hot encoding
        assert features[0, 1] == 1.0  # B
        assert features[0, 2] == 0.0  # not V
        assert features[1, 1] == 0.0
        assert features[1, 2] == 1.0  # V

    def test_encode_observations_bolometric(self):
        y = np.array([42.5, 43.0, 42.8])
        t = np.array([10.0, 20.0, 30.0])

        features, mask = encode_observations(y, t_days=t, t_range=(0.0, 100.0))
        assert features.shape == (3, 2)  # t_norm + y
        assert mask.all()

    def test_encode_batch_padding(self):
        batch_y = [
            np.array([15.0, 16.0]),
            np.array([14.0, 14.5, 15.0]),
        ]
        features, masks = encode_batch(
            batch_y,
            t_days_list=[np.array([10.0, 20.0]), np.array([10.0, 20.0, 30.0])],
            t_range=(0.0, 100.0),
        )
        assert features.shape == (2, 3, 3)  # batch=2, max_n_obs=3, feature_dim=2+1 (validity)
        assert masks[0, 2] == False  # padded entry
        assert masks[1, :].all()


# ---- Test: Training Data Generation ----

class TestTrainingData:

    def test_generate_and_filter(self, tf_prior_nickel, nickel_prior):
        prior, names_samp, names_all, fixed = tf_prior_nickel
        t_days = np.array([10.0, 20.0, 30.0])
        sim = make_bolometric_simulator(
            model="nickel", z=0.01, t_days=t_days,
            param_names=names_samp, names_all=names_all, fixed=fixed,
            Nx=10, Ny=20,
        )
        theta, x, stats = generate_training_data(
            simulator=sim, prior=prior, n_simulations=20,
            seed=42, show_progress=False,
        )
        assert theta.shape[0] == x.shape[0]
        assert x.shape[1] == 3
        assert stats["n_valid"] > 0

    def test_cache_roundtrip(self, tf_prior_nickel, nickel_prior):
        prior, names_samp, names_all, fixed = tf_prior_nickel
        t_days = np.array([10.0, 20.0])
        sim = make_bolometric_simulator(
            model="nickel", z=0.01, t_days=t_days,
            param_names=names_samp, names_all=names_all, fixed=fixed,
            Nx=10, Ny=20,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = os.path.join(tmpdir, "test_cache.npz")
            theta1, x1, s1 = generate_training_data(
                simulator=sim, prior=prior, n_simulations=15,
                seed=42, show_progress=False, cache_path=cache,
            )
            assert os.path.exists(cache)
            theta2, x2, s2 = generate_training_data(
                simulator=sim, prior=prior, n_simulations=15,
                seed=42, show_progress=False, cache_path=cache,
            )
            assert s2["from_cache"] is True
            np.testing.assert_array_equal(
                np.asarray(theta1), np.asarray(theta2)
            )
            np.testing.assert_array_equal(
                np.asarray(x1), np.asarray(x2)
            )


# ---- Test: NPE Training E2E ----

class TestNPETraining:

    def test_npe_train_and_sample(self, tf_prior_nickel, nickel_prior):
        """End-to-end: generate data, train NPE, sample posterior."""
        prior, names_samp, names_all, fixed = tf_prior_nickel
        t_days = np.array([10.0, 20.0, 30.0, 50.0])
        sim = make_bolometric_simulator(
            model="nickel", z=0.01, t_days=t_days,
            param_names=names_samp, names_all=names_all, fixed=fixed,
            Nx=10, Ny=20,
        )

        theta, x, stats = generate_training_data(
            simulator=sim, prior=prior, n_simulations=200,
            seed=42, show_progress=False,
        )
        assert stats["n_valid"] > 50  # Enough data to train

        from sbi.inference import SNPE
        from sbi.neural_nets import posterior_nn

        # Use MLP embedding for fixed-size (simpler for this test)
        emb = MLPEmbeddingNet(input_dim=4, hidden_features=16, output_dim=8)
        density_estimator = posterior_nn(
            model="maf",
            embedding_net=emb,
            hidden_features=16,
            num_transforms=2,
        )

        inference = SNPE(prior=prior, density_estimator=density_estimator)
        inference.append_simulations(theta, x)
        density_estimator = inference.train(
            max_num_epochs=5,
            training_batch_size=20,
            show_train_summary=False,
        )
        sbi_post = inference.build_posterior(density_estimator)

        # Sample from posterior with a mock observation
        x_obs = x[0]
        samples = sbi_post.sample((100,), x=x_obs, show_progress_bars=False)
        assert samples.shape == (100, len(names_samp))


# ---- Test: Posterior within bounds ----

class TestPosteriorBounds:

    def test_samples_within_prior_bounds(self, tf_prior_nickel, nickel_prior):
        prior, names_samp, names_all, fixed = tf_prior_nickel
        mp, _, _, _ = nickel_prior
        t_days = np.array([10.0, 20.0, 30.0])
        sim = make_bolometric_simulator(
            model="nickel", z=0.01, t_days=t_days,
            param_names=names_samp, names_all=names_all, fixed=fixed,
            Nx=10, Ny=20,
        )

        theta, x, stats = generate_training_data(
            simulator=sim, prior=prior, n_simulations=200,
            seed=42, show_progress=False,
        )

        if stats["n_valid"] < 50:
            pytest.skip("Not enough valid simulations")

        from sbi.inference import SNPE
        from sbi.neural_nets import posterior_nn

        emb = MLPEmbeddingNet(input_dim=3, hidden_features=16, output_dim=8)
        de = posterior_nn(model="maf", embedding_net=emb, hidden_features=16, num_transforms=2)
        inf = SNPE(prior=prior, density_estimator=de)
        inf.append_simulations(theta, x)
        de = inf.train(max_num_epochs=5, training_batch_size=20, show_train_summary=False)
        sbi_post = inf.build_posterior(de)

        x_obs = x[0]
        samples = sbi_post.sample((500,), x=x_obs, show_progress_bars=False)
        lo = mp.bounds[:, 0]
        hi = mp.bounds[:, 1]

        # Most samples should be within bounds (allow some tolerance for flow tails)
        frac_in = np.mean(
            np.all((np.asarray(samples) > lo) & (np.asarray(samples) < hi), axis=1)
        )
        assert frac_in > 0.8


# ---- Test: End-to-end train_sbi pipeline ----

class TestTrainSBIE2E:

    def test_fixed_cadence_embedding_uses_observation(self):
        """Regression: a fixed all-valid mask must not erase observation context."""
        t_days = np.array([10.0, 20.0, 30.0])
        posterior = tf.sbi.train_sbi(
            model="nickel",
            mode="bolometric",
            z=0.01,
            priors={
                "M_ej": (1.0, 8.0),
                "v_ej": (0.3, 2.0),
                "M_ni": (0.01, 0.5),
            },
            fixed={
                "f_ni": 0.5, "kappa_gamma": 0.03, "kappa": 0.2,
                "R_0": 10.0, "E_Th_in": 0.0, "T_floor": 5000.0, "t_shift": 0.0,
            },
            cadence_templates=[{"t_days": t_days}],
            n_simulations=100,
            max_num_epochs=2,
            hidden_features=8,
            num_transforms=2,
            training_batch_size=20,
            show_progress=False,
            Nx=10, Ny=20,
        )

        x_low = posterior._encode_observation(
            np.array([40.0, 40.0, 40.0]),
            t_days=t_days,
        )
        x_high = posterior._encode_observation(
            np.array([44.0, 44.0, 44.0]),
            t_days=t_days,
        )
        with torch.no_grad():
            embedding_low = posterior.embedding_net(x_low)
            embedding_high = posterior.embedding_net(x_high)

        assert torch.all(x_low[..., -1] == 1.0)
        assert torch.all(x_high[..., -1] == 1.0)
        assert not torch.allclose(embedding_low, embedding_high)
        assert posterior.meta["x_standardization"] == "mask_safe_embedding"

    def test_train_sbi_bolometric(self, tf_prior_nickel, nickel_prior):
        """End-to-end: train_sbi with bolometric mode and default SetSummaryNet."""
        prior, names_samp, names_all, fixed = tf_prior_nickel

        posterior = tf.sbi.train_sbi(
            model="nickel",
            mode="bolometric",
            z=0.01,
            priors={
                "M_ej": (1.0, 8.0),
                "v_ej": (0.3, 2.0),
                "M_ni": (0.01, 0.5),
            },
            fixed={
                "f_ni": 0.5, "kappa_gamma": 0.03, "kappa": 0.2,
                "R_0": 10.0, "E_Th_in": 0.0, "T_floor": 5000.0, "t_shift": 0.0,
            },
            n_simulations=200,
            max_num_epochs=5,
            hidden_features=16,
            num_transforms=2,
            training_batch_size=20,
            show_progress=False,
            Nx=10, Ny=20,
        )

        assert isinstance(posterior, SBIPosterior)
        assert posterior.model == "nickel"
        assert posterior.mode == "bolometric"
        assert len(posterior.param_names) == len(names_samp)

        # Generate a mock observation and run inference
        sim = make_bolometric_simulator(
            model="nickel", z=0.01, t_days=np.array([10.0, 20.0, 30.0]),
            param_names=names_samp, names_all=names_all, fixed=fixed,
            Nx=10, Ny=20,
        )
        theta_test = prior.sample((1,))
        x_test = sim(theta_test)
        y_obs = np.asarray(x_test[0].detach().numpy(), float)

        if not np.all(np.isfinite(y_obs)):
            pytest.skip("Simulation produced NaN, skipping inference test")

        samples = posterior.sample(50, y_obs, t_days=np.array([10.0, 20.0, 30.0]))
        assert samples.shape == (50, len(names_samp))

        median = posterior.median(y_obs, t_days=np.array([10.0, 20.0, 30.0]))
        assert isinstance(median, dict)
        assert all(name in median for name in posterior.param_names)

    def test_infer_sbi_helper(self, tf_prior_nickel, nickel_prior):
        """Test infer_sbi convenience wrapper."""
        prior, names_samp, names_all, fixed = tf_prior_nickel

        posterior = tf.sbi.train_sbi(
            model="nickel",
            mode="bolometric",
            z=0.01,
            priors={
                "M_ej": (1.0, 8.0),
                "v_ej": (0.3, 2.0),
                "M_ni": (0.01, 0.5),
            },
            fixed={
                "f_ni": 0.5, "kappa_gamma": 0.03, "kappa": 0.2,
                "R_0": 10.0, "E_Th_in": 0.0, "T_floor": 5000.0, "t_shift": 0.0,
            },
            n_simulations=200,
            max_num_epochs=5,
            hidden_features=16,
            num_transforms=2,
            training_batch_size=20,
            show_progress=False,
            Nx=10, Ny=20,
        )

        sim = make_bolometric_simulator(
            model="nickel", z=0.01, t_days=np.array([10.0, 20.0, 30.0]),
            param_names=names_samp, names_all=names_all, fixed=fixed,
            Nx=10, Ny=20,
        )
        theta_test = prior.sample((1,))
        x_test = sim(theta_test)
        y_obs = np.asarray(x_test[0].detach().numpy(), float)

        if not np.all(np.isfinite(y_obs)):
            pytest.skip("Simulation produced NaN")

        result = tf.sbi.infer_sbi(
            posterior, y_obs,
            t_days=np.array([10.0, 20.0, 30.0]),
            n_samples=50,
        )
        assert "samples" in result
        assert "median" in result
        assert "map" in result
        assert "param_names" in result
        assert result["samples"].shape[0] == 50


# ---- Test: IO roundtrip ----

class TestIO:

    def test_save_load_roundtrip(self):
        """Test save/load with a minimal SBIPosterior mock."""
        emb = MLPEmbeddingNet(input_dim=3, hidden_features=16, output_dim=8)

        # Create a minimal mock posterior (just for IO test)
        class MockPosterior:
            pass

        post = SBIPosterior(
            model="nickel",
            param_names=["M_ej", "v_ej", "M_ni"],
            posterior=MockPosterior(),
            embedding_net=emb,
            meta={"test": True},
            mode="bolometric",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_post.pt")
            save_posterior(post, path)
            assert os.path.exists(path)

            loaded = load_posterior(path, trusted=True)
            assert loaded.model == "nickel"
            assert loaded.param_names == ["M_ej", "v_ej", "M_ni"]
            assert loaded.mode == "bolometric"
            assert loaded.posterior is None
            assert loaded.meta["posterior_serialized"] is False

    def test_load_requires_trusted(self):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save({"mock": True}, f.name)
            path = f.name
        try:
            with pytest.raises(ValueError, match="trusted=True"):
                load_posterior(path)
        finally:
            os.unlink(path)

    def test_set_summary_normalization_roundtrip(self):
        emb = SetSummaryNet(feature_dim=3, hidden_features=8, output_dim=4)
        x = torch.tensor(
            [[[0.0, 10.0, 1.0], [1.0, 20.0, 1.0]]],
            dtype=torch.float32,
        )
        emb.fit_normalization(x)
        post = SBIPosterior(
            model="nickel",
            param_names=["M_ej"],
            posterior=None,
            embedding_net=emb,
            mode="bolometric",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "set_summary.pt")
            save_posterior(post, path)
            loaded = load_posterior(path, trusted=True)

        assert isinstance(loaded.embedding_net, SetSummaryNet)
        assert loaded.embedding_net.normalize_features is True
        torch.testing.assert_close(
            loaded.embedding_net.feature_mean,
            emb.feature_mean,
        )
        torch.testing.assert_close(
            loaded.embedding_net.feature_scale,
            emb.feature_scale,
        )

    def test_loads_legacy_set_summary_state(self):
        emb = SetSummaryNet(feature_dim=3, hidden_features=8, output_dim=4)
        legacy_state = {
            key: value
            for key, value in emb.state_dict().items()
            if key not in {"feature_mean", "feature_scale"}
        }
        loaded = _reconstruct_embedding_net(
            "SetSummaryNet",
            {"feature_dim": 3, "hidden_features": 8, "output_dim": 4},
            legacy_state,
        )

        torch.testing.assert_close(loaded.feature_mean, torch.zeros(2))
        torch.testing.assert_close(loaded.feature_scale, torch.ones(2))


class TestPosteriorDevice:

    def test_sample_and_log_prob_follow_posterior_device(self):
        emb = MLPEmbeddingNet(input_dim=6, hidden_features=16, output_dim=8)

        class MockPosterior:
            def __init__(self):
                self.posterior_estimator = torch.nn.Linear(6, 3)
                self.potential_fn = SimpleNamespace(
                    posterior_estimator=self.posterior_estimator
                )
                self.x_devices = []
                self.theta_devices = []

            def sample(self, sample_shape, *, x, show_progress_bars=False):
                self.x_devices.append(x.device.type)
                return torch.zeros(
                    (sample_shape[0], 3), dtype=torch.float32, device=x.device
                )

            def log_prob(self, theta, *, x):
                self.x_devices.append(x.device.type)
                self.theta_devices.append(theta.device.type)
                return torch.zeros(theta.shape[0], dtype=torch.float32, device=theta.device)

        mock = MockPosterior()
        post = SBIPosterior(
            model="nickel",
            param_names=["M_ej", "v_ej", "M_ni"],
            posterior=mock,
            embedding_net=emb,
            mode="bolometric",
        )

        y_obs = np.array([42.5, 43.0, 42.8])
        t_days = np.array([10.0, 20.0, 30.0])

        samples = post.sample(5, y_obs, t_days=t_days)
        lp = post.log_prob(np.ones((2, 3)), y_obs, t_days=t_days)

        assert samples.shape == (5, 3)
        assert lp.shape == (2,)
        assert mock.x_devices == ["cpu", "cpu"]
        assert mock.theta_devices == ["cpu"]

    def test_to_updates_nested_sbi_device_state(self):
        emb = MLPEmbeddingNet(input_dim=3, hidden_features=16, output_dim=8)

        class DummyPrior:
            def __init__(self):
                self.device = "cuda:0"

            def to(self, device):
                self.device = str(device)
                return self

        class MockPosterior:
            def __init__(self):
                self._device = "cuda:0"
                self.device = "cuda:0"
                self.prior = DummyPrior()
                self.posterior_estimator = torch.nn.Linear(3, 3)
                self.potential_fn = SimpleNamespace(
                    device="cuda:0",
                    prior=DummyPrior(),
                    posterior_estimator=self.posterior_estimator,
                )

        post = SBIPosterior(
            model="nickel",
            param_names=["M_ej", "v_ej", "M_ni"],
            posterior=MockPosterior(),
            embedding_net=emb,
            mode="bolometric",
        )

        post.to("cpu")

        assert post.posterior.device == "cpu"
        assert post.posterior._device == "cpu"
        assert post.posterior.prior.device == "cpu"
        assert post.posterior.potential_fn.device == "cpu"
        assert post.posterior.potential_fn.prior.device == "cpu"
