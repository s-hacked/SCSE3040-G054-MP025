"""
Tests for Abhibhav Manhas' (S24CSEU2780) operations components:
- PSI drift monitoring
- Harbour-wise MAE fairness monitoring
- Chronological 70/30 split
- Promotion rule (12% threshold)
- Kubernetes YAML validation
- GitHub Actions CI YAML validation
"""

from __future__ import annotations

import os
import yaml
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock

from src.mp025.monitoring import (
    compute_psi_single,
    compute_psi_all_features,
    compute_mae_by_harbour,
    chronological_split_later,
    evaluate_v1_v2,
    update_findings_monitoring,
    NUMERIC_FEATURES,
    PROMOTION_THRESHOLD,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def synthetic_train_df() -> pd.DataFrame:
    """Synthetic cleaned training data with all numeric features."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "record_id": [f"R{i:04d}" for i in range(n)],
        "date": pd.date_range("2026-01-01", periods=n, freq="D").astype(str),
        "trip_hours": np.random.uniform(5, 20, n),
        "distance_from_shore_km": np.random.uniform(10, 60, n),
        "crew": np.random.randint(2, 12, n),
        "wave_height_m": np.random.uniform(0.2, 3.0, n),
        "water_temp_c": np.random.uniform(22, 30, n),
        "net_length_m": np.random.uniform(300, 1500, n),
        "boat": np.random.choice(["country_boat", "motorised", "trawler"], n),
        "harbour": np.random.choice(["North Jetty", "South Jetty", "Lagoon"], n),
        "catch_kg": np.random.uniform(100, 800, n),
    })


@pytest.fixture
def synthetic_later_df(synthetic_train_df) -> pd.DataFrame:
    """Synthetic later.csv with same distribution as train."""
    np.random.seed(99)
    n = 100
    return pd.DataFrame({
        "record_id": [f"R2{i:04d}" for i in range(n)],
        "date": pd.date_range("2026-07-01", periods=n, freq="D").astype(str),
        "trip_hours": np.random.uniform(5, 20, n),
        "distance_from_shore_km": np.random.uniform(10, 60, n),
        "crew": np.random.randint(2, 12, n),
        "wave_height_m": np.random.uniform(0.2, 3.0, n),
        "water_temp_c": np.random.uniform(22, 30, n),
        "net_length_m": np.random.uniform(300, 1500, n),
        "boat": np.random.choice(["country_boat", "motorised", "trawler"], n),
        "harbour": np.random.choice(["North Jetty", "South Jetty", "Lagoon"], n),
        "catch_kg": np.random.uniform(100, 800, n),
    })


@pytest.fixture
def shifted_later_df() -> pd.DataFrame:
    """Later.csv where distance_from_shore_km is drastically shifted (simulates high PSI)."""
    np.random.seed(7)
    n = 100
    return pd.DataFrame({
        "record_id": [f"R3{i:04d}" for i in range(n)],
        "date": pd.date_range("2026-07-01", periods=n, freq="D").astype(str),
        "trip_hours": np.random.uniform(5, 20, n),
        "distance_from_shore_km": np.random.uniform(90, 150, n),  # shifted far from train
        "crew": np.random.randint(2, 12, n),
        "wave_height_m": np.random.uniform(0.2, 3.0, n),
        "water_temp_c": np.random.uniform(22, 30, n),
        "net_length_m": np.random.uniform(300, 1500, n),
        "boat": np.random.choice(["country_boat", "motorised", "trawler"], n),
        "harbour": np.random.choice(["North Jetty", "South Jetty", "Lagoon"], n),
        "catch_kg": np.random.uniform(100, 800, n),
    })


@pytest.fixture
def mock_model_perfect():
    """Mock model that returns actual catch_kg exactly (MAE = 0)."""
    model = MagicMock()
    # Will be configured per-test
    return model


# ===========================================================================
# A. PSI Tests
# ===========================================================================

class TestPSISingle:
    def test_identical_distributions_psi_near_zero(self):
        """PSI of a feature compared with itself should be ~0."""
        np.random.seed(0)
        data = pd.Series(np.random.normal(50, 10, 500))
        psi = compute_psi_single(data, data)
        assert psi < 0.01, f"Expected PSI ≈ 0 for identical distributions, got {psi}"

    def test_shifted_distribution_has_higher_psi(self):
        """PSI for a deliberately shifted feature must be much higher than PSI for a stable one.

        We construct fully controlled data so the test is deterministic:
        - ref_stable / cmp_stable share the SAME uniform range  -> low PSI
        - ref_shifted is U(10, 60); cmp_shifted is U(45, 60)    -> concentrated in upper tail -> high PSI
        """
        np.random.seed(0)
        n = 300

        # Stable feature: comparison drawn from the same range as reference
        ref_stable = pd.Series(np.random.uniform(5, 20, n))
        cmp_stable = pd.Series(np.random.uniform(5, 20, n // 2))

        # Shifted feature: comparison concentrated in the upper tail of reference range
        ref_shifted = pd.Series(np.random.uniform(10, 60, n))
        cmp_shifted = pd.Series(np.random.uniform(45, 60, n // 2))

        psi_stable = compute_psi_single(ref_stable, cmp_stable)
        psi_shifted = compute_psi_single(ref_shifted, cmp_shifted)

        assert psi_shifted > psi_stable, (
            f"Shifted PSI ({psi_shifted:.4f}) should exceed stable PSI ({psi_stable:.4f})"
        )
        assert psi_shifted > 0.5, f"Expected high PSI for shifted distribution, got {psi_shifted:.4f}"

    def test_empty_bins_handled_without_error(self):
        """PSI computation must not raise on sparse distributions with empty bins."""
        ref = pd.Series([1.0, 1.1, 1.2, 1.3, 1.4])
        cmp = pd.Series([10.0, 10.1, 10.2])  # all values outside ref range -> empty ref bins
        psi = compute_psi_single(ref, cmp, n_bins=10)
        assert isinstance(psi, float)
        assert not np.isnan(psi)
        assert not np.isinf(psi)

    def test_missing_values_handled(self):
        """PSI must handle NaN values by dropping them, without error."""
        ref = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
        cmp = pd.Series([1.5, np.nan, 3.5, 4.5])
        psi = compute_psi_single(ref, cmp)
        assert isinstance(psi, float)
        assert not np.isnan(psi)

    def test_constant_feature_returns_zero(self):
        """A constant feature cannot be binned; PSI should return 0.0."""
        ref = pd.Series([5.0] * 100)
        cmp = pd.Series([5.0] * 50)
        psi = compute_psi_single(ref, cmp)
        assert psi == 0.0

    def test_psi_uses_exactly_n_bins(self):
        """PSI histogram must use exactly the specified number of bins (n_bins + 1 edges)."""
        import unittest.mock as um
        ref = pd.Series(np.arange(100, dtype=float))
        cmp = pd.Series(np.arange(100, dtype=float))
        with um.patch("numpy.histogram", wraps=np.histogram) as mock_hist:
            compute_psi_single(ref, cmp, n_bins=10)
            assert mock_hist.call_count == 2, (
                f"Expected 2 np.histogram calls, got {mock_hist.call_count}"
            )
            for call in mock_hist.call_args_list:
                kwargs = call[1]
                args = call[0]
                bins_arg = kwargs["bins"] if "bins" in kwargs else args[1]
                assert hasattr(bins_arg, "__len__"), f"Expected bin edges array, got {bins_arg}"
                assert len(bins_arg) == 11, f"Expected 11 bin edges (10 bins), got {len(bins_arg)}"


class TestPSIAllFeatures:
    def test_returns_all_six_features(self, synthetic_train_df, synthetic_later_df):
        """compute_psi_all_features must return a dict with all 6 expected keys."""
        result = compute_psi_all_features(synthetic_train_df, synthetic_later_df)
        for feat in NUMERIC_FEATURES:
            assert feat in result, f"Feature {feat} missing from PSI output"

    def test_all_values_are_non_negative(self, synthetic_train_df, synthetic_later_df):
        """PSI values are always >= 0 by definition."""
        result = compute_psi_all_features(synthetic_train_df, synthetic_later_df)
        for feat, psi in result.items():
            assert psi is None or psi >= 0.0, f"PSI for {feat} is negative: {psi}"

    def test_missing_feature_returns_none(self, synthetic_train_df, synthetic_later_df):
        """If a feature is missing from one dataframe, PSI should be None for that feature."""
        df_missing = synthetic_later_df.drop(columns=["crew"])
        result = compute_psi_all_features(synthetic_train_df, df_missing)
        assert result["crew"] is None


# ===========================================================================
# B. Harbour-wise MAE Tests
# ===========================================================================

class TestMAEByHarbour:
    def test_returns_all_three_harbours(self, synthetic_later_df):
        """compute_mae_by_harbour must return a key for each distinct harbour."""
        # Mock model: predict slightly above actual
        model = MagicMock()
        model.predict.side_effect = lambda X: (
            pd.read_csv  # just to show side_effect works
            if False else np.random.uniform(100, 800, len(X))
        )

        from src.mp025.features import get_feature_target_split
        X, y = get_feature_target_split(synthetic_later_df, target_col="catch_kg")
        model.predict.return_value = y.values + 10  # constant +10 error

        result = compute_mae_by_harbour(model, synthetic_later_df)
        for harbour in ["North Jetty", "South Jetty", "Lagoon"]:
            assert harbour in result

    def test_perfect_model_has_zero_mae(self, synthetic_later_df):
        """A model that predicts exact values should give MAE=0 for all harbours."""
        from src.mp025.features import get_feature_target_split
        X, y = get_feature_target_split(synthetic_later_df, target_col="catch_kg")

        model = MagicMock()
        model.predict.return_value = y.values  # perfect predictions

        result = compute_mae_by_harbour(model, synthetic_later_df)
        for harbour, mae in result.items():
            assert abs(mae) < 1e-9, f"Expected MAE=0 for {harbour}, got {mae}"

    def test_worst_harbour_identified_correctly(self):
        """The harbour with the highest error should be flagged as worst-served."""
        # Create data where South Jetty has much higher error
        later_df = pd.DataFrame({
            "harbour": (["North Jetty"] * 10 + ["South Jetty"] * 10 + ["Lagoon"] * 10),
            "trip_hours": [10.0] * 30,
            "distance_from_shore_km": [30.0] * 30,
            "crew": [5] * 30,
            "wave_height_m": [1.5] * 30,
            "water_temp_c": [25.0] * 30,
            "net_length_m": [800.0] * 30,
            "boat": ["motorised"] * 30,
            "catch_kg": [500.0] * 30,
        })
        # Perfect for North and Lagoon, 200 error for South Jetty
        preds = np.array([500.0] * 10 + [700.0] * 10 + [500.0] * 10)
        model = MagicMock()
        model.predict.return_value = preds

        result = compute_mae_by_harbour(model, later_df)
        worst = max(result, key=lambda k: result[k])
        assert worst == "South Jetty"

    def test_raises_without_catch_kg_column(self, synthetic_later_df):
        """Should raise ValueError if catch_kg is missing."""
        model = MagicMock()
        df_no_target = synthetic_later_df.drop(columns=["catch_kg"])
        with pytest.raises(ValueError, match="catch_kg"):
            compute_mae_by_harbour(model, df_no_target)

    def test_raises_without_harbour_column(self, synthetic_later_df):
        """Should raise ValueError if harbour column is missing."""
        model = MagicMock()
        df_no_harbour = synthetic_later_df.drop(columns=["harbour"])
        with pytest.raises(ValueError, match="harbour"):
            compute_mae_by_harbour(model, df_no_harbour)


# ===========================================================================
# C. Chronological Split Tests
# ===========================================================================

class TestChronologicalSplit:
    def test_correct_row_counts(self, synthetic_later_df):
        """70/30 split should produce 70% train and 30% check rows."""
        n = len(synthetic_later_df)
        train, check = chronological_split_later(synthetic_later_df, train_fraction=0.70)
        expected_train = int(n * 0.70)
        assert len(train) == expected_train, f"Expected {expected_train} train rows, got {len(train)}"
        assert len(check) == n - expected_train

    def test_no_overlap_between_train_and_check(self, synthetic_later_df):
        """No record_id should appear in both train and check splits."""
        train, check = chronological_split_later(synthetic_later_df)
        overlap = set(train["record_id"]) & set(check["record_id"])
        assert len(overlap) == 0, f"Found overlapping record_ids: {overlap}"

    def test_sorted_chronologically(self, synthetic_later_df):
        """Train rows should contain earlier dates than check rows."""
        train, check = chronological_split_later(synthetic_later_df)
        max_train_date = train["date"].max()
        min_check_date = check["date"].min()
        assert max_train_date <= min_check_date, (
            f"Train contains dates ({max_train_date}) after check start ({min_check_date})"
        )

    def test_check_set_not_seen_during_train(self, synthetic_later_df):
        """The first 70% and last 30% must be strictly non-overlapping partitions."""
        train, check = chronological_split_later(synthetic_later_df)
        total = len(train) + len(check)
        assert total == len(synthetic_later_df), "Combined rows should equal original"

    def test_unsorted_input_still_produces_correct_chronological_order(self):
        """Even if input is shuffled, sort by date before splitting."""
        df = pd.DataFrame({
            "record_id": [f"R{i}" for i in range(10)],
            "date": ["2026-07-10", "2026-07-01", "2026-07-05",
                     "2026-07-08", "2026-07-03", "2026-07-02",
                     "2026-07-09", "2026-07-04", "2026-07-07", "2026-07-06"],
            "catch_kg": [500.0] * 10,
        })
        train, check = chronological_split_later(df, train_fraction=0.70)
        assert train["date"].max() <= check["date"].min()


# ===========================================================================
# D. Promotion Rule Tests
# ===========================================================================

class TestPromotionRule:
    """Test the 12% promotion decision logic using the threshold constant."""

    def _run_promotion(self, mae_v1: float, mae_v2: float) -> bool:
        """Return True if the promotion condition is satisfied."""
        return mae_v2 <= mae_v1 * PROMOTION_THRESHOLD

    def test_promotes_when_improvement_is_exactly_12_percent(self):
        """Exactly at the threshold (v2 = v1 × 0.88) should trigger promotion."""
        v1 = 100.0
        v2 = 88.0  # exactly 12% lower
        assert self._run_promotion(v1, v2) is True

    def test_promotes_when_improvement_exceeds_12_percent(self):
        """Greater than 12% improvement should also trigger promotion."""
        v1 = 100.0
        v2 = 50.0  # 50% improvement
        assert self._run_promotion(v1, v2) is True

    def test_does_not_promote_when_improvement_is_below_12_percent(self):
        """Less than 12% improvement must NOT trigger promotion."""
        v1 = 100.0
        v2 = 95.0  # only 5% improvement
        assert self._run_promotion(v1, v2) is False

    def test_does_not_promote_when_v2_is_worse(self):
        """If v2 has higher MAE than v1, must not promote."""
        v1 = 100.0
        v2 = 110.0
        assert self._run_promotion(v1, v2) is False

    def test_does_not_promote_when_mae_equal(self):
        """If v2 == v1 (0% improvement), must not promote."""
        v1 = 100.0
        v2 = 100.0
        assert self._run_promotion(v1, v2) is False

    def test_threshold_constant_is_correct(self):
        """The promotion threshold must be 0.88 (12% improvement required)."""
        assert PROMOTION_THRESHOLD == 0.88, (
            f"PROMOTION_THRESHOLD should be 0.88, got {PROMOTION_THRESHOLD}"
        )


class TestEvaluateV1V2:
    def test_returns_two_mae_values(self, synthetic_later_df):
        """evaluate_v1_v2 must return two float MAE values."""
        v1 = MagicMock()
        v2 = MagicMock()

        from src.mp025.features import get_feature_target_split
        X, y = get_feature_target_split(synthetic_later_df, target_col="catch_kg")
        y_vals = y.values

        v1.predict.return_value = y_vals + 20   # v1 has 20 error
        v2.predict.return_value = y_vals + 5    # v2 has 5 error (better)

        mae_v1, mae_v2 = evaluate_v1_v2(v1, v2, synthetic_later_df)
        assert isinstance(mae_v1, float)
        assert isinstance(mae_v2, float)
        assert mae_v2 < mae_v1, "v2 should have lower MAE in this test"

    def test_uses_same_check_set_for_both_models(self, synthetic_later_df):
        """Both models must be evaluated on identical data."""
        seen_data = []
        v1 = MagicMock()
        v2 = MagicMock()

        from src.mp025.features import get_feature_target_split
        X, y = get_feature_target_split(synthetic_later_df, target_col="catch_kg")

        v1.predict.side_effect = lambda X: (seen_data.append(("v1", len(X))), y.values)[1]
        v2.predict.side_effect = lambda X: (seen_data.append(("v2", len(X))), y.values)[1]

        evaluate_v1_v2(v1, v2, synthetic_later_df)

        assert seen_data[0][1] == seen_data[1][1], "Both models must see same number of rows"


# ===========================================================================
# E. Kubernetes YAML Tests
# ===========================================================================

REPO_ROOT = Path(__file__).parent.parent


class TestKubernetesYAML:
    @pytest.fixture
    def deployment(self):
        path = REPO_ROOT / "k8s" / "deployment.yaml"
        assert path.exists(), "k8s/deployment.yaml not found"
        with open(path) as f:
            return yaml.safe_load(f)

    @pytest.fixture
    def service(self):
        path = REPO_ROOT / "k8s" / "service.yaml"
        assert path.exists(), "k8s/service.yaml not found"
        with open(path) as f:
            return yaml.safe_load(f)

    def test_deployment_name_is_mp025_api(self, deployment):
        assert deployment["metadata"]["name"] == "mp025-api"

    def test_deployment_has_three_replicas(self, deployment):
        assert deployment["spec"]["replicas"] == 3

    def test_deployment_image_is_correct(self, deployment):
        containers = deployment["spec"]["template"]["spec"]["containers"]
        image = containers[0]["image"]
        assert image == "mp025-api:1.0", f"Expected 'mp025-api:1.0', got '{image}'"

    def test_deployment_container_port_is_8000(self, deployment):
        containers = deployment["spec"]["template"]["spec"]["containers"]
        ports = containers[0]["ports"]
        port_numbers = [p["containerPort"] for p in ports]
        assert 8000 in port_numbers

    def test_deployment_has_readiness_probe(self, deployment):
        containers = deployment["spec"]["template"]["spec"]["containers"]
        probe = containers[0].get("readinessProbe")
        assert probe is not None, "readinessProbe missing"
        assert probe["httpGet"]["path"] == "/health"
        assert probe["httpGet"]["port"] == 8000

    def test_deployment_has_liveness_probe(self, deployment):
        containers = deployment["spec"]["template"]["spec"]["containers"]
        probe = containers[0].get("livenessProbe")
        assert probe is not None, "livenessProbe missing"
        assert probe["httpGet"]["path"] == "/health"
        assert probe["httpGet"]["port"] == 8000

    def test_liveness_initial_delay_longer_than_readiness(self, deployment):
        containers = deployment["spec"]["template"]["spec"]["containers"]
        readiness_delay = containers[0]["readinessProbe"]["initialDelaySeconds"]
        liveness_delay = containers[0]["livenessProbe"]["initialDelaySeconds"]
        assert liveness_delay > readiness_delay, (
            f"Liveness delay ({liveness_delay}s) must exceed readiness delay ({readiness_delay}s)"
        )

    def test_service_name_is_mp025_svc(self, service):
        assert service["metadata"]["name"] == "mp025-svc"

    def test_service_port_is_80(self, service):
        ports = service["spec"]["ports"]
        assert any(p["port"] == 80 for p in ports)

    def test_service_target_port_is_8000(self, service):
        ports = service["spec"]["ports"]
        assert any(p["targetPort"] == 8000 for p in ports)

    def test_service_selector_matches_pod_labels(self, deployment, service):
        """The service selector must match the pod template labels exactly."""
        pod_labels = deployment["spec"]["template"]["metadata"]["labels"]
        svc_selector = service["spec"]["selector"]
        for key, val in svc_selector.items():
            assert pod_labels.get(key) == val, (
                f"Selector key '{key}={val}' not found in pod labels {pod_labels}"
            )


# ===========================================================================
# F. GitHub Actions CI YAML Tests
# ===========================================================================

class TestCIYAML:
    @pytest.fixture
    def ci_yaml(self):
        path = REPO_ROOT / ".github" / "workflows" / "ci.yml"
        assert path.exists(), ".github/workflows/ci.yml not found"
        with open(path) as f:
            return yaml.safe_load(f)

    @pytest.fixture
    def ci_raw(self):
        path = REPO_ROOT / ".github" / "workflows" / "ci.yml"
        with open(path) as f:
            return f.read()

    def test_ci_yaml_is_valid(self, ci_yaml):
        assert isinstance(ci_yaml, dict), "ci.yml must parse as a YAML mapping"

    def test_ci_starts_with_on_key_not_true(self, ci_raw):
        """The workflow must start with 'on:' not 'true:' (a common yaml serialization bug)."""
        assert "on:" in ci_raw, "ci.yml must contain 'on:' as a top-level key"
        # 'true:' would appear if 'on' was round-tripped through yaml.safe_dump
        assert "true:" not in ci_raw

    def test_ci_triggers_on_push(self, ci_yaml):
        triggers = ci_yaml.get("on", ci_yaml.get(True, {}))
        assert "push" in triggers or triggers == "push"

    def test_ci_has_jobs_section(self, ci_yaml):
        assert "jobs" in ci_yaml

    def test_ci_runs_on_ubuntu(self, ci_yaml):
        jobs = ci_yaml["jobs"]
        for job_name, job_def in jobs.items():
            assert "ubuntu" in str(job_def.get("runs-on", "")), (
                f"Job '{job_name}' should use an ubuntu runner"
            )

    def test_ci_installs_requirements(self, ci_yaml):
        """CI must install requirements.txt."""
        jobs = ci_yaml["jobs"]
        all_steps_text = str(jobs)
        assert "requirements.txt" in all_steps_text

    def test_ci_runs_pytest(self, ci_yaml):
        """CI must invoke pytest."""
        jobs = ci_yaml["jobs"]
        all_steps_text = str(jobs)
        assert "pytest" in all_steps_text

    def test_ci_sets_up_python(self, ci_yaml):
        """CI must set up Python (actions/setup-python or similar)."""
        jobs = ci_yaml["jobs"]
        all_steps_text = str(jobs)
        assert "python" in all_steps_text.lower()


# ===========================================================================
# G. findings.yaml update test
# ===========================================================================

class TestFindingsUpdate:
    def test_update_findings_preserves_existing_keys(self):
        """update_findings_monitoring must not destroy unrelated findings keys."""
        existing = {
            "group_id": "G054",
            "problem_id": "MP025",
            "data_issues": {"duplicate_rows": {"present": True, "count": 46}},
            "tracking": {"best_run_id": "abc123"},
            "monitoring": {},
        }
        psi = {f: 0.05 for f in NUMERIC_FEATURES}
        result = update_findings_monitoring(
            findings=existing,
            psi=psi,
            most_drifted="trip_hours",
            mae_by_harbour={"North Jetty": 45.0, "South Jetty": 196.0, "Lagoon": 40.0},
            worst_group="South Jetty",
            mae_v1_check=84.56,
            mae_v2_check=46.32,
            decision="promote",
        )
        # Check preserved keys
        assert result["group_id"] == "G054"
        assert result["data_issues"]["duplicate_rows"]["count"] == 46
        assert result["tracking"]["best_run_id"] == "abc123"
        # Check monitoring filled
        assert result["monitoring"]["most_drifted_feature"] == "trip_hours"
        assert result["monitoring"]["mae_v1_on_check"] == 84.56
        assert result["monitoring"]["decision"] == "promote"

    def test_all_psi_features_present_in_update(self):
        """PSI output in findings must contain all 6 feature keys."""
        existing = {"monitoring": {}}
        psi = {f: round(0.01 * i, 4) for i, f in enumerate(NUMERIC_FEATURES)}
        result = update_findings_monitoring(
            existing, psi, "net_length_m",
            {"North Jetty": 1.0, "South Jetty": 2.0, "Lagoon": 0.5},
            "South Jetty", 10.0, 8.0, "promote",
        )
        for feat in NUMERIC_FEATURES:
            assert feat in result["monitoring"]["psi"]
