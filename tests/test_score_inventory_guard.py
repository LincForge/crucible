"""Honest-inventory guard on crucible_score (reactor 0f6b5320).

A bare ``count: 0`` / ``error: "No scores found"`` reads identically for
"no scores match this project" and "the score store has never existed on
this host" — mirrors the 2026-08-06 baseline-list audit
(tests/test_baseline_inventory_guard.py). A zero-score response must now say
explicitly whether the score store root exists and where it was looked for.
"""

from __future__ import annotations

from unittest.mock import patch

from linc_crucible.cli import _echo_human_inventory_message
from linc_crucible.core.inventory import empty_inventory
from linc_crucible.core.scorer import QualityScore, QualityScorer


class TestEmptyInventoryUnit:
    def test_message_when_store_exists(self, tmp_path):
        result = empty_inventory(tmp_path, "project 'SniperPulse'", "scores")

        assert result == {
            "empty": True,
            "store_root": str(tmp_path),
            "store_exists": True,
            "message": (
                f"No scores stored for project 'SniperPulse' (store {tmp_path} exists)."
            ),
        }

    def test_message_when_store_absent(self, tmp_path):
        missing = tmp_path / "never-created"

        result = empty_inventory(missing, "project 'SniperPulse'", "scores")

        assert result["store_exists"] is False
        assert result["store_root"] == str(missing)
        assert "does not exist on this host" in result["message"]


class TestCrucibleScoreHistoryInventory:
    @patch("linc_crucible.server_context._ensure_initialized")
    def test_history_absent_store_says_so(self, mock_init, tmp_path):
        from linc_crucible.assay_tools import crucible_score

        scorer = QualityScorer(scores_dir=tmp_path / "never-created")
        with patch("linc_crucible.server_context._scorer", scorer):
            result = crucible_score(project="SniperPulse", history=True)

        assert result["project"] == "SniperPulse"
        assert result["count"] == 0
        assert result["scores"] == []
        assert result["empty"] is True
        assert result["store_exists"] is False
        assert result["store_root"] == str(tmp_path / "never-created")
        assert "does not exist on this host" in result["message"]

    @patch("linc_crucible.server_context._ensure_initialized")
    def test_history_existing_but_empty_store_says_so(self, mock_init, tmp_path):
        from linc_crucible.assay_tools import crucible_score

        store = tmp_path / "scores"
        store.mkdir()
        scorer = QualityScorer(scores_dir=store)
        with patch("linc_crucible.server_context._scorer", scorer):
            result = crucible_score(project="SniperPulse", history=True)

        assert result["count"] == 0
        assert result["empty"] is True
        assert result["store_exists"] is True
        assert "store" in result["message"]
        assert "exists" in result["message"]

    @patch("linc_crucible.server_context._ensure_initialized")
    def test_history_with_scores_has_no_empty_flag(self, mock_init, tmp_path):
        from linc_crucible.assay_tools import crucible_score

        store = tmp_path / "scores"
        scorer = QualityScorer(scores_dir=store)
        scorer.save(
            QualityScore(
                project="SniperPulse",
                timestamp="2026-02-20T00-00-00Z",
                mode="quick",
                composite=90.0,
                ship_gate=True,
            )
        )
        with patch("linc_crucible.server_context._scorer", scorer):
            result = crucible_score(project="SniperPulse", history=True)

        assert result["count"] == 1
        assert "empty" not in result
        assert "store_exists" not in result
        assert "store_root" not in result
        assert "message" not in result


class TestCrucibleScoreLatestInventory:
    @patch("linc_crucible.server_context._ensure_initialized")
    def test_latest_absent_store_keeps_error_and_says_so(self, mock_init, tmp_path):
        from linc_crucible.assay_tools import crucible_score

        scorer = QualityScorer(scores_dir=tmp_path / "never-created")
        with patch("linc_crucible.server_context._scorer", scorer):
            result = crucible_score(project="SniperPulse")

        assert "No scores found" in result["error"]
        assert result["store_exists"] is False
        assert result["store_root"] == str(tmp_path / "never-created")

    @patch("linc_crucible.server_context._ensure_initialized")
    def test_latest_with_score_has_no_empty_flag(self, mock_init, tmp_path):
        from linc_crucible.assay_tools import crucible_score

        store = tmp_path / "scores"
        scorer = QualityScorer(scores_dir=store)
        scorer.save(
            QualityScore(
                project="SniperPulse",
                timestamp="2026-02-20T00-00-00Z",
                mode="quick",
                composite=90.0,
                ship_gate=True,
            )
        )
        with patch("linc_crucible.server_context._scorer", scorer):
            result = crucible_score(project="SniperPulse")

        assert "empty" not in result
        assert "store_exists" not in result
        assert "error" not in result


class TestEchoHumanInventoryMessage:
    def test_prints_message_when_error_and_message_present(self, capsys):
        _echo_human_inventory_message({"error": "No scores found.", "message": "why"})

        assert "why" in capsys.readouterr().err

    def test_silent_without_error(self, capsys):
        _echo_human_inventory_message({"message": "why"})

        assert capsys.readouterr().err == ""

    def test_silent_without_message(self, capsys):
        _echo_human_inventory_message({"error": "No scores found."})

        assert capsys.readouterr().err == ""
