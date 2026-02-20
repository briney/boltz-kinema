"""Tests for the Click CLI interface."""

from __future__ import annotations

import subprocess
import sys

from click.testing import CliRunner

from kinematic.cli import cli


def test_cli_help():
    """Top-level --help shows group description and subcommands."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Kinematic" in result.output
    assert "train" in result.output
    assert "download-training-data" in result.output
    assert "precompute-trunk-embeddings" in result.output
    assert "preprocess" in result.output


def test_cli_version():
    """--version prints the package version."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "version" in result.output.lower()


def test_train_help():
    """``kinematic train --help`` shows train options."""
    runner = CliRunner()
    result = runner.invoke(cli, ["train", "--help"])
    assert result.exit_code == 0
    assert "--config" in result.output
    assert "--base-config" in result.output
    assert "--config-dir" in result.output


def test_download_help():
    """``kinematic download-training-data --help`` shows download options."""
    runner = CliRunner()
    result = runner.invoke(cli, ["download-training-data", "--help"])
    assert result.exit_code == 0
    assert "--datasets" in result.output
    assert "--output-dir" in result.output


def test_precompute_trunk_embeddings_help():
    """``kinematic precompute-trunk-embeddings --help`` shows all options."""
    runner = CliRunner()
    result = runner.invoke(cli, ["precompute-trunk-embeddings", "--help"])
    assert result.exit_code == 0
    assert "--manifest" in result.output
    assert "--checkpoint" in result.output
    assert "--output-dir" in result.output
    assert "--device" in result.output
    assert "--recycling-steps" in result.output


def test_preprocess_help():
    """``kinematic preprocess --help`` lists all dataset subcommands."""
    runner = CliRunner()
    result = runner.invoke(cli, ["preprocess", "--help"])
    assert result.exit_code == 0
    assert "atlas" in result.output
    assert "cath" in result.output
    assert "dd13m" in result.output
    assert "mdposit" in result.output
    assert "megasim" in result.output
    assert "misato" in result.output
    assert "octapeptides" in result.output


def test_preprocess_atlas_help():
    """``kinematic preprocess atlas --help`` shows shared options."""
    runner = CliRunner()
    result = runner.invoke(cli, ["preprocess", "atlas", "--help"])
    assert result.exit_code == 0
    assert "--input-dir" in result.output
    assert "--output-dir" in result.output
    assert "--ref-dir" in result.output
    assert "--manifest-out" in result.output


def test_preprocess_megasim_help():
    """``kinematic preprocess megasim --help`` shows mutant options."""
    runner = CliRunner()
    result = runner.invoke(cli, ["preprocess", "megasim", "--help"])
    assert result.exit_code == 0
    assert "--input-dir" in result.output
    assert "--subsample-mutants" in result.output
    assert "--all-mutants" in result.output


def test_preprocess_misato_help():
    """``kinematic preprocess misato --help`` shows coords-unit option."""
    runner = CliRunner()
    result = runner.invoke(cli, ["preprocess", "misato", "--help"])
    assert result.exit_code == 0
    assert "--input-dir" in result.output
    assert "--coords-unit" in result.output


def test_python_m_kinematic_help():
    """``python -m kinematic --help`` works via __main__.py."""
    result = subprocess.run(
        [sys.executable, "-m", "kinematic", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "Kinematic" in result.stdout
    assert "train" in result.stdout
