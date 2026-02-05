"""Tests for config mutation helpers and API config endpoints."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from fastapi.testclient import TestClient

from colibri.config import (
    get_sources_raw,
    load_raw_config,
    save_raw_config,
    set_sources_raw,
)

# ---------------------------------------------------------------------------
# Config mutation helpers
# ---------------------------------------------------------------------------


class TestConfigMutationHelpers:
    """Tests for load_raw_config / save_raw_config / get_sources_raw / set_sources_raw."""

    def test_load_raw_config_missing_file(self, tmp_path: Path) -> None:
        with patch("colibri.config.CONFIG_FILE", tmp_path / "missing.yaml"):
            result = load_raw_config()
        assert result == {}

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        cfg = {"sources": [{"path": "/test/Books", "mode": "static", "doc_type": "book"}]}

        with (
            patch("colibri.config.CONFIG_FILE", config_file),
            patch("colibri.config.CONFIG_DIR", tmp_path),
        ):
            save_raw_config(cfg)
            loaded = load_raw_config()

        assert loaded["sources"][0]["path"] == "/test/Books"

    def test_get_sources_raw_defaults(self, tmp_path: Path) -> None:
        with patch("colibri.config.CONFIG_FILE", tmp_path / "missing.yaml"):
            sources = get_sources_raw()
        # Should return the default Books source
        assert len(sources) >= 1
        assert sources[0]["path"] is not None

    def test_set_sources_raw_creates_file(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        sources = [
            {"path": "/tmp/Notes", "mode": "incremental", "doc_type": "note"},
            {"path": "/tmp/Books", "mode": "static", "doc_type": "book"},
        ]

        with (
            patch("colibri.config.CONFIG_FILE", config_file),
            patch("colibri.config.CONFIG_DIR", tmp_path),
        ):
            set_sources_raw(sources)

        data = yaml.safe_load(config_file.read_text())
        assert len(data["sources"]) == 2
        assert data["sources"][0]["path"] == "/tmp/Notes"

    def test_set_sources_preserves_other_config(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        # Pre-populate config with other settings
        initial = {
            "sources": [{"path": "/old/path", "mode": "static"}],
            "ollama": {"base_url": "http://custom:1234"},
        }
        config_file.write_text(yaml.dump(initial))

        new_sources = [{"path": "/new/path", "mode": "incremental"}]

        with (
            patch("colibri.config.CONFIG_FILE", config_file),
            patch("colibri.config.CONFIG_DIR", tmp_path),
        ):
            set_sources_raw(new_sources)

        data = yaml.safe_load(config_file.read_text())
        assert data["sources"] == new_sources
        # Other config preserved
        assert data["ollama"]["base_url"] == "http://custom:1234"



# ---------------------------------------------------------------------------
# API config endpoints
# ---------------------------------------------------------------------------


class TestAPIConfigEndpoints:
    """Tests for the config management REST API."""

    @pytest.fixture()
    def client(self, tmp_path: Path) -> TestClient:
        """Create a test client with a temporary config file."""
        config_file = tmp_path / "config.yaml"
        books_path = str(tmp_path / "Books")
        initial = {
            "sources": [
                {"path": books_path, "mode": "static", "doc_type": "book"},
            ]
        }
        config_file.write_text(yaml.dump(initial))

        with (
            patch("colibri.config.CONFIG_FILE", config_file),
            patch("colibri.config.CONFIG_DIR", tmp_path),
        ):
            from colibri.api_server import app

            return TestClient(app, raise_server_exceptions=False)

    def test_list_sources(self, client: TestClient, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        with (
            patch("colibri.config.CONFIG_FILE", config_file),
            patch("colibri.config.CONFIG_DIR", tmp_path),
        ):
            resp = client.get("/api/config/sources")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] >= 1

    def test_add_source(self, client: TestClient, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        new_profile = {
            "path": "/tmp/Notes",
            "mode": "incremental",
            "doc_type": "note",
        }
        with (
            patch("colibri.config.CONFIG_FILE", config_file),
            patch("colibri.config.CONFIG_DIR", tmp_path),
        ):
            resp = client.post("/api/config/sources", json=new_profile)
        assert resp.status_code == 200
        assert resp.json()["path"] == "/tmp/Notes"

    def test_add_duplicate_returns_409(self, client: TestClient, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        books_path = str(tmp_path / "Books")
        with (
            patch("colibri.config.CONFIG_FILE", config_file),
            patch("colibri.config.CONFIG_DIR", tmp_path),
        ):
            resp = client.post(
                "/api/config/sources",
                json={"path": books_path, "mode": "static", "doc_type": "book"},
            )
        assert resp.status_code == 409

    def test_update_source(self, client: TestClient, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        books_path = str(tmp_path / "Books")
        updated = {"path": books_path, "mode": "incremental", "doc_type": "book"}
        with (
            patch("colibri.config.CONFIG_FILE", config_file),
            patch("colibri.config.CONFIG_DIR", tmp_path),
        ):
            resp = client.put(f"/api/config/sources/{books_path}", json=updated)
        assert resp.status_code == 200
        assert resp.json()["mode"] == "incremental"

    def test_update_missing_returns_404(self, client: TestClient, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        with (
            patch("colibri.config.CONFIG_FILE", config_file),
            patch("colibri.config.CONFIG_DIR", tmp_path),
        ):
            resp = client.put(
                "/api/config/sources//tmp/Missing",
                json={"path": "/tmp/Missing", "mode": "static", "doc_type": "x"},
            )
        assert resp.status_code == 404

    def test_delete_source(self, client: TestClient, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        books_path = str(tmp_path / "Books")
        with (
            patch("colibri.config.CONFIG_FILE", config_file),
            patch("colibri.config.CONFIG_DIR", tmp_path),
        ):
            resp = client.delete(f"/api/config/sources/{books_path}")
        assert resp.status_code == 200

    def test_delete_missing_returns_404(self, client: TestClient, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        with (
            patch("colibri.config.CONFIG_FILE", config_file),
            patch("colibri.config.CONFIG_DIR", tmp_path),
        ):
            resp = client.delete("/api/config/sources//tmp/Missing")
        assert resp.status_code == 404

    def test_config_ui_returns_html(self, client: TestClient) -> None:
        resp = client.get("/config")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "CoLibri" in resp.text
