from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_meets_stage3_requirements():
    """Verify Dockerfile follows Python 3.12-slim, caching order, port 8000, and uvicorn CMD."""
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    lines = [line.strip() for line in dockerfile.splitlines() if line.strip() and not line.strip().startswith("#")]

    assert any(line == "FROM python:3.12-slim" for line in lines)
    assert "EXPOSE 8000" in lines

    copy_req_idx = next(i for i, line in enumerate(lines) if "COPY requirements.txt" in line)
    pip_idx = next(i for i, line in enumerate(lines) if "pip install" in line and "requirements.txt" in line)
    copy_code_idx = next(i for i, line in enumerate(lines) if line.startswith("COPY ") and "requirements.txt" not in line)

    assert copy_req_idx < pip_idx < copy_code_idx

    cmd_line = lines[-1]
    assert "uvicorn" in cmd_line
    assert "app.main:app" in cmd_line
    assert "0.0.0.0" in cmd_line
    assert "8000" in cmd_line


def test_dockerignore_excludes_non_runtime_artifacts():
    """Verify .dockerignore exists and ignores VCS, caches, raw data, and MLflow artifacts."""
    dockerignore = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8")
    entries = {line.strip() for line in dockerignore.splitlines() if line.strip() and not line.strip().startswith("#")}

    for expected in (".git/", ".venv/", "__pycache__/", ".pytest_cache/", "data/", "mlruns/", "mlartifacts/", "mlflow.db"):
        assert expected in entries


def test_requirements_all_libraries_pinned():
    """Verify every package in requirements.txt uses exact == version pinning."""
    req_text = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    reqs = [line.strip() for line in req_text.splitlines() if line.strip() and not line.strip().startswith("#")]

    assert len(reqs) >= 10
    for req in reqs:
        assert re.match(r"^[A-Za-z0-9_.-]+==[A-Za-z0-9_.*+-]+$", req), f"Unpinned requirement: {req}"
