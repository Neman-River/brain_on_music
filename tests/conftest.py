"""Pytest fixtures for Streamlit dashboard tests."""
import subprocess
import time
import socket
import pytest


def _wait_for_port(port: int, timeout: float = 15.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("localhost", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.3)
    return False


@pytest.fixture(scope="session")
def streamlit_server():
    proc = subprocess.Popen(
        [
            "uv", "run", "streamlit", "run",
            "dashboard/Sound_Basics.py",
            "--server.port", "8502",
            "--server.headless", "true",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if not _wait_for_port(8502):
        proc.terminate()
        raise RuntimeError("Streamlit server did not start in time")
    yield "http://localhost:8502"
    proc.terminate()
    proc.wait()


@pytest.fixture(scope="session")
def base_url(streamlit_server):
    return streamlit_server
