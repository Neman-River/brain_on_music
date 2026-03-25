"""Smoke tests — each page loads without errors."""
import re
from playwright.sync_api import Page, expect

PAGES = [
    ("/Explore_Track",       "Explore Track"),
    ("/Fourier_Transform",   "Fourier Transform"),
    ("/Spectrograms",        "Spectrograms"),
    ("/Audio_Features",      "Audio Features"),
    ("/Genre_Comparison",    "Genre Comparison"),
]


def _goto_page(page: Page, base_url: str, path: str):
    page.goto(f"{base_url}{path}")
    # Wait for Streamlit to finish loading (spinner disappears)
    page.wait_for_selector('[data-testid="stApp"]', timeout=20000)


def test_explore_track_loads(page: Page, base_url: str):
    _goto_page(page, base_url, "/Explore_Track")
    expect(page.get_by_role("heading", name=re.compile("Explore", re.I))).to_be_visible(timeout=15000)


def test_fourier_transform_loads(page: Page, base_url: str):
    _goto_page(page, base_url, "/Fourier_Transform")
    expect(page.get_by_role("heading", name=re.compile("Fourier|STFT|Transform", re.I))).to_be_visible(timeout=15000)


def test_spectrograms_loads(page: Page, base_url: str):
    _goto_page(page, base_url, "/Spectrograms")
    expect(page.get_by_role("heading", name=re.compile("Spectrogram", re.I))).to_be_visible(timeout=15000)


def test_audio_features_loads(page: Page, base_url: str):
    _goto_page(page, base_url, "/Audio_Features")
    expect(page.get_by_role("heading", name=re.compile("Audio|Feature", re.I))).to_be_visible(timeout=15000)


def test_genre_comparison_loads(page: Page, base_url: str):
    _goto_page(page, base_url, "/Genre_Comparison")
    expect(page.get_by_role("heading", name=re.compile("Genre|Comparison", re.I))).to_be_visible(timeout=15000)


def test_no_streamlit_exception_on_home(page: Page, base_url: str):
    """Streamlit renders an error box when an exception is raised."""
    page.goto(base_url)
    page.wait_for_selector('[data-testid="stApp"]', timeout=20000)
    error_box = page.locator('[data-testid="stException"]')
    expect(error_box).to_have_count(0)
