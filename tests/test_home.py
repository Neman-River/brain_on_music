"""Tests for the home page (Sound Basics)."""
from playwright.sync_api import Page, expect


def test_title_visible(page: Page, base_url: str):
    page.goto(base_url)
    expect(page.get_by_role("heading", name="Understand The Sound")).to_be_visible(timeout=15000)


def test_frequency_slider_exists(page: Page, base_url: str):
    page.goto(base_url)
    # Streamlit sliders render with a <input type="range"> inside a labelled block
    slider = page.locator('input[type="range"]').first
    expect(slider).to_be_visible(timeout=15000)


def test_features_table_visible(page: Page, base_url: str):
    page.goto(base_url)
    # The dataframe heading/caption contains "Total:"
    expect(page.get_by_text("Total:")).to_be_visible(timeout=15000)


def test_pipeline_diagram_visible(page: Page, base_url: str):
    page.goto(base_url)
    expect(page.get_by_text("From raw audio to model input")).to_be_visible(timeout=15000)
