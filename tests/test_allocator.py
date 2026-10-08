"""
Tests for the Portfolio Allocator and the LLM client setup.
These run without the dataset and without an API key.
Run: python -m pytest tests -q
"""
import numpy as np
import pandas as pd
import pytest

from src.agents.portfolio_strategy import (
    PortfolioAllocator,
    PortfolioStrategyAgent,
    _workspace_headers,
)


def _view():
    return pd.DataFrame({
        "campaign":          [1, 2, 3, 4],
        "cost":              [10.0, 10.0, 10.0, 10.0],
        "last_day_cost":     [2.0, 2.0, 2.0, 2.0],
        "known_conversions": [5, 1, 0, 3],
        "cpa":               [2.0, 10.0, np.nan, 3.33],
        "ctr":               [0.1, 0.1, 0.1, 0.1],
        "clicks":            [10, 10, 10, 10],
        "impressions":       [100, 100, 100, 100],
    })


STRATEGY = {"strategy": "balanced", "protect_campaigns": [], "exclude_campaigns": [],
            "focus_on_conversions": True}


@pytest.mark.parametrize("budget", [1.0, 5.0, 100.0])
def test_budget_never_exceeded(budget):
    alloc = PortfolioAllocator().allocate(_view(), budget, STRATEGY, seed=0)
    assert alloc.sum() <= budget + 1e-9


def test_capacity_respected():
    alloc = PortfolioAllocator().allocate(_view(), 100.0, STRATEGY, seed=0)
    assert (alloc <= 2.0 + 1e-9).all()


def test_excluded_campaign_gets_nothing():
    strategy = dict(STRATEGY, exclude_campaigns=[1])
    alloc = PortfolioAllocator().allocate(_view(), 100.0, strategy, seed=0)
    assert alloc[1] == 0


def test_protected_campaign_funded_first():
    strategy = dict(STRATEGY, protect_campaigns=[3])
    alloc = PortfolioAllocator().allocate(_view(), 2.0, strategy, seed=0)
    assert alloc[3] == pytest.approx(2.0)


def test_allocation_is_reproducible():
    a = PortfolioAllocator().allocate(_view(), 3.0, STRATEGY, seed=7)
    b = PortfolioAllocator().allocate(_view(), 3.0, STRATEGY, seed=7)
    pd.testing.assert_series_equal(a, b)


def test_baselines_stay_within_budget():
    for name, alloc in PortfolioAllocator().get_baselines(_view(), 3.0, seed=0).items():
        assert alloc.sum() <= 3.0 + 1e-9, name


def test_workspace_header_optional(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID", raising=False)
    assert _workspace_headers() is None
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "ws_123")
    assert _workspace_headers() == {"anthropic-workspace-id": "ws_123"}


def test_agent_builds_without_workspace_id(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    agent = PortfolioStrategyAgent()
    assert "anthropic-workspace-id" not in agent.client.default_headers
