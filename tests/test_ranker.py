"""Tests for SymbolRanker — scoring, ranking, and recommendation."""

import pytest
from datetime import datetime, timezone

from decision_engine.ranker import (
    SymbolRanker,
    RankingCriteria,
    RankedSymbol,
    RankingResult,
)
from decision_engine.models.signals import Signal, AggregatedSignal
from decision_engine.rules.base import SignalType

TS = datetime(2026, 2, 20, 14, 30, tzinfo=timezone.utc)


def _make_signal(rule_name="TestRule", confidence=0.7, signal_type=SignalType.BUY,
                 contributing_factors=None):
    return Signal(
        rule_name=rule_name,
        rule_description="desc",
        signal_type=signal_type,
        confidence=confidence,
        reasoning="test",
        contributing_factors=contributing_factors or {},
        timestamp=TS,
    )


def _make_agg(symbol, confidence=0.7, signal_type=SignalType.BUY, signals=None):
    sigs = signals or [_make_signal(confidence=confidence, signal_type=signal_type)]
    return AggregatedSignal(
        symbol=symbol,
        signal_type=signal_type,
        aggregate_confidence=confidence,
        primary_reasoning="test",
        contributing_signals=sigs,
        timestamp=TS,
        rules_triggered=len(sigs),
        rules_evaluated=5,
    )


# ---------------------------------------------------------------------------
# RankingCriteria enum
# ---------------------------------------------------------------------------

class TestRankingCriteria:
    def test_values(self):
        assert RankingCriteria.CONFIDENCE.value == "confidence"
        assert RankingCriteria.COMPOSITE.value == "composite"


# ---------------------------------------------------------------------------
# RankedSymbol
# ---------------------------------------------------------------------------

class TestRankedSymbol:
    def test_to_dict(self):
        sig = _make_agg("AAPL")
        rs = RankedSymbol(
            symbol="AAPL", rank=1, score=0.823,
            signal=sig, ranking_factors={"confidence": 0.7, "dip_depth": 0.8}
        )
        d = rs.to_dict()
        assert d["symbol"] == "AAPL"
        assert d["rank"] == 1
        assert d["score"] == 0.823
        assert d["signal_type"] == "BUY"
        assert d["ranking_factors"]["confidence"] == 0.7


# ---------------------------------------------------------------------------
# RankingResult
# ---------------------------------------------------------------------------

class TestRankingResult:
    def test_top(self):
        sigs = [
            RankedSymbol("A", 1, 0.9, _make_agg("A"), {}),
            RankedSymbol("B", 2, 0.8, _make_agg("B"), {}),
            RankedSymbol("C", 3, 0.7, _make_agg("C"), {}),
        ]
        rr = RankingResult(
            signal_type=SignalType.BUY,
            ranked_symbols=sigs,
            timestamp=TS,
            criteria_used=RankingCriteria.COMPOSITE,
        )
        assert len(rr.top(2)) == 2
        assert rr.top(2)[0].symbol == "A"

    def test_to_dict(self):
        rr = RankingResult(
            signal_type=SignalType.BUY,
            ranked_symbols=[],
            timestamp=TS,
            criteria_used=RankingCriteria.COMPOSITE,
        )
        d = rr.to_dict()
        assert d["signal_type"] == "BUY"
        assert d["criteria"] == "composite"
        assert d["total_symbols"] == 0


# ---------------------------------------------------------------------------
# SymbolRanker.rank
# ---------------------------------------------------------------------------

class TestSymbolRankerRank:
    def test_empty_signals(self):
        ranker = SymbolRanker()
        result = ranker.rank({}, SignalType.BUY)
        assert result.ranked_symbols == []

    def test_filters_by_signal_type(self):
        ranker = SymbolRanker()
        signals = {
            "AAPL": _make_agg("AAPL", signal_type=SignalType.BUY),
            "GOOG": _make_agg("GOOG", signal_type=SignalType.SELL),
        }
        result = ranker.rank(signals, SignalType.BUY)
        assert len(result.ranked_symbols) == 1
        assert result.ranked_symbols[0].symbol == "AAPL"

    def test_ranks_by_score_descending(self):
        ranker = SymbolRanker()
        signals = {
            "LOW": _make_agg("LOW", confidence=0.3),
            "HIGH": _make_agg("HIGH", confidence=0.9),
            "MID": _make_agg("MID", confidence=0.6),
        }
        result = ranker.rank(signals, SignalType.BUY)
        symbols_in_order = [r.symbol for r in result.ranked_symbols]
        assert symbols_in_order[0] == "HIGH"
        assert symbols_in_order[-1] == "LOW"

    def test_rank_numbers_sequential(self):
        ranker = SymbolRanker()
        signals = {
            "A": _make_agg("A", confidence=0.9),
            "B": _make_agg("B", confidence=0.7),
        }
        result = ranker.rank(signals, SignalType.BUY)
        assert result.ranked_symbols[0].rank == 1
        assert result.ranked_symbols[1].rank == 2

    def test_no_matching_signal_type(self):
        ranker = SymbolRanker()
        signals = {"A": _make_agg("A", signal_type=SignalType.SELL)}
        result = ranker.rank(signals, SignalType.BUY)
        assert len(result.ranked_symbols) == 0


# ---------------------------------------------------------------------------
# SymbolRanker._calculate_score
# ---------------------------------------------------------------------------

class TestCalculateScore:
    def test_default_weights(self):
        ranker = SymbolRanker()
        assert ranker.weights["dip_depth"] == 0.30
        assert ranker.weights["trend_strength"] == 0.30
        assert ranker.weights["confidence"] == 0.25
        assert ranker.weights["volatility"] == 0.15

    def test_confidence_factor(self):
        ranker = SymbolRanker()
        sig_hi = _make_agg("A", confidence=0.9)
        sig_lo = _make_agg("B", confidence=0.3)
        score_hi, _ = ranker._calculate_score(sig_hi)
        score_lo, _ = ranker._calculate_score(sig_lo)
        assert score_hi > score_lo

    def test_rsi_dip_depth_for_buy(self):
        ranker = SymbolRanker()
        # RSI 20 → dip_depth = (50-20)/30 = 1.0
        deep_sig = _make_agg("A", signals=[
            _make_signal(contributing_factors={"RSI_14": 20.0})
        ])
        # RSI 45 → dip_depth = (50-45)/30 = 0.167
        shallow_sig = _make_agg("B", signals=[
            _make_signal(contributing_factors={"RSI_14": 45.0})
        ])
        _, factors_deep = ranker._calculate_score(deep_sig)
        _, factors_shallow = ranker._calculate_score(shallow_sig)
        assert factors_deep["dip_depth"] > factors_shallow["dip_depth"]

    def test_dip_depth_defaults_to_half_without_rsi(self):
        ranker = SymbolRanker()
        sig = _make_agg("A", signals=[_make_signal(contributing_factors={})])
        _, factors = ranker._calculate_score(sig)
        assert factors["dip_depth"] == 0.5

    def test_volatility_defaults_to_half(self):
        ranker = SymbolRanker()
        sig = _make_agg("A")
        _, factors = ranker._calculate_score(sig)
        assert factors["volatility"] == 0.5

    def test_trend_quality_strong_boost(self):
        ranker = SymbolRanker()
        strong = _make_agg("A", signals=[
            _make_signal(contributing_factors={"trend_quality": "strong"})
        ])
        neutral = _make_agg("B", signals=[_make_signal(contributing_factors={})])
        _, f_strong = ranker._calculate_score(strong)
        _, f_neutral = ranker._calculate_score(neutral)
        assert f_strong["trend_strength"] > f_neutral["trend_strength"]

    def test_custom_weights(self):
        ranker = SymbolRanker(weights={
            "confidence": 1.0,
            "dip_depth": 0.0,
            "trend_strength": 0.0,
            "volatility": 0.0,
        })
        sig = _make_agg("A", confidence=0.8)
        score, _ = ranker._calculate_score(sig)
        assert score == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# SymbolRanker.get_recommendation
# ---------------------------------------------------------------------------

class TestGetRecommendation:
    def test_recommendation_string(self):
        ranker = SymbolRanker()
        signals = {"AAPL": _make_agg("AAPL", confidence=0.85)}
        rec = ranker.get_recommendation(signals)
        assert rec is not None
        assert "BUY AAPL" in rec
        assert "0.85" in rec

    def test_no_buy_signals(self):
        ranker = SymbolRanker()
        signals = {"A": _make_agg("A", signal_type=SignalType.SELL)}
        rec = ranker.get_recommendation(signals)
        assert rec is None

    def test_empty_signals(self):
        ranker = SymbolRanker()
        rec = ranker.get_recommendation({})
        assert rec is None
