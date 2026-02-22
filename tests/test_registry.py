"""Tests for RuleRegistry — rule creation, config loading, validation, symbol overrides."""

import math
import pytest

from decision_engine.rules.registry import RuleRegistry, RULE_REGISTRY
from decision_engine.rules.base import Rule


# ---------------------------------------------------------------------------
# RULE_REGISTRY mapping
# ---------------------------------------------------------------------------

class TestRuleRegistryMapping:
    def test_all_entries_are_rule_subclasses(self):
        for name, cls in RULE_REGISTRY.items():
            assert issubclass(cls, Rule), f"{name} -> {cls} is not a Rule"

    def test_expected_count(self):
        # 3 RSI + 3 MACD + 6 Trend + 4 Composite + 3 Enhanced + 5 Mining = 24
        assert len(RULE_REGISTRY) == 24

    def test_known_keys_present(self):
        expected = [
            "rsi_oversold", "rsi_overbought", "rsi_approaching_oversold",
            "macd_bullish_crossover", "macd_bearish_crossover", "macd_momentum",
            "weekly_uptrend", "monthly_uptrend", "trend_alignment",
            "trend_break_warning", "golden_cross", "death_cross",
            "buy_dip_in_uptrend", "strong_buy_signal", "rsi_macd_confluence",
            "dip_recovery",
            "enhanced_buy_dip", "momentum_reversal", "trend_continuation",
            "commodity_breakout", "miner_metal_ratio", "dollar_weakness",
            "seasonality", "volume_breakout",
        ]
        for key in expected:
            assert key in RULE_REGISTRY, f"Missing registry key: {key}"


# ---------------------------------------------------------------------------
# get_available_rules / get_rule_class
# ---------------------------------------------------------------------------

class TestRuleRegistryLookup:
    def test_get_available_rules(self):
        names = RuleRegistry.get_available_rules()
        assert isinstance(names, list)
        assert "rsi_oversold" in names

    def test_get_rule_class_known(self):
        cls = RuleRegistry.get_rule_class("rsi_oversold")
        assert cls is not None
        assert issubclass(cls, Rule)

    def test_get_rule_class_unknown(self):
        cls = RuleRegistry.get_rule_class("does_not_exist")
        assert cls is None


# ---------------------------------------------------------------------------
# _validate_params
# ---------------------------------------------------------------------------

class TestValidateParams:
    def test_valid_params(self):
        err = RuleRegistry._validate_params("test", {"threshold": 30})
        assert err is None

    def test_rsi_key_out_of_range(self):
        err = RuleRegistry._validate_params("test", {"threshold": 150})
        assert err is not None
        assert "out of RSI range" in err

    def test_rsi_key_negative(self):
        err = RuleRegistry._validate_params("test", {"rsi_oversold": -5})
        assert err is not None

    def test_pct_key_negative(self):
        err = RuleRegistry._validate_params("test", {"pullback_tolerance_pct": -1})
        assert err is not None

    def test_pct_key_too_large(self):
        err = RuleRegistry._validate_params("test", {"breakout_threshold_pct": 200})
        assert err is not None

    def test_weight_key_negative(self):
        err = RuleRegistry._validate_params("test", {"min_volume_ratio": -0.5})
        assert err is not None

    def test_nan_rejected(self):
        err = RuleRegistry._validate_params("test", {"min_volume_ratio": float("nan")})
        assert err is not None

    def test_inf_rejected(self):
        err = RuleRegistry._validate_params("test", {"min_volume_ratio": float("inf")})
        assert err is not None

    def test_bool_skipped(self):
        err = RuleRegistry._validate_params("test", {"require_volume_confirm": True})
        assert err is None

    def test_string_skipped(self):
        err = RuleRegistry._validate_params("test", {"name": "foo"})
        assert err is None


# ---------------------------------------------------------------------------
# create_rule
# ---------------------------------------------------------------------------

class TestCreateRule:
    def test_create_known_rule_no_params(self):
        rule = RuleRegistry.create_rule("rsi_oversold", {})
        assert rule is not None
        assert rule.name == "RSI Oversold"

    def test_create_with_params(self):
        rule = RuleRegistry.create_rule("rsi_oversold", {"threshold": 25})
        assert rule is not None
        assert rule.threshold == 25

    def test_create_unknown_returns_none(self):
        rule = RuleRegistry.create_rule("nonexistent_rule", {})
        assert rule is None

    def test_create_with_invalid_params_returns_none(self):
        rule = RuleRegistry.create_rule("rsi_oversold", {"threshold": 200})
        assert rule is None

    def test_create_enhanced_rule(self):
        rule = RuleRegistry.create_rule("enhanced_buy_dip", {
            "rsi_oversold": 40,
            "rsi_extreme": 30,
            "min_trend_spread": 2.0,
            "require_volume_confirm": True,
        })
        assert rule is not None
        assert rule.rsi_oversold == 40

    def test_create_mining_rule(self):
        rule = RuleRegistry.create_rule("commodity_breakout", {
            "breakout_threshold_pct": 3.0,
            "min_trend_strength": 1.5,
        })
        assert rule is not None
        assert rule.breakout_threshold_pct == 3.0


# ---------------------------------------------------------------------------
# load_rules_from_config
# ---------------------------------------------------------------------------

class TestLoadRulesFromConfig:
    def test_loads_enabled_rules(self):
        config = {
            "rules": {
                "rsi_oversold": {"enabled": True, "threshold": 30, "weight": 2.0},
                "macd_bullish_crossover": {"enabled": True, "weight": 1.5},
            }
        }
        rules, weights = RuleRegistry.load_rules_from_config(config)
        assert len(rules) == 2
        assert "RSI Oversold" in weights
        assert weights["RSI Oversold"] == 2.0

    def test_skips_disabled_rules(self):
        config = {
            "rules": {
                "rsi_oversold": {"enabled": True},
                "macd_momentum": {"enabled": False},
            }
        }
        rules, weights = RuleRegistry.load_rules_from_config(config)
        assert len(rules) == 1
        assert rules[0].name == "RSI Oversold"

    def test_default_weight_is_1(self):
        config = {"rules": {"rsi_oversold": {}}}
        rules, weights = RuleRegistry.load_rules_from_config(config)
        assert weights.get("RSI Oversold") == 1.0

    def test_invalid_weight_defaults_to_1(self):
        config = {"rules": {"rsi_oversold": {"weight": -5}}}
        rules, weights = RuleRegistry.load_rules_from_config(config)
        assert weights.get("RSI Oversold") == 1.0

    def test_string_weight_defaults_to_1(self):
        config = {"rules": {"rsi_oversold": {"weight": "high"}}}
        rules, weights = RuleRegistry.load_rules_from_config(config)
        assert weights.get("RSI Oversold") == 1.0

    def test_empty_config(self):
        rules, weights = RuleRegistry.load_rules_from_config({})
        assert rules == []
        assert weights == {}

    def test_unknown_rule_skipped(self):
        config = {"rules": {"fake_rule": {"enabled": True}}}
        rules, weights = RuleRegistry.load_rules_from_config(config)
        assert rules == []


# ---------------------------------------------------------------------------
# load_symbol_rules
# ---------------------------------------------------------------------------

class TestLoadSymbolRules:
    def test_no_override_returns_none(self):
        config = {"rules": {"rsi_oversold": {}}}
        rules, weights, exit_strategy = RuleRegistry.load_symbol_rules(config, "AAPL")
        assert rules is None
        assert weights is None

    def test_with_override(self):
        config = {
            "rules": {"rsi_oversold": {"threshold": 30}},
            "symbol_overrides": {
                "AAPL": {
                    "rules": ["rsi_oversold"],
                    "exit_strategy": {"profit_target": 0.10, "stop_loss": 0.04},
                }
            },
        }
        rules, weights, exit_strategy = RuleRegistry.load_symbol_rules(config, "AAPL")
        assert len(rules) == 1
        assert rules[0].name == "RSI Oversold"
        assert exit_strategy["profit_target"] == 0.10

    def test_default_exit_strategy(self):
        config = {
            "rules": {},
            "symbol_overrides": {"AAPL": {"rules": []}},
        }
        _, _, exit_strategy = RuleRegistry.load_symbol_rules(config, "AAPL")
        assert exit_strategy == {"profit_target": 0.07, "stop_loss": 0.05}


# ---------------------------------------------------------------------------
# get_symbol_overrides
# ---------------------------------------------------------------------------

class TestGetSymbolOverrides:
    def test_returns_mapping(self):
        config = {
            "symbol_overrides": {
                "AAPL": {"rules": ["rsi_oversold", "macd_momentum"]},
                "GOOG": {"rules": ["trend_alignment"]},
            }
        }
        result = RuleRegistry.get_symbol_overrides(config)
        assert result["AAPL"] == ["rsi_oversold", "macd_momentum"]
        assert result["GOOG"] == ["trend_alignment"]

    def test_empty_config(self):
        result = RuleRegistry.get_symbol_overrides({})
        assert result == {}


# ---------------------------------------------------------------------------
# describe_rules
# ---------------------------------------------------------------------------

class TestDescribeRules:
    def test_returns_string(self):
        desc = RuleRegistry.describe_rules()
        assert isinstance(desc, str)
        assert "Available Rules" in desc
        assert "rsi_oversold" in desc
