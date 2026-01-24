"""
Rule registry for discovering and instantiating rules.
"""

import logging
from typing import Dict, List, Optional, Type

from .base import Rule
from .rsi_rules import RSIOversoldRule, RSIOverboughtRule, RSIApproachingOversoldRule
from .macd_rules import MACDBullishCrossoverRule, MACDBearishCrossoverRule, MACDMomentumRule
from .trend_rules import (
    WeeklyUptrendRule,
    MonthlyUptrendRule,
    FullTrendAlignmentRule,
    TrendBreakWarningRule,
    GoldenCrossRule,
    DeathCrossRule,
)
from .composite_rules import (
    BuyDipInUptrendRule,
    StrongBuySignalRule,
    RSIAndMACDConfluenceRule,
    TrendDipRecoveryRule,
)

logger = logging.getLogger(__name__)


# Registry mapping config names to rule classes
RULE_REGISTRY: Dict[str, Type[Rule]] = {
    # RSI Rules
    "rsi_oversold": RSIOversoldRule,
    "rsi_overbought": RSIOverboughtRule,
    "rsi_approaching_oversold": RSIApproachingOversoldRule,

    # MACD Rules
    "macd_bullish_crossover": MACDBullishCrossoverRule,
    "macd_bearish_crossover": MACDBearishCrossoverRule,
    "macd_momentum": MACDMomentumRule,

    # Trend Rules
    "weekly_uptrend": WeeklyUptrendRule,
    "monthly_uptrend": MonthlyUptrendRule,
    "trend_alignment": FullTrendAlignmentRule,
    "trend_break_warning": TrendBreakWarningRule,
    "golden_cross": GoldenCrossRule,
    "death_cross": DeathCrossRule,

    # Composite Rules (YOUR STRATEGY)
    "buy_dip_in_uptrend": BuyDipInUptrendRule,
    "strong_buy_signal": StrongBuySignalRule,
    "rsi_macd_confluence": RSIAndMACDConfluenceRule,
    "dip_recovery": TrendDipRecoveryRule,
}


class RuleRegistry:
    """
    Registry for managing trading rules.

    Provides methods to:
    - Create rules from configuration
    - Load all enabled rules
    - Get rule by name
    """

    @staticmethod
    def get_available_rules() -> List[str]:
        """Get list of all available rule names."""
        return list(RULE_REGISTRY.keys())

    @staticmethod
    def get_rule_class(rule_name: str) -> Optional[Type[Rule]]:
        """Get the rule class for a given name."""
        return RULE_REGISTRY.get(rule_name)

    @staticmethod
    def create_rule(rule_name: str, config: dict) -> Optional[Rule]:
        """
        Create a rule instance from configuration.

        Args:
            rule_name: Name of the rule (e.g., 'rsi_oversold')
            config: Configuration dict with rule parameters

        Returns:
            Rule instance or None if not found
        """
        rule_class = RULE_REGISTRY.get(rule_name)
        if rule_class is None:
            logger.warning(f"Unknown rule: {rule_name}")
            return None

        try:
            # Extract parameters that the rule accepts
            # Common parameters for different rule types
            params = {}

            # RSI rules
            if "threshold" in config:
                params["threshold"] = config["threshold"]
            if "extreme_threshold" in config:
                params["extreme_threshold"] = config["extreme_threshold"]

            # Composite rules
            if "rsi_threshold" in config:
                params["rsi_threshold"] = config["rsi_threshold"]

            # MACD rules
            if "histogram_threshold" in config:
                params["histogram_threshold"] = config["histogram_threshold"]

            # Create the rule with extracted params
            rule = rule_class(**params) if params else rule_class()
            return rule

        except Exception as e:
            logger.error(f"Failed to create rule {rule_name}: {e}")
            return None

    @staticmethod
    def load_rules_from_config(config: dict) -> tuple[List[Rule], Dict[str, float]]:
        """
        Load all enabled rules from configuration.

        Args:
            config: Full configuration dict with 'rules' section

        Returns:
            Tuple of (list of rules, dict of rule weights)
        """
        rules = []
        weights = {}

        rules_config = config.get("rules", {})

        for rule_name, rule_config in rules_config.items():
            # Skip disabled rules
            if not rule_config.get("enabled", True):
                logger.info(f"Skipping disabled rule: {rule_name}")
                continue

            rule = RuleRegistry.create_rule(rule_name, rule_config)
            if rule:
                rules.append(rule)
                weights[rule.name] = rule_config.get("weight", 1.0)
                logger.info(f"Loaded rule: {rule.name} (weight: {weights[rule.name]})")

        logger.info(f"Loaded {len(rules)} rules")
        return rules, weights

    @staticmethod
    def describe_rules() -> str:
        """Get a human-readable description of all available rules."""
        lines = ["Available Rules:", "=" * 40]

        for name, rule_class in RULE_REGISTRY.items():
            # Create temporary instance to get description
            try:
                rule = rule_class()
                lines.append(f"\n{name}:")
                lines.append(f"  Name: {rule.name}")
                lines.append(f"  Description: {rule.description}")
                lines.append(f"  Required: {', '.join(rule.required_indicators)}")
            except Exception:
                lines.append(f"\n{name}: (unable to instantiate)")

        return "\n".join(lines)
