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
from .enhanced_rules import (
    EnhancedBuyDipRule,
    MomentumReversalRule,
    TrendContinuationRule,
)
from .mining_rules import (
    CommodityBreakoutRule,
    MinerMetalRatioRule,
    DollarWeaknessRule,
    SeasonalityRule,
    VolumeBreakoutRule,
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

    # Enhanced Rules (Quantitative Improvements)
    "enhanced_buy_dip": EnhancedBuyDipRule,
    "momentum_reversal": MomentumReversalRule,
    "trend_continuation": TrendContinuationRule,

    # Mining Stock Rules
    "commodity_breakout": CommodityBreakoutRule,
    "miner_metal_ratio": MinerMetalRatioRule,
    "dollar_weakness": DollarWeaknessRule,
    "seasonality": SeasonalityRule,
    "volume_breakout": VolumeBreakoutRule,
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

            # Enhanced Buy Dip parameters
            if "rsi_oversold" in config:
                params["rsi_oversold"] = config["rsi_oversold"]
            if "rsi_extreme" in config:
                params["rsi_extreme"] = config["rsi_extreme"]
            if "min_trend_spread" in config:
                params["min_trend_spread"] = config["min_trend_spread"]
            if "require_volume_confirm" in config:
                params["require_volume_confirm"] = config["require_volume_confirm"]

            # Momentum Reversal parameters
            if "rsi_recovery_min" in config:
                params["rsi_recovery_min"] = config["rsi_recovery_min"]
            if "rsi_recovery_max" in config:
                params["rsi_recovery_max"] = config["rsi_recovery_max"]

            # Trend Continuation parameters
            if "pullback_tolerance_pct" in config:
                params["pullback_tolerance_pct"] = config["pullback_tolerance_pct"]

            # Mining Rules parameters
            if "breakout_threshold_pct" in config:
                params["breakout_threshold_pct"] = config["breakout_threshold_pct"]
            if "min_trend_strength" in config:
                params["min_trend_strength"] = config["min_trend_strength"]
            if "support_tolerance_pct" in config:
                params["support_tolerance_pct"] = config["support_tolerance_pct"]
            if "require_macd_positive" in config:
                params["require_macd_positive"] = config["require_macd_positive"]
            if "strong_month_boost" in config:
                params["strong_month_boost"] = config["strong_month_boost"]
            if "weak_month_penalty" in config:
                params["weak_month_penalty"] = config["weak_month_penalty"]
            if "min_volume_ratio" in config:
                params["min_volume_ratio"] = config["min_volume_ratio"]

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

    @staticmethod
    def load_symbol_rules(config: dict, symbol: str) -> tuple[List[Rule], Dict[str, float], dict]:
        """
        Load rules for a specific symbol, using overrides if configured.

        Args:
            config: Full configuration dict
            symbol: The symbol to load rules for

        Returns:
            Tuple of (list of rules, dict of rule weights, exit strategy dict)
        """
        symbol_overrides = config.get("symbol_overrides", {})
        global_rules_config = config.get("rules", {})
        default_exit = config.get("exit_strategy", {"profit_target": 0.07, "stop_loss": 0.05})

        # Check if this symbol has overrides
        if symbol in symbol_overrides:
            override = symbol_overrides[symbol]
            rule_names = override.get("rules", [])
            exit_strategy = override.get("exit_strategy", default_exit)

            rules = []
            weights = {}

            for rule_name in rule_names:
                # Get params from global config if available
                rule_config = global_rules_config.get(rule_name, {})

                rule = RuleRegistry.create_rule(rule_name, rule_config)
                if rule:
                    rules.append(rule)
                    weights[rule.name] = rule_config.get("weight", 1.0)
                    logger.debug(f"Loaded override rule for {symbol}: {rule.name}")

            logger.info(f"Symbol {symbol}: using {len(rules)} override rules")
            return rules, weights, exit_strategy

        # No overrides - return None to signal use of default rules
        return None, None, default_exit

    @staticmethod
    def get_symbol_overrides(config: dict) -> Dict[str, List[str]]:
        """
        Get all symbol overrides from config.

        Returns:
            Dict mapping symbol -> list of rule names
        """
        symbol_overrides = config.get("symbol_overrides", {})
        return {
            symbol: override.get("rules", [])
            for symbol, override in symbol_overrides.items()
        }
