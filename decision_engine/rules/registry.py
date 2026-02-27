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
    def _validate_params(rule_name: str, params: dict) -> Optional[str]:
        """
        Validate rule parameters are within sane ranges.

        Returns an error message if invalid, None if OK.
        """
        for key, value in params.items():
            # Skip non-numeric parameters
            if isinstance(value, bool):
                continue
            if not isinstance(value, (int, float)):
                continue

            # RSI values must be 0-100
            rsi_keys = {
                "threshold", "extreme_threshold", "rsi_threshold",
                "rsi_oversold", "rsi_extreme", "rsi_recovery_min",
                "rsi_recovery_max",
            }
            if key in rsi_keys:
                if not (0 <= value <= 100):
                    return f"{key}={value} out of RSI range [0, 100]"

            # Percentage values must be non-negative and reasonable
            pct_keys = {
                "pullback_tolerance_pct", "breakout_threshold_pct",
                "support_tolerance_pct", "min_trend_spread",
            }
            if key in pct_keys:
                if value < 0 or value > 100:
                    return f"{key}={value} out of range [0, 100]"

            # Multiplier/boost/penalty values must be finite and non-negative
            weight_keys = {
                "strong_month_boost", "weak_month_penalty",
                "min_volume_ratio", "min_trend_strength",
            }
            if key in weight_keys:
                if value < 0:
                    return f"{key}={value} must be non-negative"

            # histogram_threshold can be negative (valid for MACD), just check finite
            import math
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                return f"{key}={value} is not finite"

        return None

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

            # Validate parameters before creating the rule
            validation_err = RuleRegistry._validate_params(rule_name, params)
            if validation_err:
                logger.error(f"Invalid config for rule {rule_name}: {validation_err}")
                return None

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
                weight = rule_config.get("weight", 1.0)
                if not isinstance(weight, (int, float)) or weight <= 0:
                    logger.warning(
                        f"Rule {rule_name} has invalid weight={weight!r}, defaulting to 1.0"
                    )
                    weight = 1.0
                rules.append(rule)
                weights[rule.name] = weight
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
        active_tickers = config.get("active_tickers", {})
        global_rules_config = config.get("rules", {})
        default_exit = config.get("exit_strategy", {"profit_target": 0.07, "stop_loss": 0.05})

        # Check if this symbol has ticker-specific config
        if symbol in active_tickers:
            override = active_tickers[symbol]
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

            logger.info(f"Symbol {symbol}: using {len(rules)} rules")
            return rules, weights, exit_strategy

        # Not in active_tickers - return None to signal use of default rules
        return None, None, default_exit

    @staticmethod
    def get_active_tickers(config: dict) -> Dict[str, List[str]]:
        """
        Get all active tickers from config.

        Returns:
            Dict mapping symbol -> list of rule names
        """
        active_tickers = config.get("active_tickers", {})
        return {
            symbol: override.get("rules", [])
            for symbol, override in active_tickers.items()
        }
