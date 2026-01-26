# Backtesting Results - Rule Optimization

## Overview
This document tracks backtesting results to identify optimal rule configurations for each stock.

**Test Period:** 2021-01-01 to 2026-01-25

---

## Rule Sets Reference

### 4 Enhanced Rules
```
enhanced_buy_dip, momentum_reversal, trend_continuation, average_down
```

### 9 Rules (Enhanced + Indicators)
```
enhanced_buy_dip, momentum_reversal, trend_continuation, average_down,
rsi_oversold, rsi_overbought, macd_bearish_crossover, trend_alignment, trend_break_warning
```

### 14 Rules (All Rules - Enhanced + Indicators + Mining)
```
enhanced_buy_dip, momentum_reversal, trend_continuation, average_down,
rsi_oversold, rsi_overbought, macd_bearish_crossover, trend_alignment, trend_break_warning,
commodity_breakout, miner_metal_ratio, dollar_weakness, seasonality, volume_breakout
```

### 7 Rules (WPM Mining Optimized)
```
commodity_breakout, miner_metal_ratio, volume_breakout, seasonality,
enhanced_buy_dip, momentum_reversal, trend_continuation
```

---

## Test Results by Symbol

### WPM (Wheaton Precious Metals) - STREAMER
| Test | Rules | PT/SL | Trades | Win Rate | Return | Sharpe | Max DD |
|------|-------|-------|--------|----------|--------|--------|--------|
| 1 | 7 Mining Optimized | 12%/6% | 23 | 56.5% | +238.9% | 0.87 | - |
| 2 | 14 All Rules | 12%/6% | 55 | 56.4% | +203.4% | **1.04** | -21.1% |

**Best Config:** 14 All Rules @ 12%/6% (highest Sharpe 1.04)

---

### SLV (Silver ETF)
| Test | Rules | PT/SL | Trades | Win Rate | Return | Sharpe | Max DD |
|------|-------|-------|--------|----------|--------|--------|--------|
| 1 | 9 Enhanced+Indicators | 10%/5% | 60 | 55.0% | +179.6% | 0.57 | -20.7% |
| 2 | 4 Enhanced Only | 10%/5% | 26 | 53.8% | +127.2% | 0.61 | -21.9% |
| 3 | 14 All Rules | 12%/6% | 67 | 55.2% | **+265.8%** | 0.60 | -29.3% |

**Best Config:** 14 All Rules @ 12%/6% (highest return +265.8%)

---

### PPLT (Platinum ETF)
| Test | Rules | PT/SL | Trades | Win Rate | Return | Sharpe | Max DD |
|------|-------|-------|--------|----------|--------|--------|--------|
| 1 | 9 Enhanced+Indicators | 10%/5% | 57 | 59.6% | +133.1% | **1.03** | -18.0% |
| 2 | 14 All Rules | 12%/6% | 55 | 54.5% | +123.3% | 0.68 | **-21.0%** |

**Best Config:** 9 Rules @ 10%/5% (highest Sharpe 1.03, best DD)

---

### CAT (Caterpillar) - INDUSTRIAL
| Test | Rules | PT/SL | Trades | Win Rate | Return | Sharpe | Max DD |
|------|-------|-------|--------|----------|--------|--------|--------|
| 1 | 9 Enhanced+Indicators | 10%/5% | 49 | **65.3%** | +159.0% | **1.14** | -18.5% |

**Best Config:** 9 Rules @ 10%/5% (excellent Sharpe 1.14, highest WR)

---

### ETN (Eaton Corp) - INDUSTRIAL
| Test | Rules | PT/SL | Trades | Win Rate | Return | Sharpe | Max DD |
|------|-------|-------|--------|----------|--------|--------|--------|
| 1 | 9 Enhanced+Indicators | 10%/5% | 52 | 50.0% | +61.1% | 0.46 | -24.2% |
| 2 | 4 Enhanced Only | 10%/5% | 24 | 45.8% | +71.8% | **0.80** | -25.2% |

**Best Config:** 4 Enhanced Only @ 10%/5% (better Sharpe 0.80, but WR below 50%)
**Recommendation:** Consider removing from portfolio - weak performer

---

### IAUM (iShares Gold Micro) - GOLD ETF
| Test | Rules | PT/SL | Trades | Win Rate | Return | Sharpe | Max DD |
|------|-------|-------|--------|----------|--------|--------|--------|
| 1 | 9 Enhanced+Indicators | 10%/5% | 31 | 58.1% | +80.0% | 0.69 | **-13.6%** |

**Best Config:** 9 Rules @ 10%/5% (lowest drawdown of all stocks)

---

### CCJ (Cameco) - URANIUM
| Test | Rules | PT/SL | Trades | Win Rate | Return | Sharpe | Max DD |
|------|-------|-------|--------|----------|--------|--------|--------|
| 1 | 4 Enhanced Only | 10%/5% | 42 | 57.1% | **+492.4%** | **1.17** | -27.3% |

**Best Config:** 4 Enhanced Only @ 10%/5% (exceptional - highest return, best Sharpe)

---

### URNM (Uranium Mining ETF)
| Test | Rules | PT/SL | Trades | Win Rate | Return | Sharpe | Max DD |
|------|-------|-------|--------|----------|--------|--------|--------|
| 1 | 4 Enhanced Only | 10%/5% | 28 | 57.1% | +173.6% | 0.68 | -21.0% |

**Best Config:** 4 Enhanced Only @ 10%/5%

---

### UUUU (Energy Fuels) - URANIUM
| Test | Rules | PT/SL | Trades | Win Rate | Return | Sharpe | Max DD |
|------|-------|-------|--------|----------|--------|--------|--------|
| 1 | 4 Enhanced Only | 10%/5% | 29 | 55.2% | +269.9% | 0.53 | -32.4% |

**Best Config:** 4 Enhanced Only @ 10%/5% (high return but elevated DD)

---

### MP (MP Materials) - RARE EARTH
| Test | Rules | PT/SL | Trades | Win Rate | Return | Sharpe | Max DD |
|------|-------|-------|--------|----------|--------|--------|--------|
| 1 | 9 Enhanced+Indicators | 10%/5% | 89 | 53.9% | +198.3% | 0.47 | -47.4% |
| 2 | 4 Enhanced Only | 10%/5% | 34 | 50.0% | +164.2% | **0.61** | -39.4% |
| 3 | 14 All Rules | 12%/6% | 109 | 49.5% | +146.8% | 0.41 | **-66.3%** |

**Best Config:** 4 Enhanced Only @ 10%/5% (best Sharpe, lowest DD of options)
**Warning:** High risk stock - all configs have >39% drawdown

---

### HL (Hecla Mining) - SILVER MINER - REMOVED
| Test | Rules | PT/SL | Trades | Win Rate | Return | Sharpe | Max DD |
|------|-------|-------|--------|----------|--------|--------|--------|
| 1 | 14 All Rules | 12%/6% | 84 | 56.0% | +153.9% | 0.49 | -47.6% |
| 2 | 4 Enhanced | 10%/5% | 38 | **36.8%** | **-2.9%** | **-0.07** | -38.7% |

**Status:** REMOVED FROM PORTFOLIO - Loses money with any rule configuration

---

### PAAS (Pan American Silver) - SILVER MINER
| Test | Rules | PT/SL | Trades | Win Rate | Return | Sharpe | Max DD |
|------|-------|-------|--------|----------|--------|--------|--------|
| 1 | 14 All Rules | 12%/6% | 77 | 48.1% | +14.2% | 0.17 | -55.9% |

**Recommendation:** REMOVE - Below 50% WR, terrible Sharpe, massive drawdown

---

### FCX (Freeport-McMoRan) - COPPER MINER
| Test | Rules | PT/SL | Trades | Win Rate | Return | Sharpe | Max DD |
|------|-------|-------|--------|----------|--------|--------|--------|
| 1 | 14 All Rules | 12%/6% | 70 | 52.9% | +15.9% | 0.18 | -39.5% |

**Recommendation:** REMOVE - Poor return, terrible Sharpe

---

## Summary: Final Portfolio Configuration

### Tier 1 - BEST (Sharpe > 1.0) - Full Position Size
| Symbol | Type | Rules | PT/SL | Sharpe | Return | Max DD |
|--------|------|-------|-------|--------|--------|--------|
| CCJ | Uranium | 4 Enhanced | 10%/5% | **1.17** | +492.4% | -27.3% |
| CAT | Industrial | 9 Enhanced+Ind | 10%/5% | **1.14** | +159.0% | -18.5% |
| WPM | Streamer | 14 All Rules | 12%/6% | **1.04** | +203.4% | -21.1% |
| PPLT | Platinum ETF | 9 Enhanced+Ind | 10%/5% | **1.03** | +133.1% | -21.0% |

### Tier 2 - GOOD (Sharpe 0.5-1.0) - Normal Position Size
| Symbol | Type | Rules | PT/SL | Sharpe | Return | Max DD |
|--------|------|-------|-------|--------|--------|--------|
| IAUM | Gold ETF | 9 Enhanced+Ind | 10%/5% | 0.69 | +80.0% | **-13.6%** |
| URNM | Uranium ETF | 4 Enhanced | 10%/5% | 0.68 | +173.6% | -21.0% |
| SLV | Silver ETF | 14 All Rules | 12%/6% | 0.60 | +265.8% | -29.3% |

### Tier 3 - HIGH RISK (Sharpe 0.5-0.6, DD > 30%) - Reduced Position Size
| Symbol | Type | Rules | PT/SL | Sharpe | Return | Max DD |
|--------|------|-------|-------|--------|--------|--------|
| MP | Rare Earth | 4 Enhanced | 10%/5% | 0.61 | +164.2% | **-39.4%** |
| UUUU | Uranium | 4 Enhanced | 10%/5% | 0.53 | +269.9% | **-32.4%** |

### REMOVED - Do Not Trade
| Symbol | Type | Issue |
|--------|------|-------|
| HL | Silver Miner | 36.8% WR, -2.9% return, -0.07 Sharpe - LOSING STRATEGY |
| PAAS | Silver Miner | 48.1% WR, 0.17 Sharpe, -55.9% DD |
| FCX | Copper Miner | 0.18 Sharpe, +15.9% return not worth -39.5% DD |
| ETN | Industrial | 45.8% WR below 50% = coin flip |

---

## Key Insights

### By Asset Type:
1. **ETFs/Streamers** (WPM, SLV, PPLT, IAUM) - Work well with more rules (9-14)
2. **Uranium** (CCJ, URNM, UUUU) - Work best with simple 4 Enhanced rules
3. **Individual Miners** (HL, PAAS, FCX) - Poor risk-adjusted returns, avoid

### By Rule Set:
1. **4 Enhanced Rules** - Best for uranium stocks and volatile miners
2. **9 Enhanced+Indicators** - Best for industrials (CAT) and stable ETFs (PPLT, IAUM)
3. **14 All Rules** - Best for precious metals (WPM, SLV)

### Exit Strategy:
- **10%/5% PT/SL** - Better for most stocks
- **12%/6% PT/SL** - Better for WPM and SLV (lets winners run)

---

## Completed Tests

- [x] HL with 4 Enhanced rules @ 10%/5% → **DISASTER: 36.8% WR, -2.9% return**
- [ ] PAAS with 4 Enhanced rules @ 10%/5% - Not worth testing, remove
- [ ] FCX with 4 Enhanced rules @ 10%/5% - Not worth testing, remove

## Future Tests (Optional)

- [ ] WPM with 9 rules @ 10%/5% (compare to 14 rules @ 12%/6%)
- [ ] Test tighter stop loss (4%) on MP to reduce -39% DD
- [ ] Test tighter stop loss (4%) on UUUU to reduce -32% DD

---

## Final Watchlist (9 Symbols)

**Active:** CCJ, CAT, WPM, PPLT, IAUM, URNM, SLV, MP, UUUU

**Removed:** ETN, HL, PAAS, FCX

---

*Last Updated: 2026-01-25*
