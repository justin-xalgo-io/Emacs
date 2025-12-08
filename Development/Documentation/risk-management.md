# xalgo-platform-delta: Risk Management System

This document describes the main algorithms and systems used in xalgo-platform-delta for managing risk in sports betting events.

## Overview

The Delta platform implements a sophisticated real-time risk management system that monitors betting activity and adjusts model parameters to protect against unusual betting patterns, market manipulation, and excessive exposure. The system uses a **Traffic Light** metaphor (Green → Yellow → Red) to indicate risk levels.

---

## Architecture

### Core Components

| Component | Location | Description |
|-----------|----------|-------------|
| **Delta Trader** | `components/delta-v1/` | Main trading logic, processes bets, updates parameters |
| **Risk Utils** | `components/risk/risk_utils/` | Traffic light implementations and risk calculations |
| **Shared Utilities** | `components/shared/bet_utils/` | Signal aggregation, math utilities, martingale detection |

---

## Traffic Light System

The Traffic Light system is the core risk monitoring mechanism. Each traffic light monitors a specific risk metric and transitions between states:

- **🟢 GREEN**: Normal operation, no risk detected
- **🟡 YELLOW**: Warning state, increased monitoring (may trigger automatic hedging trades)
- **🔴 RED**: Critical state, trading may be suspended

### Traffic Light Types

#### 1. **Turnover Traffic Light**
**File**: `risk_utils/turnover_traffic_light.py`

Monitors the rate of turnover (stake amounts) on a specific parameter direction.

- **Algorithm**: Exponential Moving Average (EMA) with multiple half-lives (60s, 240s, 900s, 3600s)
- **Triggers**: When recent turnover exceeds a threshold percentage of total historical turnover
- **Purpose**: Detects sudden spikes in betting volume on a particular outcome

```
Ratio = EMA(recent_turnover) / (total_turnover + initial_signal)
Threshold: ~10-46% depending on time window
```

#### 2. **Liabilities Traffic Light**
**File**: `risk_utils/liabilities_traffic_light.py`

Monitors potential payouts (stake × odds) rather than just stakes.

- **Algorithm**: Same exponential approach as turnover, but applied to liabilities
- **Triggers**: When liabilities grow faster than expected relative to total liabilities
- **Purpose**: Catches high-odds betting patterns that could lead to large payouts

#### 3. **Odds Traffic Light**
**File**: `risk_utils/odds_traffic_light.py`

Detects when punters suddenly shift to betting at higher odds.

- **Algorithm**: Tracks the ratio of total liabilities to total turnover (effective average odds)
- **Triggers**: When average odds increase significantly
- **Purpose**: Identifies potentially informed betting at longer odds

#### 4. **Drift Traffic Light**
**File**: `risk_utils/drift_traffic_light.py`

Monitors how much a model parameter has moved from its initial value.

- **Algorithm**: `drift_ratio = |current_value - initial_value| / sigma`
- **Thresholds**: 
  - Yellow: 3σ drift
  - Red: 6σ drift
- **Purpose**: Alerts when the market is moving parameters significantly

#### 5. **Drift Delta Traffic Light**
**File**: `risk_utils/drift_traffic_light.py`

Similar to Drift, but only tracks drift caused by Delta (automated trading) updates.

- **Thresholds**: Yellow at 10σ, Red at 15σ
- **Resets**: When non-Delta sources update the parameter
- **Purpose**: Monitors the cumulative effect of automated trading

#### 6. **Skewness Traffic Light**
**File**: `risk_utils/skewness_traffic_light.py`

Detects imbalanced betting patterns between "plus" and "minus" directions.

- **Algorithm**: 
  ```
  skewness_ratio = (turnover_plus - turnover_minus) / (turnover_plus + turnover_minus + base)
  ```
- **Triggers**: When ratio exceeds ~65-80%
- **Purpose**: Identifies one-sided betting that could indicate informed traders

#### 7. **Skewness Asymmetric Traffic Light**
**File**: `risk_utils/skewness_traffic_light.py`

Enhanced skewness detection using a tanh model for asymmetric parameters.

- **Algorithm**: Uses `TanHyperbolicSkewnessModel` for non-linear skewness calculation
- **Purpose**: Better handles parameters where expected turnover distribution is inherently asymmetric

#### 8. **Relative Turnover Traffic Light**
**File**: `risk_utils/relative_turnover_traffic_light.py`

Compares turnover on a secondary parameter against a main parameter (e.g., `goal_a`).

- **Algorithm**: `ratio = turnover_this_param / (turnover_main_param + initial_signal)`
- **Thresholds**: Yellow at 75%, Red at 100%
- **Purpose**: Catches unusual concentration on minor parameters

#### 9. **Player Props Traffic Light**
**File**: `risk_utils/player_props_traffic_light.py`

Monitors turnover distribution across player proposition bets.

- **Algorithm**: Compares actual turnover on a player vs expected turnover based on parameter likeliness
- **Adjusts for**: Number of players, parameter probabilities
- **Purpose**: Detects when a specific player is attracting disproportionate action

#### 10. **Potential Payout Traffic Light**
**File**: `risk_utils/potential_payout_traffic_light.py`

Event-level limit on total potential payouts.

- **Algorithm**: Cumulative sum of all potential payouts
- **Thresholds**: Based on event SLA tier (e.g., 50k for LOW tier pre-match)
- **Purpose**: Hard cap on exposure per event

---

## Signal Aggregation Algorithms

### First Order Filter (Exponential Moving Average)
**File**: `shared/bet_utils/signal_aggregation.py`

Core building block for most traffic lights:

```python
state *= exp(-dt / time_constant)
state += sum(signals * exp(-dts / time_constant))
```

- **Time Constants**: Typically 60s, 240s, 900s, 3600s for different reaction speeds
- **Decay**: Older signals contribute less to current state

### Skewness Detector
**File**: `shared/bet_utils/signal_aggregation.py`

Maintains separate EMA filters for positive and negative signed axes to detect betting imbalance.

---

## Mathematical Utilities

### Gradient-to-Direction Mapping
**File**: `shared/bet_utils/math_utils.py`

Converts bet gradients to trading directions:

```python
directions = mapping * step_size
signed_axes = argmax(|mapping|) * sign(mapping)
```

**Step Size Calculation**:
- Based on `sigma_wise` (individual parameter uncertainty)
- Adjusted by time-to-start (larger steps closer to match)
- Factored by accumulated turnover

### Signed Axis
Each bet is mapped to a "signed axis" representing:
- **Parameter Index**: Which model parameter the bet affects (1-indexed)
- **Sign**: Direction of effect (+ or -)
- Example: `+3` means positive effect on parameter 3, `-3` means negative effect

### Parameter Turnovers
Different parameter types have different "expected" turnover allocations:
- Main parameters (goals): €1,000,000
- Rating parameters: €350,000
- Player props: €1,000
- Other: €100,000

---

## Instance Risk Tracking
**File**: `risk_utils/instance_risk_delta.py`

Per-parameter tracking of:
- Filtered turnover (EMA with game-state-dependent time constant)
- Filtered liability
- Total turnover and liability
- Different decay rates for pre-match (1.65 hours) vs in-play (4.5 minutes)

---

## Trading Logic

### Trader Instance
**File**: `delta-v1/delta_v1/trader_instance.py`

Main trading workflow:
1. Receive bet batches from Kinesis stream
2. Map bets to signed axes using gradients
3. Aggregate signals and update traffic lights
4. Compute parameter updates based on betting pressure
5. Apply martingale and throttling logic
6. Publish parameter updates

### Yellow Traffic Light Auto-Trading

When a traffic light goes YELLOW:
1. If `trade_on_yellow` is enabled
2. And fewer than `_ALLOWED_YELLOW_TL_TRADES` have occurred
3. → Automatically trade in the opposite direction to hedge
4. → Reset the traffic light and increment `num_yellow_delta`

### Martingale Detection
**File**: `shared/bet_utils/martingale.py`

Detects when trading is consistently moving in one direction:

```python
step_factor = sqrt(sum_step1² / sum_step2)
```

- Used to scale down trading when the market is one-sided
- Prevents over-trading during sustained directional pressure

---

## PPM (Punter Profiling Model)
**File**: `delta-v1/delta_v1/ppm/ppm.py`

Neural network model that profiles punter behavior:

**Inputs**:
- `log1p(sum_prob)`: Log of sum probabilities
- `log1p(count_won)`: Log of winning bet count
- `hypo_vip`: Hypothesis test for VIP punter
- `hypo_wg`: Hypothesis test for winning group
- `flag_vip`, `flag_wg`: Punter category flags

**Output**: Signal strength used to weight betting impact

---

## Redis Storage

Traffic light states and instance risk data are persisted in Redis using Lua scripts for atomic operations:
- `write_traffic_lights.lua`: Atomic traffic light updates
- `aggregate_objects.lua`: Aggregation operations
- `increment_counter.lua`: Counter management
- `reset_num_yellow_delta.lua`: Reset yellow trade counters

---

## Color Change Logic

Traffic lights can only escalate (GREEN → YELLOW → RED) automatically. De-escalation requires:
- Human intervention via supervisor
- Reset with a `ChangeReason` containing source `HUMAN`

When reset by human:
- `threshold_factor` doubles (makes future triggers harder)
- All internal filters reset to zero

---

## Event-Level vs Parameter-Level

| Level | Scope | Examples |
|-------|-------|----------|
| **Event** | Entire match/event | Potential Payout, Event Turnover |
| **SignedAxis** | Per parameter direction | Turnover, Liabilities, Drift, Skewness |
| **Outcome** | Unmapped/specific outcomes | Outcome Turnover, Outcome Liabilities |

---

## Configuration

### Thresholds
Most thresholds are defined as class variables (e.g., `_THRESHOLDS`) and use a pattern:
```python
{
    Color.RED: {half_life_seconds: threshold_value},
    Color.YELLOW: {half_life_seconds: threshold_value},
}
```

### Sport-Specific Config
The `SkewnessConfig` class supports sport-specific configurations:
- Default: Football/Soccer
- Tennis: Faster windows, bidirectional triggers

---

## Summary Flow

```
Bet Arrives → Map to Signed Axis → Update Traffic Lights
                                         ↓
                              Check Thresholds
                                         ↓
                              GREEN/YELLOW/RED
                                         ↓
                        [If YELLOW + auto_trade] → Hedge Trade
                        [If RED] → Suspend/Alert
```

This system enables xalgo-platform-delta to dynamically manage betting risk in real-time, protecting against market manipulation, informed betting, and excessive exposure while maintaining efficient market operations.
