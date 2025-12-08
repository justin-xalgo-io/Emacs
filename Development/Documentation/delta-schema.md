# Parameter Types in `ingest.py`

This document describes the Parameter-related classes used in the xalgo-schemas calculation unit.

## ParamterType

An enumeration that defines the types of model parameters used in calculation units. These parameters are typically used in statistical/ML models for predictions:

| Value | Description |
|-------|-------------|
| `FLOAT_2_SIGMA` | A floating-point parameter with 2-sigma (standard deviation) uncertainty - used for general parameters with statistical uncertainty bounds |
| `A_FLOAT_2_SIGMA` | An "Asymmetric" float with 2-sigma - likely for parameters where uncertainty differs in different directions |
| `S_FLOAT_2_SIGMA` | A "Symmetric" float with 2-sigma - for parameters with symmetric uncertainty |
| `EPOCH` | A time-based parameter representing seconds elapsed since a reference time |
| `BOOL_DELTA` | A boolean change/delta parameter, likely used for discrete state changes |

> **Note:** There's a typo in the enum name - it's `ParamterType` not `ParameterType`.

## SupervisorParameterType

An enumeration for parameters controlled by supervisors (human operators or automated systems):

| Value | Description |
|-------|-------------|
| `CATEGORICAL` | A single value from a discrete set of categories |
| `INT_IN_RANGE` | An integer value constrained within a min/max range |
| `FLOAT_IN_RANGE` | A float value constrained within a min/max range |
| `CATEGORICAL_DISTRIBUTION` | A probability distribution over a set of categories (values must sum to 1) |
| `PROBABILITY` | A probability value (0.0 to 1.0) |

---

## Parameter Classes

### Base Classes

| Class | Description |
|-------|-------------|
| `ParameterInfo` | Metadata about a parameter: name, description, scale (LOGIT/LOG/INTERVAL/IDENTITY), and optional tags |
| `SupervisorParameterInfo` | Parameter info for supervisor parameters (always uses IDENTITY scale) |
| `ParameterValueBase` | Abstract base for parameter values with key, single_value, and sigma_wise_value properties |
| `ParameterBase` | Abstract base containing parameter_type, info, and optional extra_info |
| `SupervisorParameterBase` | Abstract base for supervisor-controlled parameters |

### Parameter Meta Classes (for model parameters)

| Class | `ParamterType` | Purpose |
|-------|----------------|---------|
| `ParameterMetaBoolDelta` | `BOOL_DELTA` | Boolean state change parameters with expression and match entity |
| `ParameterMeta2Sigma` | `FLOAT_2_SIGMA` | Float parameters associated with teams, players, or matches |
| `ParameterMetaAS` | `A_FLOAT_2_SIGMA`, `S_FLOAT_2_SIGMA` | Asymmetric/symmetric float parameters for match entities |
| `ParameterMetaEpoch` | `EPOCH` | Time-based parameter (seconds since ref_time) |

### Supervisor Parameter Classes (for human/system control)

| Class | `SupervisorParameterType` | Purpose |
|-------|---------------------------|---------|
| `IntInRange` | `INT_IN_RANGE` | Integer value within a validated min/max range |
| `FloatInRange` | `FLOAT_IN_RANGE` | Float value within a validated min/max range |
| `ProbabilityParameter` | `PROBABILITY` | A probability value for teams/players/matches |
| `CategoricalParameter` | `CATEGORICAL` | Single selection from a list of categories |
| `CategoricalDistribution` | `CATEGORICAL_DISTRIBUTION` | Probability distribution over categories (must sum to 1.0) |

### State Classes

| Class | Description |
|-------|-------------|
| `ParameterState` | Full state containing numpy arrays for fast parameters, sigma values (shared & wise), epoch value, metadata list, and suspension info |
| `FastParameterState` | Lightweight state with topology fingerprint for quick parameter updates |
| `ParameterEpochValue` | Time-based value with a reference time and width (default 1 hour) |
| `SupervisorState` | Contains supervisor parameters, auto-stake/trade flags, suspension status, in-play mode, and bet delays |

---

## Usage Context

These parameter types are used throughout xalgo-platform-delta for:

- **Risk calculation** (`components/delta-v1/delta_v1/risk.py`) - computing risk based on parameter metadata
- **Math utilities** (`components/shared/bet_utils/math_utils.py`) - calculating parameter turnovers
- **State management** (`components/risk/risk_utils/redis.py`) - storing/retrieving parameter state from Redis
- **Parameter updates** (`applications/parameter-update-consumer/`) - consuming and applying parameter changes
