# Initial Price Model (IPM) System - Technical Documentation

## Overview

The **Initial Price Model (IPM)** system is a comprehensive machine learning pipeline that generates predictive parameters and features for soccer match simulations. It processes historical match data (closing intensities from simulations) through nightly batch jobs to calculate team ratings, player statistics, league characteristics, and counter biases. These features are stored in AWS SageMaker Feature Stores and used to initialize simulation parameters before matches begin.

The IPM system operates on a **nightly ETL pipeline** architecture with three main stages:
1. **Data Archiving & ETL** - Raw simulation outputs → Feature tables
2. **Feature Engineering** - Statistical models calculate ratings and features
3. **Feature Store Population** - Results stored for real-time retrieval

---

## Architecture & Data Flow

### 1. Data Ingestion Pipeline

**Source**: Soccer simulation runs produce three key outputs:
- `sim_values.parquet` - Simulation prediction attributes
- `sim_command.json` - Simulation command metadata
- `player_likeliness.json` - Normalized player likelihoods

**Processing Flow**:
```
Soccer Run → S3 Simulation Archive Bucket → Lambda Trigger → Kinesis Firehose Streams → S3 Raw Bucket
```

The archiving lambda (`components/soccer_gateway/archiving_lambda`) validates incoming data:
- Confirms it's a soccer simulation (not tennis/other sports)
- Filters for REAL instance settings (excludes TEST simulations)
- Flattens hierarchical data into tabular format
- Routes to 5 separate Firehose delivery streams:
  1. **Simulation delivery stream** - Prediction attributes + player likelihoods
  2. **Model input delivery stream** - Model input parameters
  3. **Supervisor parameter InRange stream** - Continuous supervisor params (e.g., start probabilities)
  4. **Supervisor parameter Categorical stream** - Categorical params (e.g., player positions)
  5. **Supervisor parameter Formation stream** - Team formation distributions

**Output**: Raw tabular data in S3 (`Soccer Raw S3 Bucket`) with prefixes like:
- `soccer-simulation-intensity/`
- `soccer-model-input/`
- `soccer-supervisor-param-categorical/`
- `soccer-supervisor-param-inrange/`
- `soccer-supervisor-param-formation/`

---

### 2. Nightly ETL Jobs (Glue/Step Functions)

Three main **Step Function orchestrators** run nightly:

#### A. SOCCERSimulationIntensityOrchestrator
**Purpose**: Process raw simulation intensities → aggregated closing intensities

**Steps**:
1. **Glue Crawler** - Creates/updates `soccer-simulation-intensity` raw table (AWS Glue Data Catalog)
2. **Intermediate Processing** - Aggregates data over two simulations per match
   - Output: `soccer-simulation-intensity` intermediate table (S3 prefix: `soccer-simulation-intensity`)
3. **Release Processing** - Produces final closing intensities
   - Output: `soccer-closing-simulation-intensity` release table

**Key Concept**: "Closing intensity" = the final predicted intensity for a match just before it starts (used as ground truth for model training)

#### B. SOCCERModelInputOrchestrator
**Purpose**: Process raw model inputs → closing model inputs

**Steps**:
1. Crawl raw bucket → `soccer-model-input` raw table
2. Release processing → `soccer-closing-model-input` release table

#### C. SOCCERSupervisorParamOrchestrator
**Purpose**: Process supervisor parameters across 3 categories

**Steps**:
1. Crawl 3 raw prefixes → 3 raw tables:
   - `soccer-supervisor-param-categorical`
   - `soccer-supervisor-param-inrange`
   - `soccer-supervisor-param-formation`
2. Release processing → 3 closing tables:
   - `soccer-closing-supervisor-param-categorical`
   - `soccer-closing-supervisor-param-inrange`
   - `soccer-closing-supervisor-param-formation`

**Glue Scripts Location**: `components/soccer_glue_scripts`

---

### 3. Feature Engineering Jobs (Nightly Step Functions)

Multiple Step Functions process closing data to generate features:

---

## Feature Engineering Algorithms

### A. Team Rating System (A+S Rating Model)

**Location**: `components/ipm_pipeline/ipm_pipeline/models/model.py` (`ScipyModel` class)

**Purpose**: Calculate team attack/defense ratings and home advantage from match goal intensities

**Mathematical Model**:

The system solves two sparse linear systems using weighted least squares:

**Rating System (A)**:
```
0.5 * (λ_home - λ_away) = rating_home - rating_away + home_advantage
```

**Aggression System (S)**:
```
0.5 * (λ_home + λ_away) = aggression_home + aggression_away
```

Where:
- `λ_home`, `λ_away` = log goal intensities (from closing simulations)
- `rating` = team's relative strength (attack - defense balance)
- `aggression` = team's overall scoring rate
- `home_advantage` = league/event-group specific home field advantage

**Implementation Details**:

1. **Encoding** (`_encode` method):
   - Maps team IDs and event group IDs to matrix indices
   - Creates lookup dictionaries: `id2idx`, `idx2id`, `id2idx_eventgroup`

2. **Design Matrices**:
   - **A Matrix** (`_sparse_A_matrix`): Sparse CSC matrix of size `(n_matches, n_teams + n_event_groups)`
     - Each row represents one match
     - Columns: [team_ratings..., home_advantages...]
     - Values: +weight for home team, -weight for away team, +weight for event group
   
   - **S Matrix** (`_sparse_S_matrix`): Sparse CSC matrix of size `(n_matches, n_teams)`
     - Each row represents one match
     - Columns: [team_aggressions...]
     - Values: +weight for both home and away teams

3. **Target Vectors**:
   - **A target**: `0.5 * (λ_home - λ_away) * weight`
   - **S target**: `0.5 * (λ_home + λ_away) * weight`

4. **Weighting** (`calc_weights` from `utils.py`):
   - Exponential decay based on match age: `weight = exp(-days_since_match * ln(2) / halflife)`
   - Default halflife = 30 days (recent matches weighted more heavily)

5. **Solver**:
   - Uses `scipy.sparse.linalg.lsqr` (sparse least squares)
   - Solves both A and S systems independently
   - Extracts coefficients for ratings, aggression, home advantage

**Validation** (`validate` method):
- Merges calculated ratings with validation set (recent 60 days)
- Recalculates intensities from ratings: `λ_calc = ±rating_home ∓ rating_away + aggression_home + aggression_away ± home_adv`
- Computes relative error: `|(exp(λ_calc - λ_actual) - 1)|`
- Sets validation flag if mean relative error < 10%

**Feature Stores Populated**:
- **Single Competition Rating Feature Store** - Key: `team_xid-event_group_xid`
  - Fields: `rating`, `aggression`, `validation_flag`
- **Single Competition Home Advantage Feature Store** - Key: `event_group_xid`
  - Field: `home_advantage`
- **Team Single Comp Rating System Map Feature Store** - Key: `team_xid`
  - Maps teams to their domestic league (only if played in last 30 days)

**Step Function**: `TrainingJobsStateMachine` > Team Rating Task

**Code Path**: 
- Main script: `components/soccer_gateway/training_jobs/team_jobs/ratings`
- Pipeline: `components/ipm_pipeline/ipm_pipeline/team_ratings/processing.py`

---

### B. League Rating System (Cross-League Model)

**Location**: `components/ipm_pipeline/ipm_pipeline/models/cross_league_model.py` (`CrossLeagueModel`)

**Purpose**: Calculate relative league strength for cross-competition matches (e.g., Champions League, Europa League)

**Mathematical Model**:

Solves a weighted least squares system:

```
0.5 * (λ_home - λ_away) - a = league_rating_home - league_rating_away + home_advantage_cross
```

Where:
- `a` = team's single-competition rating (from Team Rating System above)
- `league_rating` = strength adjustment for the league
- `home_advantage_cross` = cross-competition specific home advantage

**Key Difference from Team Rating**:
- Uses team's **domestic league rating** as input (fetched from Single Competition Rating Feature Store)
- Calculates **league-level** adjustments rather than team-level
- Longer halflife (default 120 days) because cross-competition matches are rarer

**Algorithm**:

1. **Data Preparation**:
   - Requires `home_league_id_key` and `away_league_id_key` (team's domestic league)
   - Fetches team ratings from Feature Store to use as baseline (`a` parameter)

2. **Encoding**:
   - Maps league IDs to indices
   - Creates design matrix with league ratings and cross-competition home advantages

3. **Sparse Matrix Construction** (`_sparse_A_matrix`):
   - Rows: matches
   - Columns: [league_ratings..., cross_competition_home_advantages...]
   - Target adjusted by subtracting team rating: `target = 0.5 * (λ_h - λ_a) - a`

4. **Solver**:
   - Scipy sparse LSQR
   - Extracts league ratings and cross-competition home advantages

**Feature Stores Populated**:
- **League Ratings Feature Store** - Key: `event_group_xid`
  - Field: `league_a` (league rating adjustment)
- **Cross Competition Home Advantage Feature Store** - Key: `event_group_xid`
  - Field: `home_advantage` (cross-competition specific)

**Step Function**: `TrainingJobsStateMachine` > League Rating Task

**Code Path**:
- Main script: `components/soccer_gateway/training_jobs/league_jobs`
- Pipeline: `components/ipm_pipeline/ipm_pipeline/league_ratings/processing.py`

---

### C. Counter Bias Model (Linear Regression)

**Location**: `components/ipm_pipeline/ipm_pipeline/models/linear_model.py` (`LeastSquareLinearModel`)

**Purpose**: Model team-specific bias for event counters (corners, offsides, shots, fouls, cards) as a function of goal parameters

**Mathematical Model**:

For each counter (corner, offside, etc.), solves:

```
lg_λ_counter_home = k_a * a + k_s * s + bias_home
lg_λ_counter_away = -k_a * a + k_s * s + bias_away
```

Where:
- `a = (lg_λ_goal_home - lg_λ_goal_away) / 2` (attack-defense differential)
- `s = (lg_λ_goal_home + lg_λ_goal_away) / 2` (match aggression)
- `k_a`, `k_s` = league-specific linear coefficients (how counter relates to goals)
- `bias` = team-specific bias for the counter

**Counters by Tier**:

| Counter         | Low Tier | Mid Tier | High Tier |
|-----------------|----------|----------|-----------|
| Corner          | ✓        | ✓        | ✓         |
| Offside         | -        | ✓        | ✓         |
| Shotsaved       | -        | -        | ✓         |
| Shotofftarget   | -        | -        | ✓         |
| Blockedshot     | -        | -        | ✓         |
| Owngoal         | -        | -        | ✓         |
| Foul            | -        | -        | ✓         |
| Yellow Card     | -        | -        | ✓         |
| Red Card        | -        | -        | ✓         |

**Algorithm**:

1. **Prepare Counter Data**:
   - For card counters (yellow/red): aggregates `direct{card}` + `{card}fromfoul` intensities
   - For foul counter: aggregates `onlyfoul` + `yellowfromfoul` + `redfromfoul` intensities
   - Other counters use raw intensity

2. **Calculate A and S**:
   - `a = (lg_λ_goal_home - lg_λ_goal_away) / 2`
   - `s = (lg_λ_goal_home + lg_λ_goal_away) / 2`

3. **Construct Design Matrix**:
   - Interleaves home/away teams: `[team1_home, team1_away, team2_home, team2_away, ...]`
   - Columns: `[bias_team1, bias_team2, ..., k_a, k_s]`
   - Each match contributes 2 rows (home and away)

4. **Build AS Matrix**:
   - For home teams: `[0, ..., 1 (at team index), ..., 0, a, s]`
   - For away teams: `[0, ..., 1 (at team index), ..., 0, -a, s]`

5. **Solve**:
   - Scipy sparse LSQR: `AS * [biases, k_a, k_s]^T = counter_intensities`
   - Extracts team biases and league coefficients

**Feature Stores Populated**:
- **Single Competition Counter Bias Feature Store** - Key: `team_xid-event_group_xid`
  - Fields: `bias`, `k_a`, `k_s` (for each counter type)

**Step Function**: `TrainingJobsStateMachine` > Team Counter Bias Task

**Code Path**:
- Main script: `components/soccer_gateway/training_jobs/team_jobs/league_bias`
- Pipeline: `components/ipm_pipeline/ipm_pipeline/league_bias/processing.py`

---

### D. Player Props Model (Weighted Averages)

**Location**: `components/ipm_pipeline/ipm_pipeline/player/processing.py`

**Purpose**: Calculate expected player statistics from historical closing data

**Features Calculated** (by tier):

| Feature             | Low Tier | Mid Tier | High Tier | Description                                    |
|---------------------|----------|----------|-----------|------------------------------------------------|
| play_time           | -        | ✓        | ✓         | Expected playing time if started (minutes)     |
| player_started      | -        | ✓        | ✓         | Probability of starting lineup                 |
| position            | -        | ✓        | ✓         | Most common position (GK/DEF/MID/FWD)         |
| player_goal         | -        | ✓        | ✓         | Goal likelihood per match                      |
| player_shot         | -        | -        | ✓         | Shot likelihood per match                      |
| player_shotontarget | -        | -        | ✓         | Shot on target likelihood per match            |
| player_foul         | -        | -        | ✓         | Foul likelihood per match                      |
| player_yellowcard   | -        | -        | ✓         | Yellow card likelihood per match               |
| player_redcard      | -        | -        | ✓         | Red card likelihood per match                  |
| header_bodypart     | -        | -        | ✓         | % of goals scored by header                    |
| outsidebox_location | -        | -        | ✓         | % of goals scored from outside box             |

**Algorithm**:

**Core Utility**: `weighted_mean` function (`components/ipm_pipeline/ipm_pipeline/utils.py`)

```python
def weighted_mean(x, scale, weight_function, axis=0):
    """
    Compute weighted mean with scale transformations
    
    scale: LINEAR | LOG | LOGIT
    weight_function: typically exp_decay(halflife)
    """
    # Transform to linear scale
    if scale == LOG: x = exp(x)
    if scale == LOGIT: x = expit(x)  # sigmoid
    
    # Create time variable (match number, excluding NaN)
    time_var = create_time_var(x, axis)
    
    # Calculate weights
    weights = weight_function(time_var)
    
    # Weighted average
    mean = sum(weights * x) / sum(weights)
    
    # Transform back
    if scale == LOG: mean = log(mean)
    if scale == LOGIT: mean = logit(mean)
    
    return mean
```

**Processing Steps**:

1. **Create Player Key**: `player_xid-team_xid` (players tracked per team)

2. **Calculate Features** (separate functions for each):
   
   - **Play Time** (`calculate_expected_playtime`):
     - Filter: matches where player started
     - Scale: LINEAR
     - Weight: exponential decay (halflife = match_horizon_player_params)
   
   - **Starting Probability** (`calculate_expected_stating_probability`):
     - Scale: LOGIT (probability space)
     - Weight: exponential decay (halflife = match_horizon_supervisor_params)
   
   - **Position** (`calculate_expected_player_position`):
     - Mode (most frequent) over last N matches
     - Tie-breaker: most recent
   
   - **Likelihoods** (goals, shots, fouls, cards):
     - Scale: LINEAR (already likelihoods, not probabilities)
     - Weight: exponential decay
     - Normalization: likelihoods already normalized in archiving
   
   - **Goal Characteristics** (header %, outside box %):
     - Scale: LOGIT (percentages)
     - Weight: exponential decay

3. **Missing Data Handling**:
   - If no recent data → fetch latest features from Feature Store
   - Prevents cold-start issues for benched/injured players

**Weighted Mean Utilities**:

- **Exponential Decay**: `exp_decay(time_variable, halflife)`
  - `weight = exp(-match_number * ln(2) / halflife)`
  - Recent matches weighted more heavily

- **NaN-Safe Summation**: `nansumwrapper`
  - Treats NaN as zero UNLESS all values are NaN (then returns NaN)
  - Prevents biased averages from missing data

- **Time Variable Creation**: `create_time_var`
  - Assigns sequential numbers to non-NaN observations: `[0, 1, 2, ...]`
  - NaN observations get NaN time variable
  - Ensures weights only apply to valid data

**Feature Store Populated**:
- **Player Props Feature Store** - Key: `player_xid-team_xid`
  - All features listed in table above

**Step Function**: `PlayerPropsJobsStateMachine` > Player Props Task

**Code Path**:
- Main script: `components/soccer_gateway/training_jobs/player_jobs/player_likeliness`
- Pipeline: `components/ipm_pipeline/ipm_pipeline/player/`

---

### E. Team Formation Distribution

**Location**: `components/ipm_pipeline/ipm_pipeline/team_formation/`

**Purpose**: Calculate probability distribution over possible team formations at match start

**Possible Formations** (defenders, midfielders, forwards):
```
[3-3-4, 3-4-3, 3-5-2, 3-6-1,
 4-2-4, 4-3-3, 4-4-2, 4-5-1, 4-6-0,
 5-2-3, 5-3-2, 5-4-1]
```

**Algorithm** (inferred from context):

1. **Extract Formations**: From supervisor parameter categorical closing data
2. **Count Frequencies**: Over recent match window per team
3. **Normalize**: Convert counts to probability distribution
4. **Store Distribution**: As categorical distribution parameter

**Feature Store Populated**:
- **Team Formation Dist Feature Store** - Key: `team_xid`
  - Field: distribution over 12 formation categories

**Step Function**: `PlayerPropsJobsStateMachine` > Team Formation Task

**Code Path**:
- Main script: `components/soccer_gateway/training_jobs/team_jobs/team_formations`
- Pipeline: `components/ipm_pipeline/ipm_pipeline/team_formation/`

---

## Feature Retrieval & Parameter Generation

**Location**: `components/initial_price_lib/`

### Feature Store Client

**Class**: `FeatureStoreClient` (`initial_price_lib/feature_store.py`)

**Purpose**: Batch retrieval from AWS SageMaker Feature Store Runtime

**Key Methods**:

1. **create_feature_group_map**:
   - Determines which Feature Stores to query based on event group type:
     - **Domestic League**: Single Competition Ratings, Bias, Home Advantage
     - **Cross Competition**: Cross League Ratings, Cross Home Advantage
     - **Cup**: Special handling for mixed league matchups
   
2. **create_feature_store_identifiers**:
   - Builds batch get record identifiers from team/league keys
   - Example: `FSKeys(home_key="tea_xxx-egr_yyy", away_key="tea_zzz-egr_yyy", league="egr_yyy")`

3. **batch_get_records**:
   - Uses `SageMakerFeatureStoreRuntimeClient.batch_get_record`
   - Retrieves features for home/away teams in parallel
   - Handles missing records gracefully (returns defaults or None)

4. **parse_feature_store_response**:
   - Converts Feature Store JSON to structured `Feature` objects
   - Validates response status (COMPLETE vs. INCOMPLETE)

### Parameter State Generation

**Team Parameters** (`initial_price_lib/team/default.py`):

**Function**: `get_default_param_state`

**Generates**:
- **A_FLOAT_2_SIGMA**: Attack parameter (μ, σ_shared, σ_wise)
- **S_FLOAT_2_SIGMA**: Scoring parameter (μ, σ_shared, σ_wise)
- **FLOAT_2_SIGMA**: Other team parameters (e.g., counter biases)
- **BOOL_DELTA**: Draw factor (special handling for probability of draw)

**Parameter Templates** (`initial_price_lib/team/meta.py`):
- `legacy_meta_dict`: Core A+S parameters
- `added_time_meta_dict`: Added time parameters
- `penalty_meta_dict`: Penalty parameters (v2 model only)
- `extra_time_added_time_meta_dict`: Extra time parameters (v2 model only)

**Algorithm**:

1. **Fetch Default Values**: From config (`soccer_config.default_values.linear_models`)
   - Provides (μ, σ) defaults if Feature Store lookup fails

2. **Create Parameter Meta**:
   - Maps parameter templates to entities (match, home team, away team)
   - Generates expressions for BOOL_DELTA (e.g., `home_totalgoal == away_totalgoal` for draw)

3. **Construct Parameter State**:
   - Expected values (μ): from Feature Store or defaults
   - Sigma shared (σ_shared): cross-parameter correlation
   - Sigma wise (σ_wise): parameter-specific variance

**Player Parameters** (`initial_price_lib/player/default.py`):

**Supervisor Parameters**:
- **PROBABILITY**: `player_started` (logit-transformed starting probability)
- **CATEGORICAL**: `position` (GK, DEF, MID, FWD)
- **CATEGORICAL_DISTRIBUTION**: `team_formation` (distribution over 12 formations)

**Player Likelihoods**:
- **FLOAT_2_SIGMA**: `player_goal`, `player_shot`, `player_foul`, etc.
  - μ = expected likelihood from Feature Store
  - σ = variance for sampling

**Algorithm**:

1. **Fetch Player Props**: From Player Props Feature Store (key: `player_xid-team_xid`)
2. **Transform Probabilities**: `logit(player_started)` for PROBABILITY parameters
3. **Create Categorical Distributions**: Team formation gets full distribution
4. **Generate Tags**: Side (HOME/AWAY), player name for UI display

---

## Data Processing Utilities

### Weighted Mean Functions

**Core Algorithm** (`ipm_pipeline/utils.py`):

```python
weighted_mean(x, scale, weight_function, axis)
```

**Scale Transformations**:
- **LINEAR**: No transformation (raw values)
- **LOG**: `exp(x)` before averaging, `log(mean)` after
  - Used for intensities (multiplicative quantities)
- **LOGIT**: `sigmoid(x)` before averaging, `logit(mean)` after
  - Used for probabilities (bounded [0,1])
  - `sigmoid(x) = 1 / (1 + exp(-x))`
  - `logit(p) = log(p / (1-p))`

**Rationale**: Averaging in transformed space respects the geometry of the data
- Log space: geometric mean (appropriate for rates/intensities)
- Logit space: average probability (respects [0,1] bounds)

### Exponential Decay Weighting

**Function**: `exp_decay(time_variable, halflife)`

```python
decay_factor = ln(2) / halflife
weights = exp(-time_variable * decay_factor)
```

**Interpretation**:
- After `halflife` time units, weight is 50% of original
- Continuous exponential decay (not discrete step function)

**Typical Halflives**:
- Team ratings: 30 days (form changes quickly)
- Cross-league ratings: 120 days (stable league strength)
- Player props: varies by parameter (match-based, typically 10-20 matches)

### NaN-Safe Summation

**Function**: `nansumwrapper(array, axis)`

**Behavior**:
- If ANY non-NaN values exist: treat NaN as 0, return sum of non-NaN
- If ALL values are NaN: return NaN (preserves "no data" signal)

**Critical for**:
- Weighted averages with missing data
- Prevents bias from incomplete match histories
- Distinguishes "zero events" from "no observations"

---

## Key Design Patterns

### 1. Sparse Matrix Optimization

All rating systems use **scipy.sparse.linalg.lsqr** for efficiency:
- Typical problem: 1000+ teams, 10,000+ matches
- Design matrices are >99% zeros (each row touches ~3 teams)
- Sparse representation: store only non-zero elements
- LSQR: iterative solver, memory-efficient for sparse systems

### 2. Exponential Time Weighting

**Philosophy**: Recent performance is more predictive than distant past

**Implementation**:
- Every model uses exponential decay weights
- Halflife tuned per model (30-120 days)
- Weights computed once, applied to design matrix values

**Effect on Estimation**:
- Recent matches get higher leverage in regression
- Old matches contribute but don't dominate
- Smooth transition (no hard cutoff)

### 3. Hierarchical Feature Keys

**Team Features**: `team_xid-event_group_xid`
- Allows team to have different ratings in different competitions
- Example: Manchester United in Premier League vs. Champions League

**Player Features**: `player_xid-team_xid`
- Tracks players per team (handles transfers)
- Example: player moves from Team A → Team B, new key created

**League Features**: `event_group_xid`
- Single key per league/competition
- Home advantage, league rating

### 4. Multi-Scale Averaging

**Why Different Scales?**:
- **Intensities (LOG)**: Rates are multiplicative (doubling is additive in log space)
- **Probabilities (LOGIT)**: Bounded [0,1], logit unbounds to (-∞, ∞)
- **Durations (LINEAR)**: Playing time is naturally additive

**Transformation Workflow**:
```
Raw Data → Transform to Linear → Weight → Average → Transform Back → Store
```

### 5. Validation Flags

Team ratings include **validation_flag**:
- Recalculates intensities from learned ratings
- Compares to actual closing intensities
- Flags team if relative error > 10%
- Allows downstream systems to trust/distrust ratings

---

## Parameter Type Reference

### From `xalgo_schemas.calculation_unit.ingest`

**A_FLOAT_2_SIGMA**: Attack parameter with 2-sigma uncertainty
- `expected`: μ (mean attack strength)
- `sigma_shared`: correlation with other parameters
- `sigma_wise`: independent variance

**S_FLOAT_2_SIGMA**: Scoring/aggression parameter
- Similar structure to A_FLOAT_2_SIGMA

**FLOAT_2_SIGMA**: Generic float parameter with uncertainty
- Used for: counter biases, player likelihoods

**BOOL_DELTA**: Boolean-valued parameter with delta distribution
- `expected`: probability (often 0.5 for symmetric)
- `delta`: shift from expected (learned from features)
- Example: draw factor

**PROBABILITY**: Scalar probability parameter
- `value`: transformed probability (often logit scale)
- Example: `player_started`

**CATEGORICAL**: Single category selection
- `value`: category string
- Example: player position ("FWD", "MID", "DEF", "GK")

**CATEGORICAL_DISTRIBUTION**: Distribution over categories
- `distribution`: {category: probability} dict
- Example: team formation (12 formations with probabilities)

**EPOCH**: Timestamp parameter
- Special type for time-based parameters

---

## Integration with Simulation System

### Pre-Match Workflow

1. **Match Created**: Mapping-lib creates match in calc-unit
   - `MatchXID` generated
   - `SportsEventRequest` posted (no parameters yet)

2. **IPM Initialization** (Ingest services):
   - Soccer-ingest, Tennis-ingest, etc. trigger IPM use case
   - Calls `initial_price_lib` to fetch features
   - Constructs `ParameterState` from Feature Stores

3. **Feature Store Batch Get**:
   - Determine event group type (domestic/cross/cup)
   - Build feature group map
   - Batch query SageMaker Feature Store
   - Parse responses → `Feature` objects

4. **Parameter State Construction**:
   - **Team Parameters**: A+S, counters, draw factor
   - **Player Parameters**: Likelihoods, positions, formations
   - **Supervisor Parameters**: Starting probabilities, formations

5. **Model Instance Creation**:
   - `ModelInstanceKey.generate(event_id=match_xid)` (in calc-unit)
   - Attached to `EventDetail.default_mik`
   - Used to generate unique prediction attribute keys (PAKs)

6. **Simulation Ready**:
   - Parameter state sent to simulation endpoint
   - Simulation samples from (μ, σ) distributions
   - Generates match trajectories

### Post-Match Workflow

1. **Simulation Completes**: Produces closing intensities
2. **Archiving Lambda**: Writes to raw S3
3. **Nightly ETL**: Processes → release tables
4. **Feature Jobs**: Update Feature Stores
5. **Next Day**: New matches use updated features

**Feedback Loop**: Simulation outputs → IPM training data → Better features → Better simulations

---

## Error Handling & Data Quality

### Missing Feature Data

**Strategies**:

1. **Default Values**: From `soccer_config.default_values.linear_models`
   - Used if Feature Store returns no record
   - Prevents simulation failure

2. **Latest Features**: Player props use last known values
   - If no recent data (player benched), use historical features
   - Prevents cold-start issues

3. **Validation Flags**: Team ratings flagged if unreliable
   - Downstream systems can apply more uncertainty
   - Or fall back to league-average ratings

### Data Filtering

**Time Windows**:
- Training data: last 365 days (configurable)
- Validation data: last 60 days
- Feature freshness: teams must have played in last 30 days

**Outlier Handling**:
- NaN-safe summation prevents bias from missing data
- Weighted mean downweights old/unreliable observations
- Exponential decay naturally handles irregular schedules

### Tier-Based Features

**Motivation**: Not all leagues have sufficient data for complex features

**Implementation**:
- Low Tier: Basic features only (e.g., corner bias)
- Mid Tier: + player started, positions, goals
- High Tier: + advanced counters, card types, goal characteristics

**Pipeline**: Attempts to calculate all features, but data will be missing for lower tiers
- Feature Store queries return None for unavailable features
- Parameter generation falls back to defaults

---

## Performance Considerations

### Sparse Matrix Efficiency

**Problem Size**:
- 1000 teams × 5000 matches → 5M design matrix elements
- Sparse representation: ~15K non-zero elements (0.3% density)

**Memory Savings**:
- Dense: 5M × 8 bytes = 40 MB per matrix
- Sparse: 15K × 16 bytes = 240 KB per matrix
- **~160x reduction**

### Batch Feature Store Queries

**AWS SageMaker Feature Store Runtime**:
- `batch_get_record`: retrieves up to 100 records per call
- Typical match: 2 teams, 22 players → ~25 feature keys
- Single batch call vs. 25 sequential calls
- **Latency reduction**: ~10x

### Nightly Processing Window

**Job Schedule**:
- ETL Jobs: 00:00 - 02:00 UTC (crawl + aggregate)
- Feature Jobs: 02:00 - 04:00 UTC (ratings + players)
- Result: Features ready by 04:00 for next day's matches

**Data Lag**:
- Matches completed today → features available tomorrow
- Acceptable lag (ratings don't change drastically overnight)

---

## Future Enhancements (Inferred)

### 1. Real-Time Feature Updates
- Currently: nightly batch
- Future: Stream processing on match completion
- Benefit: Fresher ratings for back-to-back matches

### 2. Transfer Learning for Low-Tier Leagues
- Currently: Missing data for low-tier counters
- Future: Borrow strength from similar higher-tier leagues
- Method: Hierarchical Bayesian models

### 3. Player Ratings (Not Just Props)
- Currently: Player likelihoods (raw stats)
- Future: Player skill ratings (relative strength)
- Analogy: Team A+S system for individual players

### 4. Contextual Features
- Currently: Match-agnostic features
- Future: Opponent-specific adjustments
- Example: Team plays more defensively vs. strong opponents

### 5. Uncertainty Quantification
- Currently: Validation flags (binary)
- Future: Confidence intervals on ratings
- Benefit: Propagate uncertainty to simulation parameters

---

## Summary

The **IPM (Initial Price Model) system** is a production-scale machine learning pipeline that:

1. **Processes** 10,000+ match simulations daily
2. **Calculates** team ratings, league strengths, player stats via sparse optimization
3. **Stores** features in 10+ AWS SageMaker Feature Stores
4. **Serves** features to real-time simulation services via batch APIs
5. **Updates** nightly via orchestrated Glue/Step Function jobs

**Key Algorithms**:
- **A+S Rating System**: Sparse least squares for team attack/defense/aggression
- **Cross-League Model**: League-level strength adjustments
- **Counter Bias Model**: Linear regression of counters on goal parameters
- **Weighted Averages**: Exponential decay, multi-scale transformations

**Design Principles**:
- **Sparsity**: Optimize for large-scale sparse linear systems
- **Recency Weighting**: Exponential decay favors recent matches
- **Hierarchical Keys**: Team-league, player-team composite keys
- **Graceful Degradation**: Default values, validation flags, tier-based features

**Integration**:
- IPM features initialize simulation parameters (μ, σ distributions)
- Simulation outputs feedback to IPM training (closing loop)
- End-to-end latency: match completion → feature update → next simulation < 24 hours

This system enables accurate, data-driven initialization of complex soccer match simulations with robust handling of missing data, diverse league characteristics, and evolving team/player performance.
