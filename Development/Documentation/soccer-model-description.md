# Soccer Monte Carlo Simulation Model - Technical Documentation

**Repository:** xalgo-sport-tribes  
**Module:** `applications/soccer-model-full/soccer_model_full/core_model/`  
**Date:** November 11, 2025

---

## Overview

This module implements a **soccer match Monte Carlo simulation engine** using TensorFlow. It simulates thousands of potential match outcomes by modeling the probability and timing of different events (goals, corners, cards, shots, fouls, etc.) throughout a 90-minute game.

The system combines:
- Neural networks trained on historical data
- Monte Carlo sampling for uncertainty quantification
- Player-level granularity for detailed predictions
- Time-based snapshots and interval aggregations for rich outputs

---

## Architecture

### Module Structure

```
core_model/
├── core_model.py      # Monte Carlo simulation engine
├── constants.py       # Event types, transformations, indices
├── player_flow.py     # Player-level event distribution
└── wrapper.py         # Integration & orchestration
```

---

## Component Details

### 1. `core_model.py` — The Monte Carlo Simulation Engine

#### **ModelLoader Class**

Loads pre-trained neural network models (`.h5` files) that have learned soccer match dynamics from historical data.

**Key responsibilities:**
- Loads a main **sequential model** that predicts event probabilities given game state
- Loads a **calibration model** that adjusts intensities to match real-world frequencies
- Constructs an **occurrence matrix** that maps model outputs to meaningful game events:
  - Total goals (regular + own goals + penalty goals)
  - Goals (excluding own goals)
  - Corners
  - Red cards (direct + from fouls)
  - All cards (yellow + red)

**Model attributes:**
- `version_major/minor`: Model version compatibility
- `game_size`: Number of time steps in simulation
- `constants`: Number of constant features
- `decay_list`: Decay factors for event influence over time
- `colnames`: Column names for model outputs
- `counter_names`: Event types tracked (e.g., 'goal', 'corner', 'yellowcard')
- `occurrence_matrix`: Matrix for aggregating events

#### **monte_carlo() Function**

The core simulation loop that generates one possible match outcome.

**Process:**

1. **Initialization**
   - Sets match state to time=0, score=0-0
   - Calculates time step size (typically ~2 seconds)
   - Prepares intensity vectors for both halves
   - Initializes decay factors and halftime markers

2. **Historical Event Processing**
   - If simulation starts mid-game (e.g., halftime or live):
     - Replays all past events
     - Updates internal state accordingly
     - Applies time-based decay

3. **Simulation Loop** (main forward simulation)
   - Steps through match in small time increments
   - For each time step:
     - **Predicts** event probabilities using neural network
     - **Samples** an event based on probabilities (Monte Carlo)
     - **Updates** game state:
       - Event counters
       - Score differentials
       - Decay factors (recent events influence future less)
     - **Records** snapshots at regular intervals (every 5 minutes)

4. **Output**
   - Snapshots showing match progression
   - Next occurrence probabilities
   - Observed state at current time
   - Max/min score differentials (max lead statistics)

**Key Parameters:**

- **`rox`**: Team intensity parameters (log-scale rates for each event type)
  - Adjusted for team strength
  - Shape: `(num_simulations, num_intensity_features)`
  
- **`halftime_intensity_deltas`**: Adjustments for second-half intensity changes
  - Models momentum shifts after halftime
  - Applied differently to first half (+0.625) vs second half (-0.5)

- **`injury_time`**: Added time in each half
  - Stochastic with mean ~3 minutes per half
  
- **`use_decay_scaling`**: Whether to apply decay scaling factor
  - When enabled, doubles the decay time constant
  - Affects how quickly past events fade in influence

- **`occurrence_history`**: Historical events (for live simulations)
  - One-hot encoded event types
  
- **`time_history`**: Timestamps of historical events (seconds)

- **`period_history`**: Period indicators (0=first half, 1=second half)

**Returns:**
```python
(
    snapshots,              # TensorArray of game state at intervals
    next_occurrence,        # Probabilities of next event by type
    observed_snapshot,      # State at current match time
    observed_occurrences,   # Aggregated current occurrences
    observed_k,            # Snapshot index at current time
    home_away_max_lead     # Maximum lead for each team
)
```

---

### 2. `constants.py` — Configuration & Transformations

#### **Event Types**

**TEAM_INTENSITIES** (20 features):
```python
[
    "goal_s",           # Home goal intensity (GOAL_S)
    "goal_a",           # Away goal intensity (GOAL_A)
    "owngoal_s",
    "owngoal_a",
    "corner_s",
    "corner_a",
    "shotsaved_s",
    "shotsaved_a",
    "shotofftarget_s",
    "shotofftarget_a",
    "blockedshot_s",
    "blockedshot_a",
    "foul_s",
    "foul_a",
    "offsidepass_s",
    "offsidepass_a",
    "yellowcard_s",
    "yellowcard_a",
    "redcard_s",
    "redcard_a",
]
```

**COUNTER_NAMES** (12 event types output by neural network):
```python
[
    "yellowfromfoul",    # Yellow card resulting from a foul
    "directyellow",      # Direct yellow card (no foul)
    "redfromfoul",       # Red card from a foul
    "directred",         # Direct red card
    "goal",              # Regular goal
    "owngoal",           # Own goal
    "corner",            # Corner kick
    "shotsaved",         # Shot on target saved by keeper
    "shotofftarget",     # Shot off target
    "blockedshot",       # Shot blocked by defender
    "onlyfoul",          # Foul without card
    "offsidepass",       # Offside
]
```

#### **GOAL_A and GOAL_S Explained**

- **GOAL_S**: Home team ("side" or "striker") goal intensity
- **GOAL_A**: Away team goal intensity

These parameters represent the **log-scale base rate** at which each team scores goals:
- Higher values = more likely to score
- Values are team-specific and learned from historical performance
- Adjusted by the calibration model to match real-world frequencies
- Modified by halftime deltas to model momentum changes

#### **Transformation Matrices**

**TRANSFORM_MATRIX**: Maps raw intensity inputs into neural network format
- Handles interactions between events (e.g., cards from fouls vs. direct cards)
- Separates home/away intensities
- Applies empirical coefficients learned from data

**Conversion Vectors**:
- `CARD_AND_FOUL_CONVERSION_VECTOR`: Splits card intensities into direct cards vs. cards from fouls
- `CARD_OR_FOUL_CONVERSION_VECTOR`: Handles foul-only events

**Conversion Factors**:
```python
DIRECT_RED_FROM_RED_FACTOR = 0.0515        # 5.15% of red cards are direct
DIRECT_YELLOW_FROM_YELLOW_FACTOR = 0.104   # 10.4% of yellows are direct
ONLY_FOUL_FROM_FOUL_FACTOR = 0.872         # 87.2% of fouls have no card
```

#### **Aggregation Indices**

Define how to combine raw counters into meaningful statistics:

- **Total Goals**: `home_goal + away_owngoal` (home's perspective)
- **Total Shots**: `goals + shots_saved + blocked_shots + shots_off_target`
- **Shots on Target**: `goals + shots_saved`
- **Yellow Cards**: `yellow_from_foul + direct_yellow`
- **Red Cards**: `red_from_foul + direct_red`
- **Total Fouls**: `only_foul + yellow_from_foul + red_from_foul`

#### **Player Feature Headers**

**Full Feature Set**:
```python
[
    "play_time",              # Expected minutes played
    "goal_likeliness",        # Log probability of scoring
    "shotontarget_likeliness",
    "shot_likeliness",
    "foul_likeliness",
    "yellowcard_likeliness",
    "redcard_likeliness",
    "header_bodypart",        # Probability of header
    "assist_likeliness",
    "outsidebox_location",    # Probability of long-range goal
]
```

**Goal-Only Feature Set** (for faster simulations):
```python
[
    "play_time",
    "goal_likeliness",
]
```

#### **Time Intervals**

**5-Minute Intervals**:
- First Half: 0-5, 5-10, 10-15, ..., 40-45 minutes
- Second Half: 45-50, 50-55, ..., 85-90 minutes

**15-Minute Intervals**:
- First Half: 0-15, 15-30, 30-45 minutes
- Second Half: 45-60, 60-75, 75-90 minutes

Used for aggregating events within time bins (e.g., "goals in first 15 minutes").

---

### 3. `player_flow.py` — Player-Level Simulation

Distributes team-level events to individual players based on their attributes.

#### **Key Functions**

**`expand_formations(formation_batch)`**
- Converts formation arrays (e.g., `[4, 3, 3]`) to position groups
- Example: `[4,3,3]` → `[0,0,0,0,1,1,1,2,2,2]` (4 defenders, 3 midfielders, 3 attackers)

**`sample_lineups(...)`**
- Samples which 10 players start based on:
  - Formation (e.g., 4-4-2, 3-5-2)
  - Player starting probabilities
  - Player positions (Defender, Midfielder, Attacker)
- Ensures no duplicate selections
- Handles substitutes (extra "sub" players with low probability)

**`restrict_features_to_lineups(...)`**
- Filters player features to only include selected players
- Adds synthetic "sub" players for events not attributable to specific players

**`likeliness_per_period(...)`**
- Adjusts player likelihoods based on playing time in each half
- Normalizes probabilities so they sum to ≤1.0
- Handles substitutions (fractional playing time)

**`sample_multinomial(counts, logits)`**
- Distributes event counts to players
- Uses categorical sampling based on logits (log probabilities)
- Returns player-level event counts

**`sample_joint_dist(counts, joint_distribution, ...)`**
- Samples events from joint distributions (e.g., goal scorer + assister)
- Handles dependencies between events
- Ensures valid combinations (e.g., no self-assists)

#### **Player Flow Modes**

**1. Full Player Flow (`player_flow`)**

Simulates all player-level events:
- Goals (with assists)
- Headers and outside-the-box goals
- Shots (on target, off target, blocked, saved)
- Fouls and cards (yellow, red, direct, from fouls)

**Process:**
1. Sample formations and lineups for both teams
2. Adjust player features for playing time
3. Create joint distribution for goals and assists:
   - Goal scorer probability × Assister probability
   - Remove self-assists
   - Separate assisted vs. unassisted goals
4. Sample headers and outside-the-box goals from goal events
5. Distribute shots, cards, and fouls using multinomial sampling
6. Aggregate to player-level counters
7. Apply validity mask (reject simulations where a player gets >3 cards)

**2. Goals-Only Player Flow (`player_flow_goals_only`)**

Simplified version that only simulates:
- Which player scored
- Next goal scorer (for live betting markets)

Used when detailed player stats aren't needed (faster execution).

#### **Player Outputs**

```python
{
    "home_player_counters": tf.Tensor,    # Shape: (n_sims, n_counters, n_players)
    "away_player_counters": tf.Tensor,
    "home_player_started": tf.Tensor,     # Which players started
    "away_player_started": tf.Tensor,
    "home_player_goal_next": tf.Tensor,   # Next goal scorer probabilities
    "away_player_goal_next": tf.Tensor,
    "weight": tf.Tensor,                  # Simulation validity weights
}
```

**Player Counters** (12 types for full flow, 1 for goals-only):
```python
[
    "player_goal_fulltime",
    "player_assist_fulltime",
    "player_header_fulltime",
    "player_outsidebox_fulltime",
    "player_shot_fulltime",
    "player_shotontarget_fulltime",
    "player_shotofftarget_fulltime",
    "player_blockedshot_fulltime",
    "player_shotsaved_fulltime",
    "player_foul_fulltime",
    "player_yellowcard_fulltime",
    "player_redcard_fulltime",
]
```

---

### 4. `wrapper.py` — Integration & Post-Processing

**`CoreModelWrapper`** orchestrates the entire simulation pipeline.

#### **Simulation Pipeline**

```
1. Parse Input
   ↓
2. Sample Team Intensities (PCA-based)
   ↓
3. Run Team-Level Monte Carlo Simulation
   ↓
4. Run Player-Level Simulation
   ↓
5. Aggregate & Post-Process
   ↓
6. Stream Results
```

#### **Key Methods**

**`predict(simulation_job, streaming_response, tier)`**
- Main entry point for predictions
- Calls `simulate()` and streams results back

**`simulate(simulation_job)`**
- Parses inputs (teams, players, formations, match state)
- Samples correlated team intensities
- Runs Monte Carlo simulation
- Distributes events to players
- Aggregates into output formats

**`parse_event_state(aggregated_state)`**
- Converts live match data into tensors:
  - Current period (first half, second half, etc.)
  - Current match time (seconds)
  - Historical events (occurrence types, times, periods)
  - Player goal history
- Encodes events using fixed indices
- Handles different match stages (prematch, first half, halftime, etc.)

**`parse_player_features(simulation_job)`**
- Extracts player features from simulation job:
  - Formations and formation distributions
  - Starting probabilities
  - Player positions (D, M, A, G, or special states like INJURED, SUSPENDED)
  - Fielded players (for live simulations with known lineups)

**`sample_fast(simulation_job)`**
- Samples team intensity parameters using PCA decomposition
- Applies shared (market-wide) and team-wise uncertainty
- Ensures correlated sampling across simulations

**`sample_woodwork(woodwork_logits, team_shotofftarget)`**
- Models shots hitting the post/crossbar
- Samples binomially from shots off target
- Adds realism to shot statistics

**`column_aggregation(count_tensor, axis)`**
- Aggregates raw event counts into derived statistics:
  - Total goals = goals + opponent own goals
  - Total shots = on-target + off-target + blocked
  - Cards = yellow + red (from all sources)

**`weighted_sample_var/std(x, w)`**
- Computes weighted statistics across simulations
- Downweights invalid simulations (e.g., excessive cards)

**`append_intervals(...)`**
- Aggregates events into time intervals
- Handles both 5-minute and 15-minute bins
- Separately processes first and second halves

#### **Output Formats**

**Integer Counters**:
- Fulltime and halftime counts for all event types
- Aggregated statistics (total goals, shots, cards, etc.)

**Boolean Indicators**:
- First to score
- First goal (excluding own goals)
- First corner, red card, card
- Next occurrence probabilities (Nth goal, corner, card)

**Interval Aggregations**:
- 5-minute intervals (18 total: 9 per half)
- 15-minute intervals (6 total: 3 per half)
- All event types aggregated within each interval

**Player Statistics**:
- Per-player counters (goals, assists, shots, cards, etc.)
- Starting lineups
- Next goal scorer probabilities
- Weighted by simulation validity

**Max Lead Statistics**:
- Maximum score differential for each team
- Tracked across all 5 occurrence types (total goals, goals, corners, red cards, cards)

---

## Data Flow Example

### Pre-Match Prediction

**Input:**
```python
SimulationJob(
    size=10000,  # 10k simulations
    parameter_state={
        "fast": [intensity parameters],  # 20 team intensities
        "sigma_shared": [...],           # Market-wide uncertainty
        "sigma_wise": [...],             # Team-specific uncertainty
    },
    supervisor_state={
        "supervisor_parameter": [
            FormationDistribution(home: {4-4-2: 0.6, 4-3-3: 0.4}),
            PlayerStartingProbability(player_id, prob),
            PlayerPosition(player_id, "M"),  # Midfielder
            ...
        ]
    },
    event_state={
        "match_state": None,  # Pre-match
        "entities": [HomeTeam, AwayTeam, Players...]
    }
)
```

**Processing:**
1. Sample 10k sets of team intensities (with correlation)
2. For each simulation:
   - Run `monte_carlo()` → team event counts
   - Run `player_flow()` → player event assignments
3. Aggregate across 10k simulations → probability distributions

**Output:**
```python
PredictionAttributes(
    "home_totalgoal_fulltime": Distribution(mean=1.8, std=1.2, prob_0=0.15, prob_1=0.30, ...),
    "away_totalgoal_fulltime": Distribution(mean=1.4, std=1.1, ...),
    "home_win": 0.48,
    "draw": 0.27,
    "away_win": 0.25,
    "player_X_goal_fulltime": Distribution(mean=0.35, ...),
    "home_first_goal": 0.52,
    ...
)
```

### Live (In-Play) Prediction

**Input:**
```python
SimulationJob(
    ...,
    event_state={
        "match_state": {
            "time_info": {
                "current_period": MatchStage.FIRST_HALF,
                "current_match_time": 23.5,  # 23:30 elapsed
            },
            "occurrences": {
                "occ1": Occurrence(
                    type="GOAL",
                    side=TwoSide.HOME,
                    time=12.3,
                    period=MatchStage.FIRST_HALF,
                    players={player_xid: 1.0}
                ),
                "occ2": Occurrence(
                    type="CORNER",
                    side=TwoSide.AWAY,
                    time=18.7,
                    ...
                ),
            }
        }
    }
)
```

**Processing:**
1. Parse historical events:
   - Period 0, Time 12.3 min: Home goal
   - Period 0, Time 18.7 min: Away corner
   - Current: Period 0, Time 23.5 min
2. Run `monte_carlo()` starting from current state:
   - Replay historical events to update state
   - Continue simulation from 23.5 min → 90+ min
3. Player flow assigns remaining events

**Output:**
```python
PredictionAttributes(
    "home_totalgoal_fulltime": Distribution(mean=2.3, ...),  # Conditional on 1-0 at 23:30
    "away_totalgoal_fulltime": Distribution(mean=1.1, ...),
    "home_win": 0.62,  # Updated win probabilities
    "next_goal_home": 0.54,  # Next goal probabilities
    ...
)
```

---

## Performance Characteristics

### Computational Complexity

**Team-Level Simulation:**
- Time complexity: O(n_sims × n_timesteps × n_features)
- Typical: 10,000 sims × 2,700 steps × 20 features ≈ 540M operations
- TensorFlow GPU acceleration: ~100-500ms per job

**Player-Level Simulation:**
- Additional: O(n_sims × n_events × n_players)
- Typical: 10,000 sims × 5 goals × 30 players ≈ 1.5M operations
- Adds ~50-200ms

**Total Prediction Time:**
- Pre-match: 200-700ms (full player flow)
- Live: 150-500ms (depends on time remaining)
- Goals-only mode: 100-300ms

### Memory Usage

- Model weights: ~50MB (neural network + calibration)
- Simulation tensors: ~100-500MB (depends on n_sims)
- GPU memory: 1-2GB recommended

### Accuracy

- Calibrated to historical match data
- Cross-validation R² ≈ 0.7-0.8 for goal totals
- Better for high-level aggregates (match result) than exact scores
- Player-level predictions less accurate (more variance)

---

## Configuration & Tuning

### Key Hyperparameters

**Simulation Settings:**
```python
n_simulations = 10000      # More sims → smoother distributions, slower
time_step_divisor = 60     # Divides period into steps (higher = finer resolution)
decay_scaling = True       # Apply 2x decay time constant
```

**Model Parameters:**
```python
game_size = 540            # Typical value for ~2 second steps
halftime_delta_factor_fh = 0.625   # First half intensity adjustment
halftime_delta_factor_sh = -0.5    # Second half intensity adjustment
```

**Player Flow Settings:**
```python
assist_probability = 0.65  # Fraction of goals with assists
header_prob_reduction = 4.0  # Log-scale reduction for headers on assists
```

### Environment Variables

```bash
OPTIMISTIC_MODE=1          # Allows simulations even if some player data is missing
```

---

## Known Limitations & Future Work

### Current Limitations

1. **No support for extra time/penalties**
   - Only models regulation 90 minutes
   - Could be extended with additional periods

2. **Fixed formation throughout match**
   - Doesn't model tactical changes mid-game
   - Formation distribution sampled once at start

3. **Simplified substitution model**
   - Players have fractional playing time
   - No explicit substitution events or timing

4. **Card accumulation constraints**
   - Rejects simulations where player gets >3 cards
   - Could be handled more elegantly

5. **Limited own goal modeling**
   - Own goals treated symmetrically with regular goals
   - Could benefit from separate intensity parameters

### Potential Improvements

1. **Dynamic formations**
   - Model formation changes based on score/time
   - Incorporate tactical adjustments

2. **Explicit substitution events**
   - Sample substitution times
   - Model impact on team intensity

3. **Player fatigue modeling**
   - Reduce effectiveness over time
   - Higher injury/card risk when tired

4. **Weather/venue effects**
   - Incorporate pitch conditions
   - Home advantage modeling

5. **Referee bias modeling**
   - Different card/foul thresholds
   - Simulation-specific referee sampling

6. **Set piece specialization**
   - Separate models for free kicks, penalties
   - Corner kick outcomes

---

## Technical Dependencies

### Core Libraries

```python
tensorflow >= 2.x          # Neural network engine
numpy >= 1.20             # Numerical operations
pandas                    # Data manipulation (wrapper)
h5py                      # Model file I/O
```

### Internal Dependencies

```python
run_control_plane         # Orchestration framework
xalgo_schemas            # Data schemas
soccer_schemas           # Soccer-specific schemas
```

### Hardware Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 8GB
- Storage: 1GB for models

**Recommended:**
- GPU: NVIDIA with 4GB+ VRAM (10-50x speedup)
- CPU: 8+ cores
- RAM: 16GB+
- Storage: 5GB+ (includes data)

---

## Usage Examples

### Basic Pre-Match Prediction

```python
from soccer_model_full.core_model import CoreModelWrapper

wrapper = CoreModelWrapper(model_path="models/soccer_v4.h5")

result, predictions, rename_dict, refs = wrapper.simulate(simulation_job)

# Access predictions
home_goals_dist = predictions["home_totalgoal_fulltime"]
print(f"Expected home goals: {home_goals_dist.mean}")
print(f"Home win probability: {predictions['home_win']}")
```

### Live Prediction with Historical Events

```python
# SimulationJob includes current match state
simulation_job.event_state.match_state = MatchState(
    time_info=TimeInfo(
        current_period=MatchStage.FIRST_HALF,
        current_match_time=35.0
    ),
    occurrences={
        "goal1": Occurrence(
            occurrence_type="GOAL",
            side=TwoSide.HOME,
            time=Minutes(12.5),
            period=MatchStage.FIRST_HALF
        )
    }
)

result, predictions, _, _ = wrapper.simulate(simulation_job)
print(f"Next goal probability (home): {predictions['next_goal_home']}")
```

### Player-Specific Predictions

```python
# Filter to player predictions
player_preds = {k: v for k, v in predictions.items() if "player_" in k}

for player_xid in home_squad:
    goal_key = f"player_{player_xid}_goal_fulltime"
    if goal_key in player_preds:
        print(f"Player {player_xid} goal expectation: {player_preds[goal_key].mean}")
```

---

## Appendix: Mathematical Formulation

### Event Intensity Model

The probability of event type `e` occurring in a small time interval `Δt` is:

```
P(event = e | state) = σ(f_θ(state) + λ_e) × Δt
```

Where:
- `σ()` is the sigmoid function
- `f_θ(state)` is the neural network output
- `λ_e` is the calibrated intensity for event type `e`
- `Δt` is the time step (~2 seconds)

### State Evolution

The internal state `x_t` evolves as:

```
x_{t+1} = x_t × exp(-δ × Δt) + ε_e
```

Where:
- `δ` is the decay rate (how quickly events fade)
- `ε_e` is the state increment for event `e`

### Halftime Adjustment

Intensity in second half:

```
λ_e^{2H} = λ_e^{1H} + Δλ_e^{HT}
```

Where `Δλ_e^{HT}` models momentum shifts.

### Player Event Distribution

Given team event count `n_e`, player `p` receives:

```
n_{e,p} ~ Multinomial(n_e, π_p)
```

Where:

```
π_p = softmax(log(L_{e,p}) + log(T_p))
```

- `L_{e,p}` is player's likeliness for event `e`
- `T_p` is player's playing time (0 to 1)

### Joint Goal-Assist Distribution

For goals with assists:

```
P(scorer=i, assister=j) ∝ G_i × A_j × (1 - δ_{ij}) × min(T_i, T_j)
```

Where:
- `G_i` is goal likeliness of player `i`
- `A_j` is assist likeliness of player `j`
- `δ_{ij}` removes self-assists
- Playing time uses minimum of scorer and assister

---

## Glossary

**Calibration**: Adjusting model outputs to match empirical frequencies  
**Decay**: Reduction in influence of past events over time  
**Halftime delta**: Change in team intensity after halftime  
**Intensity**: Log-scale rate parameter for event occurrence  
**Monte Carlo**: Simulation method using random sampling  
**Occurrence matrix**: Maps model outputs to game events  
**ROX**: Team intensity parameters (likely "Rate Of X" events)  
**Snapshot**: Saved game state at specific time point  
**Softplus**: Smooth approximation to ReLU activation  
**TensorArray**: TensorFlow data structure for variable-length sequences  

---

## Contact & Support

For questions or issues related to this module:
- Repository: sportsbetting-services/xalgo-sport-tribes
- Module: applications/soccer-model-full/
- Documentation Date: November 11, 2025

---

**End of Documentation**
