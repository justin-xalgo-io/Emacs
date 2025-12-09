# DataFrame Manipulation Guide

A comprehensive guide to pandas DataFrame operations learned on December 9, 2025.

---

## Table of Contents
1. [Removing Duplicate Rows](#removing-duplicate-rows)
2. [Filtering Rows](#filtering-rows)
3. [Selecting Columns](#selecting-columns)
4. [String Manipulation](#string-manipulation)
5. [DynamoDB Integration](#dynamodb-integration)
6. [Parsing Malformed Strings](#parsing-malformed-strings)
7. [Adding and Populating New Columns](#adding-and-populating-new-columns)

---

## Removing Duplicate Rows

### Drop duplicates based on column values

```python
# Keep first occurrence (default)
df = df.drop_duplicates(subset=['column_name'])

# Keep last occurrence
df = df.drop_duplicates(subset=['column_name'], keep='last')

# Remove all duplicates (keep none)
df = df.drop_duplicates(subset=['column_name'], keep=False)

# Multiple columns
df = df.drop_duplicates(subset=['col1', 'col2', 'col3'])

# In-place modification
df.drop_duplicates(subset=['column_name'], inplace=True)
```

### Using boolean masks

```python
# Keep first
df = df[~df.duplicated(subset=['column_name'], keep='first')]

# Keep last
df = df[~df.duplicated(subset=['column_name'], keep='last')]

# Remove all duplicates
df = df[~df.duplicated(subset=['column_name'], keep=False)]
```

### Keep row with specific criteria

```python
# Keep row with highest timestamp for each duplicate group
idx = df.groupby(['col1', 'col2'])['timestamp'].idxmax()
df = df.loc[idx].reset_index(drop=True)
```

---

## Filtering Rows

### Remove rows where column equals a value

```python
# Not equal to specific value
df = df[df['column_name'] != specific_value]

# Remove rows where column is in a list
df = df[~df['column_name'].isin(['value1', 'value2', 'value3'])]

# Multiple conditions
df = df[(df['status'] != 'cancelled') & (df['amount'] != 0)]
```

### Remove rows with None/NaN values

```python
# Remove rows where column is None/NaN
df = df[df['column_name'].notna()]

# Alternative using dropna
df = df.dropna(subset=['column_name'])

# Multiple columns
df = df[df[['col1', 'col2', 'col3']].notna().all(axis=1)]

# In-place
df.dropna(subset=['column_name'], inplace=True)
```

### Keep only rows matching criteria

```python
# Keep rows where column is in specific values
df = df[df['current_game_state'].isin(['ENDED', 'ABORTED'])]
```

---

## Selecting Columns

### Keep only specific columns

```python
# Keep specific columns
df = df[['column1', 'column2', 'column3']]

# Using filter
df = df.filter(['column1', 'column2', 'column3'])

# Using loc
df = df.loc[:, ['column1', 'column2', 'column3']]
```

### Drop all columns except specified ones

```python
columns_to_keep = ['column1', 'column2', 'column3']
df = df[columns_to_keep]

# Or drop specific columns
df = df.drop(columns=['col_to_remove1', 'col_to_remove2'])
```

---

## String Manipulation

### Apply operations to DataFrame cells

```python
# Modern pandas (>= 2.1.0) - use map instead of applymap
quoted = df.map(lambda s: "'" + str(s).replace("'", "''") + "'")

# For older pandas (< 2.1.0)
quoted = df.applymap(lambda s: "'" + str(s).replace("'", "''") + "'")
```

### Aggregate columns into formatted strings

```python
# Create tuple-like strings with comma at end
df["values_tuple"] = quoted.agg(",".join, axis=1).apply(lambda s: f"({s}),")

# Remove comma from last row
df.loc[df.index[-1], "values_tuple"] = df.loc[df.index[-1], "values_tuple"].rstrip(",")

# Alternative: add comma to all but last
df["values_tuple"] = quoted.agg(",".join, axis=1).apply(lambda s: f"({s})")
df.loc[df.index[:-1], "values_tuple"] = df.loc[df.index[:-1], "values_tuple"] + ","
```

### Join list to string with newlines

```python
# Join with newlines between elements (no trailing newline)
values_list = df["values_tuple"].astype(str).tolist()
tuples_only_str = "\n".join(values_list)

# With trailing newline
tuples_only_str = "\n".join(values_list) + "\n"
```

---

## DynamoDB Integration

### Query DynamoDB and convert to DataFrame

```python
import boto3
import pandas as pd
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError

def query_dynamodb_to_dataframe(table_name: str, date_prefix: str, profile_name: str = 'production') -> pd.DataFrame:
    """
    Scan DynamoDB by date_prefix and return as DataFrame.
    
    Args:
        table_name: DynamoDB table name
        date_prefix: Value to filter by
        profile_name: AWS profile name (e.g., 'xalgo_admin_production')
    
    Returns:
        DataFrame with all matching items
    """
    # Use specific AWS profile
    session = boto3.Session(profile_name=profile_name)
    dynamodb = session.resource('dynamodb', region_name='eu-west-1')
    table = dynamodb.Table(table_name)
    
    items = []
    
    try:
        response = table.scan(
            FilterExpression=Key('date_prefix').eq(date_prefix)
        )
        items.extend(response['Items'])
        
        # Paginate through all results
        while 'LastEvaluatedKey' in response:
            response = table.scan(
                FilterExpression=Key('date_prefix').eq(date_prefix),
                ExclusiveStartKey=response['LastEvaluatedKey']
            )
            items.extend(response['Items'])
        
        return pd.DataFrame(items)
    
    except ClientError as e:
        print(f"Error scanning DynamoDB: {e.response['Error']['Message']}")
        raise

# Usage
df = query_dynamodb_to_dataframe(
    table_name='your-table-name',
    date_prefix='2025-12-09',
    profile_name='xalgo_admin_production'
)
```

### Verify AWS Profile

```bash
# List available profiles
aws configure list-profiles

# Or in Python
import boto3
session = boto3.Session()
profiles = session.available_profiles
print(profiles)

# Verify which account a profile points to
aws sts get-caller-identity --profile xalgo_admin_production
```

---

## Parsing Malformed Strings

### Convert string representation to actual data structures

```python
import json
import ast

# For valid JSON strings (double quotes)
string_data = '[{"key1": "value1"}, {"key2": "value2"}]'
array = json.loads(string_data)

# For Python literal strings (single quotes)
string_data = "[{'key1': 'value1'}, {'key2': 'value2'}]"
array = ast.literal_eval(string_data)

# Try JSON first, fallback to ast
try:
    result = json.loads(string_data)
except json.JSONDecodeError:
    result = ast.literal_eval(string_data)
```

### Fix malformed strings with unquoted values

Common issues:
- `NULL` instead of `None`
- Unquoted enum values like `TEAM`, `PARTICIPANT`
- Unquoted text like `Petra Holesinska` or `ČEZ Basket Nymburk`

```python
import re

def fix_malformed_string(s):
    """Fix common malformed string issues before parsing."""
    
    # Fix NULL -> None
    s = s.replace('NULL', 'None')
    
    # Fix booleans
    s = s.replace('true', 'True')
    s = s.replace('false', 'False')
    
    # Fix unquoted enums (TEAM, PARTICIPANT, etc.)
    s = re.sub(r":\s*(TEAM|PARTICIPANT)([,}\]])", r": '\1'\2", s)
    
    # Fix unquoted text after specific keys
    # Handles Unicode characters, spaces, special chars
    # Stops at quotes, braces, commas
    s = re.sub(
        r"('(?:defaultText|firstName|lastName)':\s*)([^'}\],][^'}\],]*?)([,}\]])",
        r"\1'\2'\3",
        s
    )
    
    return s

# Usage
malformed = "[{'participantType': TEAM, 'defaultText': ČEZ Basket Nymburk}]"
fixed = fix_malformed_string(malformed)
data = ast.literal_eval(fixed)
```

**Handles:**
- ✅ `NULL` → `None`
- ✅ `true/false` → `True/False`
- ✅ `TEAM` → `'TEAM'`
- ✅ `PARTICIPANT` → `'PARTICIPANT'`
- ✅ `ČEZ Basket Nymburk` → `'ČEZ Basket Nymburk'`
- ✅ `Levickí Patrioti` → `'Levickí Patrioti'`
- ✅ `Petra Holesinska` → `'Petra Holesinska'`

---

## Adding and Populating New Columns

### Add new columns to DataFrame

```python
# Add empty columns
df['column1'] = None  # or '' for string, 0 for numeric
df['column2'] = None
df['column3'] = None
df['column4'] = None
```

### Iterate and populate (simple approach)

```python
import ast

for index, row in df.iterrows():
    # Parse data
    participants_str = row['participants']
    fixed = fix_malformed_string(participants_str)
    participants = ast.literal_eval(fixed)
    
    # Extract and assign data
    if len(participants) > 0:
        df.at[index, 'home_team_name'] = participants[0].get('participantName', {}).get('defaultName', {}).get('defaultText', None)
        df.at[index, 'home_team_id'] = participants[0].get('participantId', None)
    
    if len(participants) > 1:
        df.at[index, 'away_team_name'] = participants[1].get('participantName', {}).get('defaultName', {}).get('defaultText', None)
        df.at[index, 'away_team_id'] = participants[1].get('participantId', None)
```

### More efficient using apply() (recommended)

```python
import ast
import pandas as pd

def parse_participants(participants_str):
    """Parse participants string and extract team data."""
    try:
        fixed = fix_malformed_string(participants_str)
        participants = ast.literal_eval(fixed)
        
        # Find home and away teams
        home = next((p for p in participants if p.get('isHome') == True), {})
        away = next((p for p in participants if p.get('isHome') == False or p.get('isHome') is None), {})
        
        return pd.Series({
            'home_team_name': home.get('participantName', {}).get('defaultName', {}).get('defaultText'),
            'home_team_id': home.get('participantId'),
            'away_team_name': away.get('participantName', {}).get('defaultName', {}).get('defaultText'),
            'away_team_id': away.get('participantId'),
        })
    except Exception as e:
        return pd.Series({
            'home_team_name': None,
            'home_team_id': None,
            'away_team_name': None,
            'away_team_id': None,
        })

# Add new columns from apply
df[['home_team_name', 'home_team_id', 'away_team_name', 'away_team_id']] = df['participants'].apply(parse_participants)
```

### Alternative: extract to dict then concat

```python
def extract_participant_data(participants_str):
    """Extract data and return as dict."""
    try:
        fixed = fix_malformed_string(participants_str)
        participants = ast.literal_eval(fixed)
        
        return {
            'home_team': participants[0].get('participantName', {}).get('defaultName', {}).get('defaultText', None) if len(participants) > 0 else None,
            'home_team_id': participants[0].get('participantId', None) if len(participants) > 0 else None,
            'away_team': participants[1].get('participantName', {}).get('defaultName', {}).get('defaultText', None) if len(participants) > 1 else None,
            'away_team_id': participants[1].get('participantId', None) if len(participants) > 1 else None,
        }
    except:
        return {'home_team': None, 'home_team_id': None, 'away_team': None, 'away_team_id': None}

# Apply and expand to new columns
new_cols = df['participants'].apply(extract_participant_data).apply(pd.Series)
df = pd.concat([df, new_cols], axis=1)
```

---

## Common Workflow Example

```python
import boto3
import pandas as pd
import ast
import re
from boto3.dynamodb.conditions import Key

# 1. Query DynamoDB
def query_dynamodb_to_dataframe(table_name, date_prefix, profile_name='production'):
    session = boto3.Session(profile_name=profile_name)
    dynamodb = session.resource('dynamodb', region_name='eu-west-1')
    table = dynamodb.Table(table_name)
    
    items = []
    response = table.scan(FilterExpression=Key('date_prefix').eq(date_prefix))
    items.extend(response['Items'])
    
    while 'LastEvaluatedKey' in response:
        response = table.scan(
            FilterExpression=Key('date_prefix').eq(date_prefix),
            ExclusiveStartKey=response['LastEvaluatedKey']
        )
        items.extend(response['Items'])
    
    return pd.DataFrame(items)

# 2. Fix malformed strings
def fix_malformed_string(s):
    s = s.replace('NULL', 'None')
    s = s.replace('true', 'True')
    s = s.replace('false', 'False')
    s = re.sub(r":\s*(TEAM|PARTICIPANT)([,}\]])", r": '\1'\2", s)
    s = re.sub(
        r"('(?:defaultText|firstName|lastName)':\s*)([^'}\],][^'}\],]*?)([,}\]])",
        r"\1'\2'\3",
        s
    )
    return s

# 3. Load data
df = query_dynamodb_to_dataframe(
    table_name='match-statistics',
    date_prefix='2025-11',
    profile_name='xalgo_admin_production'
)

# 4. Filter data
df = df.dropna(subset=['instance_key', 'data_provider', 'current_game_state'])
df = df[df['current_game_state'].isin(['ENDED', 'ABORTED'])]
df = df[df['data_provider'] != 'MANUAL']

# 5. Remove duplicates
df = df.drop_duplicates(subset=['instance_key'], keep='first')

# 6. Parse and extract nested data
def parse_participants(participants_str):
    try:
        fixed = fix_malformed_string(participants_str)
        participants = ast.literal_eval(fixed)
        
        home = next((p for p in participants if p.get('isHome') == True), {})
        away = next((p for p in participants if not p.get('isHome')), {})
        
        return pd.Series({
            'home_team_name': home.get('participantName', {}).get('defaultName', {}).get('defaultText'),
            'away_team_name': away.get('participantName', {}).get('defaultName', {}).get('defaultText'),
        })
    except:
        return pd.Series({'home_team_name': None, 'away_team_name': None})

df[['home_team_name', 'away_team_name']] = df['participants'].apply(parse_participants)

# 7. Select final columns
df = df[['instance_key', 'home_team_name', 'away_team_name', 'current_game_state']]

print(df.head())
```

---

## Requirements.txt

For projects using these techniques:

```txt
boto3>=1.34.0
psycopg>=3.1.0
pandas>=2.1.0
```

---

## Tips and Best Practices

1. **Use `apply()` over `iterrows()`** for better performance
2. **Always handle exceptions** when parsing external data
3. **Use AWS profiles** to ensure you're querying the right environment
4. **Test regex patterns** on sample data before applying to full dataset
5. **Use `notna()` instead of checking for None** to catch both None and NaN
6. **Keep column selection** at the end after all transformations
7. **Use `drop_duplicates()` early** to reduce data processing
8. **Reset index** after filtering: `df.reset_index(drop=True)`

---

Generated: December 9, 2025
