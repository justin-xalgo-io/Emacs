# xalgo-sport-tribes Development Setup Guide

This guide provides comprehensive instructions for setting up the xalgo-sport-tribes development environment, including solutions to common issues.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Initial Setup](#initial-setup)
- [Common Issues and Solutions](#common-issues-and-solutions)
- [Development Workflow](#development-workflow)
- [IDE Configuration](#ide-configuration)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Software
- **Python 3.11+** (installed via Homebrew recommended)
- **AWS CLI** configured with `xalgo_admin_development` profile
- **Pants build system** - Follow [Pants installation guide](https://www.pantsbuild.org/dev/docs/getting-started/installing-pants)

### Python Installation (Homebrew)
```bash
# Install Python 3.11 via Homebrew
brew install python@3.11

# Verify installation
/opt/homebrew/bin/python3.11 --version
# Should output: Python 3.11.x
```

### AWS Configuration
Ensure you have the correct AWS profile:
```bash
aws configure list-profiles | grep xalgo_admin_development
```

## Initial Setup

### 1. Quick Setup (Recommended)
Run the automated setup script:
```bash
./3rdparty/python/lock.sh
```

### 2. Manual Setup (If Quick Setup Fails)

#### Step 2.1: AWS Authentication
```bash
export ARTIFACTORY_TOKEN=$(aws --profile xalgo_admin_development codeartifact get-authorization-token --domain xalgo --domain-owner 532154079145 --query authorizationToken --output text)
```

#### Step 2.2: Generate Lockfiles
```bash
# Set Python interpreter constraints to avoid discovery issues
export PANTS_PYTHON_INTERPRETER_CONSTRAINTS='["CPython==3.11.*"]'
export PY=/opt/homebrew/bin/python3.11

# Generate lockfiles (this may take several minutes)
pants generate-lockfiles
```

#### Step 2.3: Export Virtual Environment
```bash
pants export
```

#### Step 2.4: Create Environment Configuration
```bash
echo "PYTHONPATH=applications:components:projects:tests:development" > .env
```

### 3. Activate Development Environment
```bash
# Load environment variables
source .env

# Activate Python virtual environment
source dist/export/python/virtualenvs/python-default/3.11.13/bin/activate
```

## Common Issues and Solutions

### Issue 1: Pants Stuck on "Find interpreter" for Hours

**Problem**: Pants hangs indefinitely while trying to find a Python interpreter.

**Root Cause**: 
- Pyenv shims interfering with interpreter discovery
- Complex search paths causing delays
- Broad interpreter constraints making Pants search extensively

**Solution**:
1. **Update pants.toml** to simplify the search path:
```toml
[python-bootstrap]
search_path = [
    # Force Homebrew Python first
    "/opt/homebrew/bin",
]
```

2. **Use explicit environment variables**:
```bash
export PANTS_PYTHON_INTERPRETER_CONSTRAINTS='["CPython==3.11.*"]'
export PY=/opt/homebrew/bin/python3.11
```

3. **Clear Pants cache if needed**:
```bash
rm -rf .pants.d
```

**Explanation**: This forces Pants to use your Homebrew Python directly instead of searching through pyenv shims and multiple directories.

### Issue 2: AWS CodeArtifact Authentication Errors

**Problem**: `Could not connect to the endpoint URL` errors.

**Solution**:
1. Ensure AWS SSO is logged in:
```bash
aws sso login --profile xalgo_admin_development
```

2. Verify profile configuration:
```bash
aws configure list --profile xalgo_admin_development
```

### Issue 3: Invalid .env File Causing Pants Errors

**Problem**: Pants fails with parsing errors related to .env file.

**Solution**:
1. Remove the problematic .env file:
```bash
rm -f .env
```

2. Create a clean .env file:
```bash
echo "PYTHONPATH=applications:components:projects:tests:development" > .env
```

**Explanation**: The `pants roots` command sometimes includes bootstrap messages that break the .env file format.

### Issue 4: Interpreter Constraints Mismatch

**Problem**: Tests fail with `InvalidLockfileError` about incompatible interpreter constraints.

**Error Message**:
```
InvalidLockfileError: You are consuming requirements from the `python-default` lockfile 
with incompatible inputs.
- The inputs use interpreter constraints (`CPython<3.13,>=3.11`) that are not a subset 
  of those used to generate the lockfile (`CPython==3.11.*`).
```

**Solution**:
1. Update `pants.toml` to match the lockfile constraints:
```toml
[python]
enable_resolves = true
interpreter_constraints = ["CPython==3.11.*"]
```

2. If you need to change constraints, regenerate the lockfile:
```bash
export PANTS_PYTHON_INTERPRETER_CONSTRAINTS='["CPython==3.11.*"]'
export PY=/opt/homebrew/bin/python3.11
pants generate-lockfiles --resolve=python-default
```

**Explanation**: The lockfile is generated with specific Python interpreter constraints. If your `pants.toml` has different constraints, Pants will refuse to use the lockfile to prevent compatibility issues.

### Issue 5: Permission or Path Issues

**Problem**: Python interpreter not found despite being installed.

**Solution**:
1. Verify Homebrew Python installation:
```bash
ls -la /opt/homebrew/bin/python*
/opt/homebrew/bin/python3.11 --version
```

2. Update PATH if necessary:
```bash
export PATH="/opt/homebrew/bin:$PATH"
```

### Issue 6: Pants Dependency Inference Warnings

**Problem**: Warnings about imports that Pants cannot infer owners for, or ambiguous dependencies.

**Example Warning**:
```
[WARN] Pants cannot infer owners for the following imports in the target:
  * tests.conftest.TestServer (line: 8)
```

**Solutions**:
1. **For expected uninferable imports**, add a comment to ignore:
```python
from tests.conftest import TestServer  # pants: no-infer-dep
```

2. **For ambiguous imports** (multiple targets own the same module), explicitly add dependency in BUILD file:
```python
python_test(
    name="tests",
    sources=["test_*.py"],
    dependencies=[
        "path/to/specific/conftest.py:target",
    ],
)
```

3. **For test utilities**, consider creating a shared test library target.

**Explanation**: Pants tries to automatically infer dependencies from imports, but sometimes it can't determine which target owns a module, especially with test utilities that exist in multiple places.

## Development Workflow

### Daily Development Commands

#### Environment Activation
```bash
# Always run these before starting development
source .env
source dist/export/python/virtualenvs/python-default/3.11.13/bin/activate
```

#### Code Quality
```bash
# Format code
pants fmt ::

# Type checking
pants check ::

# Linting
pants lint ::

# Run all code quality checks
pants fmt :: && pants lint :: && pants check ::
```

#### Testing
```bash
# Run all tests (may take a long time)
pants test ::

# Run tests for specific component/application
pants test applications/feed-booking-handler/
pants test components/soccer_commons/

# Run a specific test file
pants test applications/feed-booking-handler/tests/test_handler.py:../tests

# Run tests without coverage (faster)
pants test :: --no-test-use-coverage

# Find all test targets
pants --filter-target-type=python_test list ::

# Run tests in parallel (default, but can be controlled)
pants test :: --test-processes=4
```

#### Building and Running
```bash
# Build a specific application
pants package applications/soccer-model-full:

# Run a specific application
pants run applications/soccer-model-full:

# List all available targets
pants list :: | head -20
```

### Working with Specific Components

#### Applications (Standalone Services)
- **Feed Integrations**: `applications/betradar-feed-integration/`, `applications/opta-feed-integration/`
- **Sport Models**: `applications/soccer-model-full/`, `applications/tennis-ingest/`
- **Infrastructure**: `applications/feed-booking-handler/`, `applications/feed-message-store/`

#### Components (Reusable Libraries)
- **Sport Logic**: `components/soccer_commons/`, `components/tennis-schemas/`
- **Infrastructure**: `components/mapping-lib/`, `components/psql_client/`
- **Configuration**: `components/soccer_config/`, `components/ice-hockey-config/`

#### Projects (Combined Applications)
- **Infrastructure**: `projects/xalgo_tribe_infrastucture/`
- **Sport Gateways**: `projects/xalgo_soccer_sport_gateway/`

## IDE Configuration

### VS Code Setup
1. Install Python extension
2. Set Python interpreter to the exported virtual environment:
   ```
   dist/export/python/virtualenvs/python-default/3.11.13/bin/python
   ```
3. Ensure PYTHONPATH is configured (from .env file)

### PyCharm Setup
1. Configure Project Interpreter to use the exported virtual environment
2. Add source roots: applications, components, projects, tests, development
3. Enable Pants integration if available

## Troubleshooting

### Environment Variables for Consistent Builds
If you encounter interpreter issues, always set these before running Pants commands:
```bash
export PANTS_PYTHON_INTERPRETER_CONSTRAINTS='["CPython==3.11.*"]'
export PY=/opt/homebrew/bin/python3.11
```

### Regenerating Environment
If your environment gets corrupted:
```bash
# Clean everything
rm -rf .pants.d dist/export .env

# Re-run setup
./3rdparty/python/lock.sh
```

### Debugging Pants Issues
```bash
# Run with debug logging
pants --level=debug <command>

# Check Pants version
pants --version

# Validate configuration
pants help python-bootstrap
```

### Performance Tips
1. **Use Pants daemon**: Pants runs faster with the daemon enabled (default)
2. **Limit test scope**: Use specific paths instead of `::` when possible
3. **Parallel execution**: Pants automatically parallelizes compatible tasks

## Repository Architecture

### Key Concepts
- **Platform Domain**: Core infrastructure (persistence, orchestration)
- **Sport Domain (Tribe)**: Sport-specific logic and models
- **Polylith Architecture**: Components are reusable across applications

### Data Flow
1. **Fixture Integration**: Fetch match data from providers
2. **Feed Integration**: Consume live match events  
3. **Ingest**: Process feed data and update match state
4. **Run**: Execute sports models and generate predictions
5. **Trading**: Monitor betting activity and adjust parameters

### Technology Stack
- **Build System**: Pants 2.26.0
- **Python**: 3.11+ with strong typing (mypy, pyright)
- **Code Quality**: Black, flake8, isort, bandit
- **Infrastructure**: AWS Lambda, Docker, Kinesis, DynamoDB
- **Sports Data**: Multiple providers (Betradar, Opta, RunningBall)

## Getting Help

### Internal Resources
- Repository README.md for architecture overview
- Individual component/application README files
- Development team Slack channels

### External Resources
- [Pants Documentation](https://www.pantsbuild.org/docs)
- [Python Type Checking](https://mypy.readthedocs.io/)
- [AWS CodeArtifact](https://docs.aws.amazon.com/codeartifact/)

## Quick Reference

### Essential Commands
```bash
# Setup
./3rdparty/python/lock.sh

# Activate environment
source .env && source dist/export/python/virtualenvs/python-default/3.11.13/bin/activate

# Development workflow
pants fmt :: && pants lint :: && pants check :: && pants test ::

# Build and run
pants package <target>
pants run <target>
```

### File Structure
```
├── applications/          # Standalone services
├── components/           # Reusable components  
├── projects/            # Combined applications
├── development/         # Development tools
├── tests/              # Test files
├── 3rdparty/           # Third-party dependencies
└── pants.toml          # Build configuration
```

---

*Last updated: September 30, 2025*
*For questions or issues, contact the development team or create an issue in the repository.*