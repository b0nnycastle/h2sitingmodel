# Contributing to H2 Station Siting Model

We welcome contributions to the H2 Station Siting Model! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/h2-station-siting.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Test your changes
6. Commit with a clear message: `git commit -m "Add feature: description"`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Create a Pull Request

## Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/h2-station-siting.git
cd h2-station-siting

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Check code style
black --check .
flake8



Code Style

Follow PEP 8
Use Black for code formatting
Add type hints where appropriate
Document all public functions and classes
Keep line length to 88 characters (Black default)

Testing

Write tests for new functionality
Ensure all tests pass before submitting PR
Aim for >80% code coverage
Use pytest for testing

Example test:

def test_demand_estimation():
    model = H2StationSitingModel()
    # Test implementation
    assert model.estimate_demand() > 0

Documentation

Update docstrings for new/modified functions
Update README.md if adding new features
Add examples for new functionality
Update data dictionary for new fields

Pull Request Process

Update documentation
Add tests for new functionality
Ensure all tests pass
Update CHANGELOG.md
Create descriptive PR with:

Summary of changes
Related issue numbers
Testing performed
Screenshots (if applicable)



Reporting Issues

Use GitHub Issues
Include minimal reproducible example
Specify Python version and OS
Include full error traceback
Attach sample data if relevant

Code of Conduct

Be respectful and inclusive
Welcome newcomers
Focus on constructive criticism
Respect differing viewpoints

Thank you for contributing!
