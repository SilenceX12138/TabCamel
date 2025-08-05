rm -rf build/
rm -rf dist/
rm -rf *.egg-info/
rm -rf .pytest_cache/
rm -rf .coverage
rm -rf htmlcov/
find . -type d -name __pycache__ -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
