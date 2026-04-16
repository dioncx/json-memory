#!/bin/bash
# Publish json-memory to PyPI
# Usage: ./publish.sh

set -e

echo "📦 Building json-memory for PyPI..."

# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build
python3 -m pip install --upgrade build twine -q
python3 -m build

echo ""
echo "📋 Built files:"
ls -la dist/

echo ""
echo "📤 Upload to TestPyPI first (safe):"
echo "   python3 -m twine upload --repository testpypi dist/*"
echo ""
echo "📤 Upload to PyPI (production):"
echo "   python3 -m twine upload dist/*"
echo ""
echo "🔑 You'll need a PyPI account:"
echo "   1. Create account: https://pypi.org/account/register/"
echo "   2. Create API token: https://pypi.org/manage/account/token/"
echo "   3. Save in ~/.pypirc or enter when prompted"
echo ""
echo "✅ After upload, users can: pip install json-memory"
