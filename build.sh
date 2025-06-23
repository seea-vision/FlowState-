#!/usr/bin/env bash
set -e

echo "🔧 Installing setuptools and wheel..."
pip install setuptools==68.0.0 wheel

echo "📦 Installing requirements..."
pip install -r requirements.txt

echo "✅ Build complete"
