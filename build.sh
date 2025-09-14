#!/bin/bash

# Build script for PyInstaller on macOS/Linux

echo "🔨 Building Side Project CLI with PyInstaller..."

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf build/
rm -rf dist/

# Build with PyInstaller
echo "🚀 Building executable..."
pyinstaller side-project.spec

# Check if build was successful
if [ -f "dist/side-project" ]; then
    echo "✅ Build successful!"
    echo "📁 Executable location: dist/side-project"
    echo ""
    echo "🔍 Testing executable..."
    ./dist/side-project --help
    echo ""
    echo "🎉 You can now run: ./dist/side-project"
else
    echo "❌ Build failed!"
    exit 1
fi
