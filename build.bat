@echo off

REM Build script for PyInstaller on Windows

echo 🔨 Building Side Project CLI with PyInstaller...

REM Clean previous builds
echo 🧹 Cleaning previous builds...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

REM Build with PyInstaller
echo 🚀 Building executable...
pyinstaller side-project.spec

REM Check if build was successful
if exist "dist\side-project.exe" (
    echo ✅ Build successful!
    echo 📁 Executable location: dist\side-project.exe
    echo.
    echo 🔍 Testing executable...
    dist\side-project.exe --help
    echo.
    echo 🎉 You can now run: dist\side-project.exe
) else (
    echo ❌ Build failed!
    exit /b 1
)
