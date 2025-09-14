#!/bin/bash
set -e

REPO="imagetec-laboratory/ocr-perfomance-test"
BINARY_NAME="opt-test"
BASE_URL="https://github.com/$REPO/releases/latest/download"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Installing OCR Performance Test...${NC}"

# Check if curl is available
if ! command -v curl >/dev/null 2>&1; then
    echo -e "${RED}❌ curl is required but not installed.${NC}"
    exit 1
fi

# Detect OS and Architecture
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

echo -e "${BLUE}🔍 Detected OS: $OS, Architecture: $ARCH${NC}"

# Determine filename based on OS and Architecture
case "$OS" in
    linux)
        case "$ARCH" in
            x86_64|amd64)
                FILENAME="side-project-linux-x64"
                ;;
            *)
                echo -e "${YELLOW}⚠️  Unsupported architecture: $ARCH for Linux${NC}"
                echo -e "${BLUE}Trying Linux x64 binary...${NC}"
                FILENAME="side-project-linux-x64"
                ;;
        esac
        ;;
    darwin)
        case "$ARCH" in
            arm64)
                FILENAME="side-project-macos-arm64"
                ;;
            x86_64|amd64)
                echo -e "${YELLOW}⚠️  Intel Mac detected, trying ARM64 binary (should work with Rosetta)...${NC}"
                FILENAME="side-project-macos-arm64"
                ;;
            *)
                echo -e "${YELLOW}⚠️  Unsupported architecture: $ARCH for macOS${NC}"
                echo -e "${BLUE}Trying macOS ARM64 binary...${NC}"
                FILENAME="side-project-macos-arm64"
                ;;
        esac
        ;;
    mingw*|cygwin*|msys*)
        case "$ARCH" in
            x86_64|amd64)
                FILENAME="side-project-windows-x64.exe"
                ;;
            *)
                echo -e "${YELLOW}⚠️  Unsupported architecture: $ARCH for Windows${NC}"
                echo -e "${BLUE}Trying Windows x64 binary...${NC}"
                FILENAME="side-project-windows-x64.exe"
                ;;
        esac
        ;;
    *)
        echo -e "${YELLOW}⚠️  Unsupported OS: $OS${NC}"
        echo -e "${BLUE}Trying Linux x64 binary...${NC}"
        FILENAME="side-project-linux-x64"
        ;;
esac

# Get latest version info
echo -e "${BLUE}📡 Fetching latest release info...${NC}"
LATEST_RELEASE=$(curl -s "https://api.github.com/repos/$REPO/releases/latest" 2>/dev/null) || {
    echo -e "${RED}❌ Failed to fetch release info. Please check your internet connection.${NC}"
    exit 1
}

VERSION=$(echo "$LATEST_RELEASE" | grep '"tag_name":' | cut -d '"' -f 4)
if [ -z "$VERSION" ]; then
    echo -e "${RED}❌ No releases found for this repository.${NC}"
    echo -e "${BLUE}💡 Please check: https://github.com/$REPO/releases${NC}"
    exit 1
fi

echo -e "${GREEN}📦 Latest version: $VERSION${NC}"

# Download URL
DOWNLOAD_URL="$BASE_URL/$FILENAME"

# Download the binary
echo -e "${BLUE}📥 Downloading $FILENAME...${NC}"
echo -e "${BLUE}🔗 URL: $DOWNLOAD_URL${NC}"

# Download
curl -L "$DOWNLOAD_URL" -o "$BINARY_NAME" || {
    echo -e "${RED}❌ Download failed!${NC}"
    echo -e "${BLUE}💡 Available files at: https://github.com/$REPO/releases/latest${NC}"
    exit 1
}

# Verify file size (should be > 1MB for a real binary)
FILE_SIZE=$(stat -f%z "$BINARY_NAME" 2>/dev/null || stat -c%s "$BINARY_NAME" 2>/dev/null || echo "0")
if [ "$FILE_SIZE" -lt 1000000 ]; then
    echo -e "${RED}❌ Downloaded file seems too small ($FILE_SIZE bytes). Something went wrong.${NC}"
    rm -f "$BINARY_NAME"
    exit 1
fi

# Make executable (Unix systems)
if [ "$OS" != "mingw"* ] && [ "$OS" != "cygwin"* ] && [ "$OS" != "msys"* ]; then
    chmod +x "$BINARY_NAME"
fi

# Success message
echo -e "${GREEN}✅ Installation complete!${NC}"
echo -e "${GREEN}📁 Binary saved as: ./$BINARY_NAME (${FILE_SIZE} bytes)${NC}"
echo -e "${YELLOW}🎉 Run with: ./$BINARY_NAME${NC}"

# Optional: show version if binary supports it
echo -e "${BLUE}📋 Testing installation...${NC}"
if ./"$BINARY_NAME" --version 2>/dev/null; then
    echo -e "${GREEN}✅ Installation verified!${NC}"
elif ./"$BINARY_NAME" --help >/dev/null 2>&1; then
    echo -e "${GREEN}✅ Installation verified!${NC}"
else
    echo -e "${YELLOW}⚠️  Binary installed but version check failed (this might be normal)${NC}"
fi

echo -e "${BLUE}🔗 For more info: https://github.com/$REPO${NC}"