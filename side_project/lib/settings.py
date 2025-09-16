"""
Settings management with cross-platform support

🔧 Features:
- Cross-platform config directory detection
- Basic error handling and fallbacks
- JSON-based configuration storage
"""

import json
import os
import platform
from pathlib import Path
from typing import Dict, Any

from side_project.config import DEFAULT_VALUES, APP_NAME


def get_config_dir(app_name: str) -> Path:
    """Get appropriate config directory for each OS"""
    system = platform.system()
    
    if system == "Darwin":  # macOS
        base = Path.home() / "Library" / "Application Support"
    elif system == "Windows":
        base = Path(os.getenv("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:  # Linux และ Unix-like
        base = Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config"))
    
    return base / app_name


class Settings:
    def __init__(self):
        self.config_dir = get_config_dir(APP_NAME)
        self.config_file = self.config_dir / "config.json"
        self.fallback_file = Path.cwd() / ".local-config.json"
        self.settings = self._load_settings()
    
    def _ensure_config_dir(self):
        """Create config directory if it doesn't exist"""
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError):
            # ถ้าสร้าง system config ไม่ได้ ใช้ local fallback
            pass
    
    def _load_settings(self) -> Dict[str, Any]:
        """Load settings from file or use defaults"""
        settings = DEFAULT_VALUES.copy()
        
        # Try to load from system config
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    user_settings = json.load(f)
                settings.update(user_settings)
                return settings
        except (json.JSONDecodeError, PermissionError, OSError):
            pass
        
        # Fallback to local config
        try:
            if self.fallback_file.exists():
                with open(self.fallback_file, 'r', encoding='utf-8') as f:
                    user_settings = json.load(f)
                settings.update(user_settings)
        except (json.JSONDecodeError, PermissionError, OSError):
            pass
            
        return settings
    
    def save_settings(self):
        """Save settings to file"""
        # Try to save to system config first
        try:
            self._ensure_config_dir()
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=2, ensure_ascii=False)
            return
        except (OSError, PermissionError):
            pass
        
        # Fallback to local config
        try:
            with open(self.fallback_file, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=2, ensure_ascii=False)
        except (OSError, PermissionError) as e:
            raise RuntimeError(f"Cannot save settings: {e}")
    
    def get(self, key: str, default=None):
        """Get setting value"""
        return self.settings.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set setting value"""
        self.settings[key] = value
        self.save_settings()
    
    def reset(self):
        """Reset to default values"""
        self.settings = DEFAULT_VALUES.copy()
        self.save_settings()
    
    def get_config_info(self) -> Dict[str, Any]:
        """Get configuration paths and status"""
        return {
            "config_dir": str(self.config_dir),
            "config_file": str(self.config_file),
            "fallback_file": str(self.fallback_file),
            "config_exists": self.config_file.exists(),
            "fallback_exists": self.fallback_file.exists(),
            "using_fallback": not self.config_file.exists() and self.fallback_file.exists()
        }


# Singleton instance
settings = Settings()