"""Config command - Manage configuration settings"""

import typer
from typing import Optional
from rich.console import Console

from side_project.lib.settings import settings

console = Console()


def config_command(
    key: str = typer.Argument(None, help="Configuration key to view/set"),
    value: str = typer.Argument(None, help="Value to set (omit to view current value)"),
    list_all: bool = typer.Option(False, "--list", "-l", help="List all settings"),
    reset: bool = typer.Option(False, "--reset", help="Reset to defaults"),
    info: bool = typer.Option(False, "--info", help="Show config file locations")
):
    """Manage configuration settings"""
    
    if info:
        config_info = settings.get_config_info()
        console.print("Configuration:")
        console.print(f"  directory: {config_info['config_dir']}")
        console.print(f"  file:      {config_info['config_file']}")
        console.print(f"  fallback:  {config_info['fallback_file']}")
        
        # แสดงสถานะและ path ที่ใช้งานจริง
        if config_info['config_exists']:
            console.print(f"  status:    using system config")
            console.print(f"  active:    {config_info['config_file']}")
        elif config_info['using_fallback']:
            console.print(f"  status:    using fallback config")
            console.print(f"  active:    {config_info['fallback_file']}")
            console.print(f"  note:      permission issue with system config")
        else:
            console.print(f"  status:    using defaults")
            console.print(f"  will create: {config_info['config_file']}")
        return
    
    if reset:
        if typer.confirm("Reset all settings to defaults?"):
            settings.reset()
            console.print("Settings reset to defaults")
        return
    
    if list_all:
        console.print("Settings:")
        for k, v in settings.settings.items():
            console.print(f"  {k}: {v}")
        return
    
    if key is None:
        console.print("Error: configuration key required")
        console.print("Use --list to see available settings")
        raise typer.Exit(1)
    
    if value is None:
        # View current value
        current_value = settings.get(key)
        if current_value is not None:
            console.print(f"{key}: {current_value}")
        else:
            console.print(f"Error: setting '{key}' not found")
            console.print("Use --list to see available settings")
    else:
        # Set new value
        # Try to convert to appropriate type
        try:
            if value.lower() in ('true', 'false'):
                value = value.lower() == 'true'
            elif value.isdigit():
                value = int(value)
            elif value.replace('.', '').isdigit():
                value = float(value)
        except ValueError:
            pass  # Keep as string
        
        settings.set(key, value)
        console.print(f"Set {key} = {value}")