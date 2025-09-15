"""Version command - Show version information"""

from rich.console import Console

from side_project.config import PROJECT_INFO

console = Console()


def version_command():
    """Show version. Version is set in config.py"""
    console.print(f"{PROJECT_INFO['name']} [bold]v{PROJECT_INFO['version']}[/bold]")