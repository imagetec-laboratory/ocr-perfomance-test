"""Info command - Show project information"""

import typer
from rich.console import Console

from side_project.config import PROJECT_INFO
from side_project.lib.utils import create_info_table

console = Console()


def info_command():
    """Show project information. Customize in config.py"""
    table = create_info_table(PROJECT_INFO)
    console.print(table)