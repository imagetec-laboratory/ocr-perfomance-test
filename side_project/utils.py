"""
Utility functions for your CLI

🛠️ Add your helper functions here.
These functions can be used across different commands.
"""

from rich.console import Console
from rich.table import Table
from typing import Dict, Any

console = Console()


def print_styled_message(message: str, style: str = "bold") -> None:
    """
    Print a styled message to console.
    
    Example:
        print_styled_message("Success!", "bold green")
        print_styled_message("Error occurred", "bold red")
    """
    console.print(f"[{style}]{message}[/{style}]")


def create_info_table(project_info: Dict[str, Any]) -> Table:
    """
    Create a nice table showing project information.
    
    This is used by the 'info' command.
    Modify this function to show different information.
    """
    table = Table(title="Project Information")
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    
    table.add_row("Name", project_info.get("name", "Unknown"))
    table.add_row("Version", project_info.get("version", "Unknown"))
    table.add_row("Description", project_info.get("description", "Unknown"))
    table.add_row("Author", project_info.get("author", "Unknown"))
    
    return table


def print_greeting(name: str, greeting: str, count: int) -> None:
    """
    Print greeting message with Rich formatting.
    
    This is used by the 'hello' command.
    Customize the formatting as needed.
    """
    for _ in range(count):
        console.print(f"[bold green]{greeting}[/bold green] [blue]{name}[/blue]! 🎉")