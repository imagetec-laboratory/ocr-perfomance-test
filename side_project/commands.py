"""
CLI Commands

🎯 Define your CLI commands here.
Each function becomes a CLI command when registered in main.py

Command Function Template:
def your_command(
    arg: str = typer.Argument(..., help="Description"),
    option: str = typer.Option("default", "--flag", "-f", help="Description")
):
    \"\"\"Command description shown in help\"\"\"
    # Your command logic here
    console.print("Command executed!")
"""

import typer
import sys
from typing import Optional
from rich.console import Console

# Import modules
import side_project.config as config
import side_project.utils as utils

PROJECT_INFO = config.PROJECT_INFO
DEFAULT_VALUES = config.DEFAULT_VALUES
create_info_table = utils.create_info_table
print_greeting = utils.print_greeting
print_styled_message = utils.print_styled_message

console = Console()


def hello_command(
    name: str = typer.Argument(..., help="Name to greet"),
    greeting: Optional[str] = typer.Option(
        DEFAULT_VALUES["greeting"], 
        "--greeting", "-g", 
        help="Greeting message"
    ),
    count: int = typer.Option(
        DEFAULT_VALUES["count"], 
        "--count", "-c", 
        help="Number of greetings"
    )
):
    """Say hello to someone. Example command you can customize."""
    if count > DEFAULT_VALUES["max_count"]:
        print_styled_message(f"Count too high! Max is {DEFAULT_VALUES['max_count']}", "bold red")
        raise typer.Exit(1)
    
    print_greeting(name, greeting, count)


def info_command():
    """Show project information. Customize in config.py"""
    table = create_info_table(PROJECT_INFO)
    console.print(table)


def version_command():
    """Show version. Version is set in config.py"""
    console.print(f"{PROJECT_INFO['name']} [bold]v{PROJECT_INFO['version']}[/bold]")