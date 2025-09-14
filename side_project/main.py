"""
Side Project CLI Template

A simple CLI template using Typer and Rich.
Customize this template by:
1. Change project info in config.py
2. Add your commands in commands.py  
3. Add utility functions in utils.py
"""

import typer
import sys

# Import handling for both development and PyInstaller
try:
    # Try relative imports first (development)
    from .config import PROJECT_INFO
    from .commands import hello_command, info_command, version_command
except ImportError:
    # Fallback for PyInstaller executable
    import side_project.config as config
    import side_project.commands as commands
    PROJECT_INFO = config.PROJECT_INFO
    hello_command = commands.hello_command
    info_command = commands.info_command
    version_command = commands.version_command

# Create CLI app
app = typer.Typer(
    name=PROJECT_INFO["name"].lower().replace(" ", "-"),
    help=f"{PROJECT_INFO['name']} - {PROJECT_INFO['description']}"
)

# Register commands
app.command(name="hello")(hello_command)
app.command(name="info")(info_command) 
app.command(name="version")(version_command)

# TODO: Add your own commands here
# app.command(name="your-command")(your_command_function)


def main():
    """Main entry point"""
    app()


if __name__ == "__main__":
    main()
