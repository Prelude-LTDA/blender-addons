# Voxel Terrain

Voxel-based terrain generation and editing tools for Blender 5.0+.

## Features

- N-panel UI in the 3D Viewport sidebar
- Voxel terrain generation
- Type annotations throughout
- Proper module reloading support
- Linting with Ruff
- Type checking with Pyright
- Development type stubs via fake-bpy-module

## Project Structure

```
voxel-terrain/
├── __init__.py              # Main addon entry point with reloading support
├── operators.py             # Operator classes
├── panels.py                # UI panel classes
├── blender_manifest.toml    # Blender 5.0+ extension manifest
├── pyproject.toml           # Python project config (dev tools)
├── pyrightconfig.json       # Pyright configuration
├── .vscode/
│   ├── settings.json        # VS Code settings
│   └── extensions.json      # Recommended extensions
├── .gitignore
└── README.md
```

## Development Setup

1. **Create a virtual environment:**
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   # or
   .venv\Scripts\activate     # On Windows
   ```

2. **Install development dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

3. **Install recommended VS Code extensions** (see `.vscode/extensions.json`)

## Building the Extension

Use Blender's command-line tool to build the extension:

```bash
blender --command extension build
```

This creates a `.zip` file ready for distribution.

## Installing for Development

### Option 1: Symlink (Recommended for development)

Create a symlink in Blender's extensions directory:

```bash
# macOS
ln -s /path/to/this/folder ~/Library/Application\ Support/Blender/5.0/extensions/user_default/hello_world_addon

# Linux
ln -s /path/to/this/folder ~/.config/blender/5.0/extensions/user_default/hello_world_addon

# Windows (run as admin)
mklink /D "%APPDATA%\Blender Foundation\Blender\5.0\extensions\user_default\hello_world_addon" "C:\path\to\this\folder"
```

### Option 2: Install from Disk

1. Build the extension: `blender --command extension build`
2. In Blender: Edit → Preferences → Get Extensions → Install from Disk
3. Select the generated `.zip` file

## Usage

1. Open Blender 5.0+
2. Enable the addon in Preferences → Add-ons
3. Press `N` in the 3D Viewport to open the sidebar
4. Find the "Hello" tab

## Linting & Type Checking

```bash
# Run Ruff linter
ruff check .

# Fix auto-fixable issues
ruff check --fix .

# Format code
ruff format .

# Run Pyright type checker
pyright
```

## License

GPL-3.0-or-later
