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
├── addons/
│   ├── node_layout/         # Auto-layout addon for node editors
│   ├── uv_map/               # UV Map modifier addon
│   └── voxel_terrain/        # Main voxel terrain addon
├── shared/                   # Shared modules used across addons
│   ├── node_layout/          # PCB-style node layout algorithm
│   ├── uv_map/               # UV map node generation
│   └── node_group_compare.py # Node group comparison utilities
├── scripts/                  # Build and utility scripts
├── pyproject.toml            # Python project config (dev tools)
├── pyrightconfig.json        # Pyright configuration
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
ln -s /path/to/addons/voxel_terrain ~/Library/Application\ Support/Blender/5.0/extensions/user_default/voxel_terrain

# Linux
ln -s /path/to/addons/voxel_terrain ~/.config/blender/5.0/extensions/user_default/voxel_terrain

# Windows (run as admin)
mklink /D "%APPDATA%\Blender Foundation\Blender\5.0\extensions\user_default\voxel_terrain" "C:\path\to\addons\voxel_terrain"
```

### Option 2: Install from Disk

1. Build the extension: `blender --command extension build`
2. In Blender: Edit → Preferences → Get Extensions → Install from Disk
3. Select the generated `.zip` file

## Usage

1. Open Blender 5.0+
2. Enable the addon in Preferences → Add-ons
3. Press `N` in the 3D Viewport to open the sidebar
4. Find the "Voxel Terrain" tab

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
