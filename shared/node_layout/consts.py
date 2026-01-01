"""Constants for the node layout system."""

__all__ = [
    "NODE_HEADER_HEIGHT",
    "SOCKET_BASE_HEIGHT",
    "SOCKET_HEIGHT_MULTIPLIERS",
]


# Socket type multipliers for height estimation (when NOT connected)
# Some socket types render larger UI elements when displaying input fields
# When connected, all sockets are effectively 1 row tall
SOCKET_HEIGHT_MULTIPLIERS: dict[str, float] = {
    # Vector types (show X, Y, Z fields when not connected)
    "VECTOR": 3.0,
    "NodeSocketVector": 3.0,
    "NodeSocketVectorDirection": 3.0,
    "NodeSocketVectorEuler": 3.0,
    "NodeSocketVectorTranslation": 3.0,
    "NodeSocketVectorVelocity": 3.0,
    "NodeSocketVectorAcceleration": 3.0,
    "NodeSocketVectorXYZ": 3.0,
    # Rotation (shows X, Y, Z, W or euler when not connected)
    "ROTATION": 3.0,
    "NodeSocketRotation": 3.0,
    # Color types - inline color picker, always 1 row
    # (no multiplier needed, defaults to 1.0)
    # Matrix (4x4 = 16 values, but usually collapsed)
    "MATRIX": 2.0,
    "NodeSocketMatrix": 2.0,
    # Default for standard types (float, int, bool, string, color, etc.) is 1.0
}

# Base height estimates (in pixels)
NODE_HEADER_HEIGHT = 30.0  # Node title bar
SOCKET_BASE_HEIGHT = 22.0  # Height per socket row
