# This file makes the 'routes' directory a Python package.

# Import the route modules to ensure the endpoints are registered
# with the APIRouter instances defined in router_registry.py
# when the 'app.api.routes' package is imported (e.g., by main.py).

from . import health
from . import system
from . import transcription
from . import diarization

# If you add new route files (e.g., custom.py), import them here:
# from . import custom