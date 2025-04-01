"""
Central registry for API routers.

Defines APIRouter instances with appropriate prefixes and tags
for organizing API endpoints defined in the `app.api.routes` package.
"""

from fastapi import APIRouter

# Define routers for different API sections
router_health = APIRouter(
    tags=["Health"],
    # No prefix for the main health check, usually / or /health
)

router_system = APIRouter(
    prefix="/system",
    tags=["System & Monitoring"],
)

router_transcription = APIRouter(
    prefix="/transcriptions",
    tags=["Transcription"],
)

router_diarization = APIRouter(
    prefix="/diarize",
    tags=["Diarization"],
)

# Example for adding a new feature router:
# router_custom_feature = APIRouter(
#     prefix="/custom",
#     tags=["Custom Feature"],
# )