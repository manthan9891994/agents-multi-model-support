import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

GOOGLE_API_KEY    = os.getenv("GOOGLE_API_KEY", "")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

DEFAULT_PROVIDER  = os.getenv("DEFAULT_PROVIDER", "google")

LAYER2_ENABLED    = os.getenv("LAYER2_ENABLED", "false").lower() == "true"
LAYER3_ENABLED    = os.getenv("LAYER3_ENABLED", "false").lower() == "true"
LAYER4_ENABLED    = os.getenv("LAYER4_ENABLED", "false").lower() == "true"
