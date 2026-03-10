from pathlib import Path
from dotenv import load_dotenv


def load_env() -> None:
    """Load environment variables from .env."""
    load_dotenv()


def ensure_output_dir(path: Path) -> None:
    """Create output directories if they do not already exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
