from pathlib import Path

from dotenv import load_dotenv

# Root of the repository (src/radar/__init__.py -> src/radar -> src -> repository).
# Every path in the project is built from here, so nothing depends on the
# directory a script is started from.
ROOT = Path(__file__).resolve().parents[2]

# Load .env once, centrally: this runs before any module of the package is
# imported, so every module can rely on the environment variables being set.
load_dotenv(ROOT / '.env')


def main() -> None:
    print("Hello from radar!")
