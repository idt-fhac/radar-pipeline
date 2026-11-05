from dotenv import load_dotenv

# Load .env once, centrally: this runs before any module of the package is
# imported, so every module can rely on the environment variables being set.
load_dotenv()


def main() -> None:
    print("Hello from radar!")
