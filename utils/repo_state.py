import subprocess
from pathlib import Path


def is_git_repo_clean() -> bool:
    """
    Checks if the Git repository is clean. Dynamically locates the repo root
    relative to this script's location (utils/get_repo_hash.py -> parent directory).

    Returns:
        True if the repo is clean, False otherwise.
    """

    repo_root = Path(__file__).resolve().parent.parent

    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return len(result.stdout.strip()) == 0

    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_repo_hash(short: bool = True) -> str:
    repo_root = Path(__file__).resolve().parent.parent
    cmd = (
        ["git", "rev-parse", "--short", "HEAD"]
        if short
        else ["git", "rev-parse", "HEAD"]
    )

    try:
        result = subprocess.run(
            cmd, cwd=repo_root, capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


# This ensures it runs cleanly if executed directly, but won't interfere when imported
if __name__ == "__main__":
    print(f"Current Commit Hash: {get_repo_hash()}")
    print(f"Is Repo Clean? {is_git_repo_clean()}")
