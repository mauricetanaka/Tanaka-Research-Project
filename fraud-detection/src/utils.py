import json
import joblib
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt

# Import paths from config
from config import MODELS_DIR, METRICS_DIR, PLOTS_DIR, REPORTS_DIR

# Directory setup
def ensure_dirs():
    """Create all output directories if they don't already exist."""
    for d in [MODELS_DIR, METRICS_DIR, PLOTS_DIR, REPORTS_DIR]:
        # Using pathlib's mkdir instead of os.makedirs
        d.mkdir(parents=True, exist_ok=True)


# Console logging helpers for consistent formatting across scripts

def section(title: str):
    """Print a clearly visible section header to the console."""
    width = 62
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def step(number: int, total: int, description: str):
    """Print a numbered step label, e.g. [3/7] Cleaning data..."""
    print(f"\n[{number}/{total}] {description}...")


def info(message: str):
    """Print an indented informational line."""
    print(f"    {message}")


def success(message: str):
    """Print a success confirmation."""
    print(f"    ✅ {message}")


def warn(message: str):
    """Print a warning."""
    print(f"    ⚠️  {message}")


# File saving helpers
def save_plot(fig: plt.Figure, filename: str, subdir: str = ""):
    """
    Save a matplotlib figure to the plots directory.
    """
    # pathlib handles the slashes automatically
    directory = PLOTS_DIR / subdir if subdir else PLOTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    
    path = directory / filename
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    success(f"Saved plot: {path.name}")
    return path


def save_metrics(metrics: dict, filename: str):
    """
    Save a metrics dictionary as a formatted JSON file.
    """
    path = METRICS_DIR / filename
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
        
    success(f"Saved metrics: {path.name}")
    return path


def load_metrics(filename: str) -> dict:
    """Load a previously saved metrics JSON file."""
    path = METRICS_DIR / filename
    with open(path, "r") as f:
        return json.load(f)


def save_model(model, filename: str):
    """
    Serialize and save a trained machine learning model (.pkl).
    """
    path = MODELS_DIR / filename
    joblib.dump(model, path)
    success(f"Saved model weights: {path.name}")
    return path


def load_model(filename: str):
    """Load a serialized machine learning model (.pkl)."""
    path = MODELS_DIR / filename
    return joblib.load(path)