import os
import sys
import logging
import argparse
from datetime import datetime

# =============================================================================
# Logging Setup - Redirect all output to file
# =============================================================================

# Store original stdout for tqdm
_original_stdout = sys.stdout
_original_stderr = sys.stderr


def setup_main_logger(log_dir: str = "logs", experiment_name: str = None) -> str:
    """
    Set up main logger to redirect all output to a file.

    This captures:
    - All logging.* calls
    - All print() statements (via stdout redirect)

    Args:
        log_dir: Directory for log files
        experiment_name: Optional experiment name for log filename

    Returns:
        Path to the created log file
    """
    os.makedirs(log_dir, exist_ok=True)

    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_suffix = f"_{experiment_name}" if experiment_name else ""
    log_filename = f"main{exp_suffix}_{timestamp}.log"
    log_file = os.path.join(log_dir, log_filename)

    # Configure root logger. Use force=True to replace existing handlers
    # so repeated calls (multiple experiment modes) create separate files.
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.FileHandler(log_file)],
        force=True,
    )

    # Create a custom stream to redirect print() to logger
    class LoggerWriter:
        """Redirect stdout/stderr to logger and file."""

        def __init__(self, logger, level, log_file_handle):
            self.logger = logger
            self.level = level
            self.log_file = log_file_handle
            self.buffer = ""

        def write(self, message):
            if message and message.strip():
                self.logger.log(self.level, message.strip())

        def flush(self):
            if self.log_file:
                self.log_file.flush()

    # Open log file for print redirect
    log_file_handle = open(log_file, "a")

    # Get main logger
    main_logger = logging.getLogger("main")

    # Redirect print statements to logger
    sys.stdout = LoggerWriter(main_logger, logging.INFO, log_file_handle)
    sys.stderr = LoggerWriter(main_logger, logging.ERROR, log_file_handle)

    # Log initialization (use original stdout for immediate feedback)
    _original_stdout.write(f"Logging to: {log_file}\n")
    _original_stdout.flush()

    main_logger.info("=" * 80)
    main_logger.info("MAIN EXPERIMENT LOG")
    main_logger.info(f"Experiment: {experiment_name if experiment_name else 'default'}")
    main_logger.info(f"Log file: {log_file}")
    main_logger.info("=" * 80)

    return log_file


def get_main_logger():
    """Get the main logger instance."""
    return logging.getLogger("main")


def get_tqdm_file():
    """Get file handle for tqdm progress bars (uses original stderr)."""
    return _original_stderr


# -----------------------------------------------------------------------------
# Build Global Folder Paths from Config
# -----------------------------------------------------------------------------
def load_global_paths(config):
    """
    Build global path variables using the base 'project_path' directory provided
    in the config.
    """
    PROJECT_PATH = config["project_path"].rstrip("/")  # Ensure no trailing slash
    ANNOTATION_PATH = os.path.join(PROJECT_PATH, "data", "annotation.txt")
    ORTHOMAP_PATH = "/media/bota/BOTA/wheat/example-run-001_20241014T1739_ortho_dsm.tif"
    TILE_PIXEL_PATH = os.path.join(PROJECT_PATH, "data", "tiles_to_pixels.txt")
    MODEL_PATH = os.path.join(
        PROJECT_PATH,
        "binary_classifier",
        "models",
        "best_model_auc91_lr1_-05_bs128_wd_2.5-04.pth",
    )
    CACHE_DIR = os.path.join(PROJECT_PATH, "data", "predictions_cache")
    return (
        PROJECT_PATH,
        ANNOTATION_PATH,
        ORTHOMAP_PATH,
        TILE_PIXEL_PATH,
        MODEL_PATH,
        CACHE_DIR,
    )


# -----------------------------------------------------------------------------
# Parse Command-Line Arguments
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run active sensing experiments using a configuration file."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to the JSON configuration file.",
    )
    return parser.parse_args()
