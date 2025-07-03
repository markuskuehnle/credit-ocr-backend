#!/usr/bin/env python3
"""
Test runner script that ensures proper cleanup and efficient test execution.
"""

import sys
import subprocess
import signal
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] [%(asctime)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Global process tracking
_test_process = None


def signal_handler(signum, frame):
    """Handle interrupt signals to ensure cleanup."""
    logger.info(f"Received signal {signum}, cleaning up...")
    cleanup()
    sys.exit(1)


def cleanup():
    """Clean up any running containers and processes."""
    global _test_process
    
    # Stop test process if running
    if _test_process and _test_process.poll() is None:
        logger.info("Stopping test process...")
        _test_process.terminate()
        try:
            _test_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("Test process didn't stop gracefully, killing...")
            _test_process.kill()
            _test_process.wait()
    
    # Stop any running containers
    logger.info("Stopping any running containers...")
    try:
        subprocess.run(["docker", "stop", "$(docker ps -q)"], shell=True, timeout=30)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        logger.warning("Failed to stop containers gracefully")
    
    # Remove stopped containers
    try:
        subprocess.run(["docker", "container", "prune", "-f"], timeout=30)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        logger.warning("Failed to prune containers")


def run_tests(test_pattern=None, verbose=False, coverage=False):
    """Run tests with proper setup and cleanup."""
    global _test_process
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Build pytest command
        cmd = ["python3", "-m", "pytest"]
        
        if verbose:
            cmd.append("-v")
        
        if coverage:
            cmd.extend(["--cov=src", "--cov-report=term-missing"])
        
        # Add test pattern if specified
        if test_pattern:
            cmd.append(test_pattern)
        else:
            cmd.append("tests/")
        
        logger.info(f"Running tests with command: {' '.join(cmd)}")
        
        # Start test process
        _test_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Stream output
        for line in iter(_test_process.stdout.readline, ''):
            if line:
                print(line.rstrip())
        
        # Wait for completion
        exit_code = _test_process.wait()
        
        if exit_code == 0:
            logger.info("Tests completed successfully")
        else:
            logger.error(f"Tests failed with exit code {exit_code}")
        
        return exit_code
        
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        return 1
    finally:
        cleanup()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run tests with proper cleanup")
    parser.add_argument(
        "test_pattern", 
        nargs="?", 
        help="Test pattern to run (e.g., 'tests/test_dms_mock.py')"
    )
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true", 
        help="Verbose output"
    )
    parser.add_argument(
        "--coverage", 
        action="store_true", 
        help="Run with coverage reporting"
    )
    
    args = parser.parse_args()
    
    # Ensure we're in the right directory
    project_root = Path(__file__).parent
    if not (project_root / "pyproject.toml").exists():
        logger.error("Not in project root directory")
        sys.exit(1)
    
    # Run tests
    exit_code = run_tests(
        test_pattern=args.test_pattern,
        verbose=args.verbose,
        coverage=args.coverage
    )
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main() 