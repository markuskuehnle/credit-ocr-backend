import os
import sys
import signal
import uvicorn

# Add the src directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import AppConfig
from src.api.main import app, cleanup_dms_environment

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    print(f"\nReceived signal {signum}, shutting down gracefully...")
    cleanup_dms_environment()
    sys.exit(0)

def main():
    """Start the FastAPI server with proper configuration."""
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Load configuration
    config = AppConfig()
    
    # Set default values
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    
    print(f"Starting Credit OCR Demo Backend API")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Reload: {reload}")
    print(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
    print("-" * 50)
    
    try:
        # Mark the app as started so the startup event will run
        from src.api.main import mark_app_as_started
        mark_app_as_started()
        
        # Start the server
        uvicorn.run(
            "src.api.main:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    finally:
        # Ensure cleanup happens even if uvicorn.run() fails
        signal_handler(signal.SIGTERM, None)

if __name__ == "__main__":
    main() 