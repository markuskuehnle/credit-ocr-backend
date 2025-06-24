import os
import sys
import signal
import uvicorn
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

from api.main import app
from config import AppConfig

def signal_handler(signum, frame):
    """Handle shutdown signals to ensure proper cleanup."""
    print(f"\nReceived signal {signum}, shutting down gracefully...")
    
    # Import here to avoid circular imports
    try:
        from api.main import cleanup_dms_environment
        cleanup_dms_environment()
        print("DMS mock environment stopped")
    except Exception as e:
        print(f"Error stopping DMS mock environment: {e}")
    
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
        # Start the server
        uvicorn.run(
            "api.main:app",
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