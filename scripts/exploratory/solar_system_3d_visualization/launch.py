#!/usr/bin/env python3
"""
Launcher script for the 3D Solar System Visualization
"""

import os
import sys
import webbrowser
import subprocess
import time
from pathlib import Path

def main():
    """Launch the 3D Solar System Visualization"""
    script_dir = Path(__file__).parent
    port = 8348

    print("🚀 Launching TEP-GNSS Solar System 3D Visualization...")
    print(f"📁 Directory: {script_dir}")
    print(f"🌐 URL: http://localhost:{port}")
    print("🔄 Starting local server...")

    try:
        # Start the HTTP server
        server_process = subprocess.Popen([
            sys.executable, '-m', 'http.server', str(port)
        ], cwd=script_dir)

        # Wait a moment for server to start
        time.sleep(1.5)

        # Open browser
        print("🌐 Opening in default browser...")
        webbrowser.open(f'http://localhost:{port}')

        print("✅ Visualization launched successfully!")
        print("🎮 Controls:")
        print("   - Mouse: Rotate and zoom the view")
        print("   - Play/Pause: Start/stop animation")
        print("   - Speed: Adjust animation speed")
        print("   - Reset: Return to start date")
        print("\nPress Ctrl+C to stop the server")

        # Wait for server process
        server_process.wait()

    except KeyboardInterrupt:
        print("\n🛑 Stopping server...")
        server_process.terminate()
        server_process.wait()
        print("✅ Server stopped")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
