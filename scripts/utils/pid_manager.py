#!/usr/bin/env python3
"""
PID Manager Utility for TEP-GNSS Pipeline

Provides single-instance enforcement for pipeline steps by:
1. Detecting existing instances of a script
2. Automatically killing all existing instances and workers
3. Creating PID file locks
4. Cleaning up on exit

Usage:
    from utils.pid_manager import ensure_single_instance
    
    @ensure_single_instance
    def main():
        # Your main function
        pass
"""

import os
import sys
import time
import subprocess
import signal
from pathlib import Path
from functools import wraps


class PIDManager:
    """Manages PID files and ensures single instance execution"""
    
    def __init__(self, script_name: str, root_dir: Path):
        """
        Initialize PID manager for a specific script
        
        Args:
            script_name: Name of the script (e.g., 'step_2_0_tep_correlation_analysis')
            root_dir: Root directory of the project
        """
        self.script_name = script_name
        self.root_dir = root_dir
        self.pid_file = root_dir / "results/tmp" / f"{script_name}.pid"
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)
        
    def kill_existing_instances(self) -> int:
        """
        Find and kill all existing instances of this script and related TEP-GNSS processes
        
        Returns:
            Number of processes killed
        """
        killed_count = 0
        
        try:
            # Simple approach: just kill processes with the exact script name
            result = subprocess.run(
                ['pgrep', '-f', f'{self.script_name}.py'],
                capture_output=True, text=True, timeout=3
            )
            
            existing_pids = []
            if result.returncode == 0 and result.stdout.strip():
                existing_pids.extend([int(pid) for pid in result.stdout.strip().split('\n') if pid])
            
            # Remove duplicates and filter out our own PID
            our_pid = os.getpid()
            unique_pids = list(set(existing_pids))
            other_pids = [pid for pid in unique_pids if pid != our_pid]
            
            if other_pids:
                print(f"\n{'='*80}")
                print(f"WARNING: Found {len(other_pids)} existing instance(s) of {self.script_name}")
                print("Terminating all previous instances...")
                print("="*80)
                
                for pid in other_pids:
                    try:
                        # Try graceful termination first
                        os.kill(pid, signal.SIGTERM)
                        time.sleep(0.5)
                        # Force kill if still running
                        os.kill(pid, signal.SIGKILL)
                        print(f"  Killed PID {pid}")
                        killed_count += 1
                    except OSError:
                        pass  # Already dead
                
                time.sleep(1)  # Wait for cleanup
                print(f"Successfully terminated {killed_count} process(es)\n")
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            # pgrep not available or timed out - continue anyway
            pass
        
        # Clean up any stale PID file
        if self.pid_file.exists():
            self.pid_file.unlink()
        
        return killed_count
    
    def acquire_lock(self):
        """Create PID file with our process ID"""
        with open(self.pid_file, 'w') as f:
            f.write(str(os.getpid()))
    
    def release_lock(self):
        """Remove PID file"""
        if self.pid_file.exists():
            try:
                self.pid_file.unlink()
            except Exception:
                pass  # Best effort cleanup


def ensure_single_instance(func):
    """
    Decorator to ensure only one instance of a script runs at a time
    
    Usage:
        @ensure_single_instance
        def main():
            # Your main function
            pass
    
    The decorator will:
    1. Extract script name from the calling module
    2. Kill any existing instances
    3. Create a PID lock
    4. Execute the function
    5. Clean up PID lock on exit
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get the script name from the calling module
        import inspect
        frame = inspect.currentframe()
        caller_frame = frame.f_back
        caller_module = inspect.getmodule(caller_frame)
        
        if caller_module and hasattr(caller_module, '__file__'):
            script_path = Path(caller_module.__file__)
            script_name = script_path.stem  # Filename without extension
            
            # Find project root (look for scripts/utils directory)
            root_dir = script_path.parent
            while root_dir.name != 'TEP-GNSS' and root_dir != root_dir.parent:
                root_dir = root_dir.parent
            
            # If we couldn't find root, use current directory's parent
            if root_dir.name != 'TEP-GNSS':
                root_dir = Path.cwd()
        else:
            # Fallback if we can't determine the script name
            script_name = "unknown_script"
            root_dir = Path.cwd()
        
        # Create PID manager
        pid_manager = PIDManager(script_name, root_dir)
        
        # Kill existing instances
        pid_manager.kill_existing_instances()
        
        # Acquire lock
        pid_manager.acquire_lock()
        
        try:
            # Run the actual function
            return func(*args, **kwargs)
        finally:
            # Always clean up PID lock
            pid_manager.release_lock()
    
    return wrapper


def kill_all_tep_processes() -> int:
    """
    Kill all TEP-GNSS related Python processes.
    
    This is a more aggressive cleanup function that can be called manually
    when you need to ensure all TEP processes are terminated.
    
    Returns:
        Number of processes killed
    """
    killed_count = 0
    
    try:
        # Find all Python processes related to TEP-GNSS
        patterns = [
            'TEP-GNSS.*\.py',
            'step_.*\.py',
            'multiprocessing.spawn',
            'control_band_analysis',
            'tep.*analysis'
        ]
        
        all_pids = set()
        for pattern in patterns:
            try:
                result = subprocess.run(
                    ['pgrep', '-f', pattern],
                    capture_output=True, text=True, timeout=5
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    pids = [int(pid) for pid in result.stdout.strip().split('\n') if pid]
                    all_pids.update(pids)
            except (subprocess.TimeoutExpired, Exception):
                pass  # Skip if pgrep hangs or fails
        
        # Filter out our own PID
        our_pid = os.getpid()
        other_pids = [pid for pid in all_pids if pid != our_pid]
        
        if other_pids:
            print(f"\n{'='*80}")
            print(f"AGGRESSIVE CLEANUP: Found {len(other_pids)} TEP-GNSS process(es)")
            print("Terminating all TEP-GNSS processes...")
            print("="*80)
            
            for pid in other_pids:
                try:
                    # Try graceful termination first
                    os.kill(pid, signal.SIGTERM)
                    time.sleep(0.5)
                    # Force kill if still running
                    os.kill(pid, signal.SIGKILL)
                    print(f"  Killed PID {pid}")
                    killed_count += 1
                except OSError:
                    pass  # Already dead
            
            time.sleep(2)  # Wait for cleanup
            print(f"Successfully terminated {killed_count} process(es)\n")
        else:
            print("No TEP-GNSS processes found to terminate.")
            
    except FileNotFoundError:
        print("pgrep not available on this system")
    except Exception as e:
        print(f"Error during cleanup: {e}")
    
    return killed_count


def with_pid_lock(script_name: str, root_dir: Path = None):
    """
    Context manager for PID locking
    
    Usage:
        from pathlib import Path
        from utils.pid_manager import with_pid_lock
        
        ROOT = Path(__file__).resolve().parents[2]
        
        with with_pid_lock('my_script', ROOT):
            # Your code here
            pass
    
    Args:
        script_name: Name of the script (without .py extension)
        root_dir: Root directory of the project (optional, uses cwd if not provided)
    """
    if root_dir is None:
        root_dir = Path.cwd()
    
    class PIDLockContext:
        def __init__(self, script_name, root_dir):
            self.pid_manager = PIDManager(script_name, root_dir)
        
        def __enter__(self):
            self.pid_manager.kill_existing_instances()
            self.pid_manager.acquire_lock()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.pid_manager.release_lock()
            return False
    
    return PIDLockContext(script_name, root_dir)


if __name__ == "__main__":
    # Test the PID manager
    print("Testing PID Manager")
    
    test_root = Path.cwd()
    pid_manager = PIDManager("test_script", test_root)
    
    print(f"Script: test_script")
    print(f"PID file: {pid_manager.pid_file}")
    print(f"Current PID: {os.getpid()}")
    
    killed = pid_manager.kill_existing_instances()
    print(f"Killed {killed} existing instances")
    
    pid_manager.acquire_lock()
    print(f"Lock acquired, PID file created")
    
    time.sleep(1)
    
    pid_manager.release_lock()
    print(f"Lock released, PID file removed")
