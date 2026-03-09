#!/usr/bin/env python3
"""
Build script to create cross-platform executables using PyInstaller
"""

import os
import sys
import subprocess

def build_executable():
    """Build the executable using PyInstaller."""
    try:
        # Check if PyInstaller is installed
        try:
            import PyInstaller
        except ImportError:
            print("PyInstaller not found. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        
        # Determine which main file to use
        if os.path.exists('simple_denoise.py'):
            main_file = 'simple_denoise.py'
            app_name = 'X-ray-Denoiser-Simple'
        else:
            main_file = 'main.py'
            app_name = 'X-ray-Denoiser'
        
        print(f"Building executable from {main_file}...")
        
        # Build command
        cmd = [
            sys.executable, "-m", "PyInstaller",
            "--onefile",
            "--windowed",
            f"--name={app_name}",
            main_file
        ]
        
        # Add icon if available (optional)
        if os.path.exists('icon.ico'):
            cmd.extend(['--icon', 'icon.ico'])
        elif os.path.exists('icon.png'):
            cmd.extend(['--icon', 'icon.png'])
        
        # Run PyInstaller
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Build successful! Executable created in dist/{app_name}")
            print("You can find your executable in the 'dist' directory.")
        else:
            print("Build failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            
    except Exception as e:
        print(f"Error building executable: {e}")
        print("Make sure you have PyInstaller installed and all dependencies are available.")

if __name__ == "__main__":
    build_executable()