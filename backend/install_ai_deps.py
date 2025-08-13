#!/usr/bin/env python3
"""
Install AI dependencies with proper version compatibility
"""
import subprocess
import sys
import os

def run_command(cmd):
    """Run command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def main():
    """Install AI dependencies with compatibility fixes"""
    print("🔧 Installing AI Dependencies for Thermal Inspection System")
    print("=" * 60)
    
    commands = [
        ("Upgrade pip", "pip install --upgrade pip setuptools wheel"),
        ("Install PyTorch (latest)", "pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"),
        ("Install core dependencies", "pip install numpy scipy scikit-learn matplotlib seaborn"),
        ("Install OpenCV", "pip install opencv-python opencv-contrib-python"),
        ("Install ultralytics", "pip install ultralytics"),
        ("Install super-gradients (latest)", "pip install super-gradients --no-deps"),
        ("Install remaining deps", "pip install -r requirements-ai.txt"),
        ("Verify installations", "python -c 'import torch, torchvision, ultralytics, cv2; print(\"Core AI dependencies installed successfully\")'"),
        ("Test super-gradients", "python -c 'import super_gradients; print(f\"super-gradients version: {super_gradients.__version__}\")'")
    ]
    
    for desc, cmd in commands:
        print(f"\n🔍 {desc}...")
        success, stdout, stderr = run_command(cmd)
        
        if success:
            print(f"  ✅ {desc} completed")
        else:
            print(f"  ❌ {desc} failed: {stderr}")
            if "torch" in cmd.lower():
                print("  🔄 Trying alternative PyTorch installation...")
                alt_success, _, _ = run_command("pip install torch torchvision")
                if alt_success:
                    print("  ✅ Alternative PyTorch installation successful")
                else:
                    print("  ❌ PyTorch installation failed completely")
    
    print("\n" + "=" * 60)
    print("AI Dependencies Installation Complete")
    print("=" * 60)

if __name__ == "__main__":
    main()
