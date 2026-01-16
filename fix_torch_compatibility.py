import subprocess
import sys

def get_compatible_torchvision(torch_version):
    compatibility_map = {
        '2.0': '0.15', '2.1': '0.16', '2.2': '0.17', '2.3': '0.18',
        '2.4': '0.19', '2.5': '0.20', '2.6': '0.21', '2.7': '0.22',
        '2.8': '0.23', '2.9': '0.24',
    }
    version_parts = torch_version.split('+')[0].split('.')
    major_minor = '.'.join(version_parts[:2])
    return compatibility_map.get(major_minor, '0.17')

def main():
    try:
        import torch
        import torchvision
        
        torch_version = torch.__version__
        tv_version = torchvision.__version__
        
        print(f"Current PyTorch version: {torch_version}")
        print(f"Current torchvision version: {tv_version}")
        
        compatible_tv = get_compatible_torchvision(torch_version)
        current_tv_major_minor = '.'.join(tv_version.split('.')[:2])
        
        if compatible_tv != current_tv_major_minor:
            print(f"\nVersion mismatch detected!")
            print(f"PyTorch {torch_version} requires torchvision {compatible_tv}.x")
            print(f"Installing torchvision=={compatible_tv}.0...")
            
            cmd = [sys.executable, "-m", "pip", "install", f"torchvision=={compatible_tv}.0", "--upgrade"]
            subprocess.check_call(cmd)
            
            print(f"\nSuccessfully updated torchvision to {compatible_tv}.0")
            print("Please restart your Python session for changes to take effect.")
        else:
            print(f"\nVersions are compatible!")
            
    except ImportError as e:
        print(f"Error: {e}")
        print("Make sure PyTorch is installed first.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
