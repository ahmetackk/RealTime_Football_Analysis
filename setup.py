import subprocess
import sys
import importlib.util
import os

class ProjectInstaller:
    def __init__(self):
        print("Football Analysis Project Setup Wizard Starting (Optimized for RTX 30/40 Series)...")
        
        # TensorRT ve ONNX eklendi, versiyonlar senin sistemine göre güncellendi
        self.critical_requirements = [
            "numpy==1.26.4",  # TensorRT ve OpenCV uyumu için kritik
            "opencv-python-headless>=4.10.0",
            "Pillow>=10.0.0",
            "ultralytics>=8.3.27", # Güncel sürüm
            "tensorrt>=10.0.0",    # EKLENDİ: Hızlandırma için şart
            "onnx>=1.19.0",        # EKLENDİ
            "onnxruntime-gpu>=1.21.0", # EKLENDİ
            "onnxsim>=0.4.0",      # EKLENDİ (Opsiyonel ama iyi)
            "supervision>=0.27.0",
            "pytorchvideo>=0.1.5",
            "scikit-learn>=1.3.0",
            "scipy>=1.11.0",
            "pandas>=2.0.0",
            "h5py>=3.10.0",
            "decord==0.6.0",
            "albumentations>=1.4.0",
            "tqdm>=4.66.0",
            "matplotlib>=3.7.0",
            "requests>=2.31.0",
            "psutil>=5.9.0",
        ]

        self.gui_requirements = [
            "PyQt5>=5.15.0",
        ]
        
        self.extra_requirements = [
            "lmdb>=1.4.0",
            "nltk>=3.8.0",
            "editdistance>=0.6.0",
            "timm>=0.9.0",
            "pytorch_lightning>=2.4.0",
            "hydra-core>=1.3.0",
            "omegaconf>=2.3.0",
            "shapely>=2.0.0",
            "termcolor>=2.3.0",
            "yacs>=0.1.8",
        ]

    def _log(self, message, status="INFO"):
        colors = {
            "INFO": "\033[94m",
            "SUCCESS": "\033[92m",
            "WARNING": "\033[93m",
            "ERROR": "\033[91m",
            "RESET": "\033[0m"
        }
        prefix = f"[{status}]"
        print(f"{colors.get(status, '')}{prefix} {message}{colors['RESET']}")

    def run_command(self, command, description, ignore_errors=False):
        self._log(f"{description}...", "INFO")
        try:
            subprocess.check_call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._log(f"COMPLETED: {description}", "SUCCESS")
            return True
        except subprocess.CalledProcessError as e:
            if ignore_errors:
                self._log(f"WARNING: {description} skipped (error ignored)", "WARNING")
                return False
            else:
                self._log(f"ERROR: {description} failed. (Code: {e.returncode})", "ERROR")
                return False

    def setup_pytorch_specific(self):
        self._log("Installing PyTorch 2.5.1 with CUDA 12.1 Support...", "WARNING")
        
        # Senin sistemindeki çalışan konfigürasyonu zorluyoruz
        # torch 2.5.1, torchvision 0.20.1, torchaudio 2.5.1
        
        try:
            # Önce mevcut torch varsa kaldıralım (versiyon çakışmasını önlemek için)
            subprocess.run([sys.executable, "-m", "pip", "uninstall", "torch", "torchvision", "torchaudio", "-y"], 
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            cmd = [
                sys.executable, "-m", "pip", "install",
                "torch==2.5.1+cu121", 
                "torchvision==0.20.1+cu121", 
                "torchaudio==2.5.1+cu121",
                "--index-url", "https://download.pytorch.org/whl/cu121"
            ]
            
            success = self.run_command(cmd, "Installing PyTorch 2.5.1+cu121", ignore_errors=False)
            
            if success:
                import torch
                if torch.cuda.is_available():
                    self._log(f"PyTorch Installed Successfully! GPU: {torch.cuda.get_device_name(0)}", "SUCCESS")
                else:
                    self._log("PyTorch installed but GPU not detected. Please check NVIDIA Drivers.", "WARNING")
            
        except Exception as e:
            self._log(f"Critical Error installing PyTorch: {e}", "ERROR")

    def install_git_packages(self):
        cmd = [
            sys.executable, "-m", "pip", "install",
            "git+https://github.com/roboflow/sports.git"
        ]
        self.run_command(cmd, "Roboflow Sports Library (Git)", ignore_errors=True)

    def install_requirements(self):
        self.run_command(
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
            "Updating Pip"
        )

        self._log("Installing Critical Packages (TensorRT, ONNX, etc.)...", "INFO")
        for req in self.critical_requirements:
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", req],
                    stdout=subprocess.DEVNULL, # Hata detayını görmek istersen bunu kaldırabilirsin
                    stderr=subprocess.DEVNULL
                )
                print(f"  OK: {req.split('>=')[0].split('==')[0]}")
            except:
                self._log(f"WARNING: Problem installing {req}", "WARNING")

        self._log("Installing GUI Libraries...", "INFO")
        for req in self.gui_requirements:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", req], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except:
                pass

        self._log("Installing Extra Tools...", "INFO")
        cmd_extra = [sys.executable, "-m", "pip", "install"] + self.extra_requirements
        self.run_command(cmd_extra, "Extra Packages", ignore_errors=True)

    def force_numpy_downgrade(self):
        # TensorRT ve bazı eski kütüphaneler NumPy 2.0 ile çalışmaz. 1.26.4'e sabitliyoruz.
        self._log("Enforcing NumPy 1.26.4 for Compatibility...", "WARNING")
        cmd = [
            sys.executable, "-m", "pip", "install",
            "numpy==1.26.4", "--force-reinstall", "--no-deps"
        ]
        self.run_command(cmd, "Locking NumPy to 1.26.4", ignore_errors=True)

    def setup_nltk(self):
        try:
            import nltk
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True) # Yeni versiyonlarda gerekebiliyor
        except:
            pass

    def create_directories(self):
        for folder in ["data", "models", "results", "runs"]:
            if not os.path.exists(folder):
                os.makedirs(folder)

    def check_health(self):
        print("\n" + "="*60)
        print("INSTALLATION HEALTH REPORT (RTX OPTIMIZED)")
        print("="*60)

        # 1. NumPy Check
        try:
            import numpy
            print(f"NumPy Version   : {numpy.__version__} (Target: 1.26.4)")
        except: print("NumPy: ERROR")

        # 2. PyTorch & CUDA Check
        try:
            import torch
            print(f"PyTorch Version : {torch.__version__} (Target: 2.5.1+cu121)")
            if torch.cuda.is_available():
                print(f"CUDA Status     : ACTIVE ✅")
                print(f"GPU Model       : {torch.cuda.get_device_name(0)}")
                print(f"CUDA Version    : {torch.version.cuda}")
            else:
                print(f"CUDA Status     : INACTIVE ❌ (Check NVIDIA Drivers)")
        except: print("PyTorch: ERROR")

        # 3. TensorRT Check
        try:
            import tensorrt
            print(f"TensorRT        : {tensorrt.__version__} ✅")
        except ImportError:
            print(f"TensorRT        : NOT FOUND ❌ (Engine files won't work!)")

        # 4. Ultralytics Check
        try:
            import ultralytics
            print(f"Ultralytics     : {ultralytics.__version__}")
        except: print("Ultralytics: ERROR")

        print("="*60)
        print("REMINDER: Do not copy .engine files from another PC.")
        print("Run 'yolo export model=x.pt format=engine device=0 half=True' on THIS machine.")
        print("="*60)

    def start(self):
        print("\n--- FOOTBALL ANALYSIS SETUP (GPU EDITION) ---\n")
        self.create_directories()
        self.setup_pytorch_specific() # Önce PyTorch (En ağırı ve temeli)
        self.install_requirements()   # Sonra diğerleri
        self.install_git_packages()
        self.force_numpy_downgrade()  # En son NumPy düzeltmesi
        self.setup_nltk()
        self.check_health()

if __name__ == "__main__":
    # Yönetici hakları kontrolü (Windows için opsiyonel ama iyi olur)
    try:
        installer = ProjectInstaller()
        installer.start()
    except KeyboardInterrupt:
        print("\nSetup cancelled by user.")
    except Exception as e:
        print(f"\nUnexpected Error: {e}")