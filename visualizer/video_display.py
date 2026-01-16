import cv2
import numpy as np
import os
import sys
from typing import Optional

def setup_pyqt_env():
    try:
        import PyQt5
        pyqt_path = os.path.dirname(PyQt5.__file__)
        plugins_path = os.path.join(pyqt_path, 'Qt5', 'plugins')
        
        if os.path.exists(plugins_path):
            os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugins_path
        
        alt_plugins = os.path.join(pyqt_path, 'Qt', 'plugins')
        if os.path.exists(alt_plugins):
            os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = alt_plugins
            
        site_packages = os.path.dirname(pyqt_path)
        qt_plugins = os.path.join(site_packages, 'PyQt5', 'Qt5', 'plugins')
        if os.path.exists(qt_plugins):
            os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = qt_plugins
            
    except Exception as e:
        print(f"PyQt5 environment setup error: {e}")

HAS_PYQT5 = False
try:
    setup_pyqt_env()
    from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QWidget, QVBoxLayout, QShortcut
    from PyQt5.QtGui import QImage, QPixmap, QKeySequence
    from PyQt5.QtCore import Qt, QTimer
    HAS_PYQT5 = True
except ImportError:
    print("PyQt5 not found. OpenCV will be used.")
except Exception as e:
    print(f"PyQt5 loading error: {e}. OpenCV will be used.")


class HighQualityDisplay:
    def __init__(self, window_title: str = "Analysis", max_width: int = 1920, max_height: int = 1080):
        self.window_title = window_title
        self.max_width = max_width
        self.max_height = max_height
        self.should_quit = False
        
        if not HAS_PYQT5:
            self.app = None
            self.window = None
            return
        
        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication([])
        
        self.window = QMainWindow()
        self.window.setWindowTitle(window_title)
        
        self.central_widget = QWidget()
        self.window.setCentralWidget(self.central_widget)
        
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(False)
        self.layout.addWidget(self.image_label)
        
        self.quit_shortcut = QShortcut(QKeySequence('Q'), self.window)
        self.quit_shortcut.activated.connect(self._on_quit)
        
        self.esc_shortcut = QShortcut(QKeySequence(Qt.Key_Escape), self.window)
        self.esc_shortcut.activated.connect(self._on_quit)
        
        self.window.closeEvent = self._on_close
        
        self.scale_factor = 1.0
        self.display_size = None
        
    def _on_quit(self):
        self.should_quit = True
        
    def _on_close(self, event):
        self.should_quit = True
        event.accept()
    
    def setup(self, frame_width: int, frame_height: int):
        if not HAS_PYQT5 or self.window is None:
            return
        
        scale_w = self.max_width / frame_width
        scale_h = self.max_height / frame_height
        self.scale_factor = min(scale_w, scale_h, 1.0)
        
        display_width = int(frame_width * self.scale_factor)
        display_height = int(frame_height * self.scale_factor)
        self.display_size = (display_width, display_height)
        
        self.window.setFixedSize(display_width, display_height)
        self.window.show()
        
        print(f"Display: {frame_width}x{frame_height} -> {display_width}x{display_height} (scale: {self.scale_factor:.2f})")
    
    def show_frame(self, frame: np.ndarray) -> bool:
        if not HAS_PYQT5 or self.window is None:
            cv2.imshow(self.window_title, frame)
            return cv2.waitKey(1) & 0xFF != ord('q')
        
        if self.should_quit:
            return False
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if self.scale_factor < 1.0 and self.display_size:
            rgb_frame = cv2.resize(rgb_frame, self.display_size, interpolation=cv2.INTER_LANCZOS4)
        
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)
        
        self.app.processEvents()
        
        return not self.should_quit
    
    def close(self):
        """Proper PyQt5 cleanup to prevent freeze/shrink"""
        if not HAS_PYQT5 or self.window is None:
            return
            
        try:
            # ✅ 1. Stop accepting new frames
            self.should_quit = True
            
            # ✅ 2. Process any pending events
            if self.app:
                self.app.processEvents()
            
            # ✅ 3. Hide window first (prevents shrinking)
            if self.window:
                self.window.hide()
                
            # ✅ 4. Process hide events
            if self.app:
                self.app.processEvents()
            
            # ✅ 5. Close window
            if self.window:
                self.window.close()
                
            # ✅ 6. Delete window object
            if self.window:
                self.window.deleteLater()
                self.window = None
            
            # ✅ 7. Final event processing
            if self.app:
                self.app.processEvents()
                
            print("[PyQt5] Window closed properly")
            
        except Exception as e:
            print(f"[PyQt5] Cleanup error: {e}")


class OpenCVDisplay:
    def __init__(self, window_title: str = "Analysis", max_width: int = 1920, max_height: int = 1080):
        self.window_title = window_title
        self.max_width = max_width
        self.max_height = max_height
        self.scale_factor = 1.0
        self.display_size = None
        self.native_width = 0
        self.native_height = 0
    
    def setup(self, frame_width: int, frame_height: int):
        self.native_width = frame_width
        self.native_height = frame_height
        
        cv2.namedWindow(self.window_title, cv2.WINDOW_NORMAL)
        
        scale_w = self.max_width / frame_width
        scale_h = self.max_height / frame_height
        self.scale_factor = min(scale_w, scale_h, 1.0)
        
        display_width = int(frame_width * self.scale_factor)
        display_height = int(frame_height * self.scale_factor)
        self.display_size = (display_width, display_height)
        
        cv2.resizeWindow(self.window_title, display_width, display_height)
        
        try:
            cv2.setWindowProperty(self.window_title, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
        except:
            pass
        
        if self.scale_factor < 1.0:
            print(f"Display: {frame_width}x{frame_height} -> {display_width}x{display_height} (scale: {self.scale_factor:.2f})")
        else:
            print(f"Display: {frame_width}x{frame_height} (native)")
    
    def show_frame(self, frame: np.ndarray) -> bool:
        display_frame = frame
        
        if self.scale_factor < 1.0 and self.display_size:
            display_frame = cv2.resize(frame, self.display_size, interpolation=cv2.INTER_LANCZOS4)
        
        cv2.imshow(self.window_title, display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            return False
        return True
    
    def close(self):
        cv2.destroyAllWindows()


def test_pyqt5() -> bool:
    if not HAS_PYQT5:
        return False
    try:
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        test_label = QLabel("test")
        test_label.close()
        return True
    except Exception as e:
        print(f"PyQt5 test failed: {e}")
        return False


def create_display(window_title: str = "Analysis", prefer_pyqt: bool = False) -> 'HighQualityDisplay | OpenCVDisplay':
    if prefer_pyqt and HAS_PYQT5:
        try:
            if test_pyqt5():
                print("PyQt5 display active")
                return HighQualityDisplay(window_title)
        except Exception as e:
            print(f"PyQt5 error: {e}")
    
    print("OpenCV display active (LANCZOS4)")
    return OpenCVDisplay(window_title)