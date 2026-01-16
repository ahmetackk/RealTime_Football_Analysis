import cv2
import threading
import queue
import time

class ThreadedVideoCapture:
    def __init__(self, source, queue_size=4):
        self.cap = cv2.VideoCapture(source)
        self.q = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._reader)
        self.thread.daemon = True
        self.thread.start()

    def _reader(self):
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                break
            # Eğer kuyruk boşsa koy, doluysa işlemcinin rahatlamasını bekle
            if not self.q.full():
                self.q.put((ret, frame))
            else:
                time.sleep(0.005) 

    def read(self):
        # Kuyruktan hazır kareyi çek
        try:
            return self.q.get(timeout=1.0)
        except queue.Empty:
            return False, None

    def release(self):
        self.stop_event.set()
        self.thread.join()
        self.cap.release()
        
    def isOpened(self):
        return self.cap.isOpened()
        
    def get(self, prop):
        return self.cap.get(prop)