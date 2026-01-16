import time
from collections import defaultdict

_ENABLED = False
_timers = defaultdict(float)
_counts = defaultdict(int)
_starts = {}

_SCHEMA = {
    # 1. Player Logic
    "Player Logic (Total)":    ("Total Logic",      "Player Logic (AI)"),
    "Player-Inference (YOLO)": ("  ├─ Detection",   "Player Logic (AI)"),
    "Player-Tracking":         ("  ├─ Tracking",    "Player Logic (AI)"),
    "Player-Crop":             ("  ├─ Crop Ops",    "Player Logic (AI)"),
    "Player-TeamLogic":        ("  ├─ Team Logic",  "Player Logic (AI)"),
    "Player-JerseyLogic":      ("  ├─ Jersey Logic","Player Logic (AI)"),
    "Player-Loops":            ("  └─ Loop Overhead","Player Logic (AI)"),

    # 2. Ball Detection
    "Ball Detection":          ("Total Ball",       "Ball Detection"),
    "Ball-Inference":          ("  └─ Inference",   "Ball Detection"),

    # 3. Pitch Detection
    "Pitch Detection":         ("Total Pitch",      "Pitch Detection"),
    "Pitch-Inference":         ("  ├─ Inference",   "Pitch Detection"),
    "Pitch-PostProcess":       ("  └─ PostProcess", "Pitch Detection"),

    # 4. Action Recognition
    "Action-Update":           ("Update Buffer",    "Action Recognition"),
    "Action-Predict (Total)":  ("Prediction Loop",  "Action Recognition"),
    "Action-PrepData":         ("  ├─ Prep Data",   "Action Recognition"),
    "Action-Inference":        ("  ├─ Inference",   "Action Recognition"),
    "Action-PostProcess":      ("  └─ PostProcess", "Action Recognition"),

    # 5. Visuals & IO (GÜNCELLENEN KISIM)
    "Visualization":           ("Total Draw Ops",   "Visuals & I/O"), # YENİ: Ana çizim bloğu
    "Visuals-Annotate":        ("  ├─ Annotate",    "Visuals & I/O"), # Detay (Varsa)
    "Visuals-Radar":           ("  └─ Radar",       "Visuals & I/O"), # Detay (Varsa)
    "Video-Write/Show":        ("Display/Write",    "Visuals & I/O"), # Gösterme/Kaydetme
    "Video-Read":              ("Frame Read",       "Visuals & I/O"), # Okuma (Opsiyonel)
    
    # Global
    "main_loop":               ("Total Loop Time",  "Global")
}

_GROUP_ORDER = ["Player Logic (AI)", "Ball Detection", "Pitch Detection", "Action Recognition", "Visuals & I/O"]

def setup(enabled=True):
    global _ENABLED
    _ENABLED = enabled
    if enabled: print("[PROFILER] Sistem AKTİF. Performans verileri toplanıyor...")

def reset():
    global _timers, _counts, _starts
    _timers.clear()
    _counts.clear()
    _starts.clear()

def start(name):
    if not _ENABLED: return
    _starts[name] = time.perf_counter()

def stop(name):
    if not _ENABLED: return
    if name in _starts:
        duration = time.perf_counter() - _starts[name]
        _timers[name] += duration
        _counts[name] += 1
        del _starts[name]

def get_report(total_frames_processed=1):
    if not _timers: return "Veri yok."
    
    if total_frames_processed < 1: total_frames_processed = 1

    lines = []
    lines.append(f"{'='*95}")
    lines.append(f"{'OPERATION / SUB-OPERATION':<45} | {'PER FRAME (ms)':<15} | {'PER CALL (ms)':<15} | {'COUNT':<5}")
    lines.append(f"{'-'*95}")

    grouped_data = defaultdict(list)
    
    # Verileri Schema'ya göre grupla
    for key, (display_name, group) in _SCHEMA.items():
        if key in _timers:
            total_sec = _timers[key]
            count = _counts[key]
            
            ms_per_frame = (total_sec / total_frames_processed) * 1000
            ms_per_call = (total_sec / count) * 1000 if count > 0 else 0
            
            grouped_data[group].append({
                "name": display_name,
                "ms_frame": ms_per_frame,
                "ms_call": ms_per_call,
                "count": count,
                "raw_name": display_name # Sıralama için
            })

    for group_name in _GROUP_ORDER:
        if group_name not in grouped_data: continue
        
        lines.append(f"[{group_name}]")
        
        items = grouped_data[group_name]
        # Sıralama: Önce "Total" içerenler, sonra süreye göre
        items.sort(key=lambda x: (not "Total" in x['raw_name'], -x['ms_frame']))

        for item in items:
            lines.append(
                f"{item['name']:<45} | {item['ms_frame']:<15.2f} | {item['ms_call']:<15.2f} | {item['count']:<5}"
            )
        lines.append(f"{'-'*95}")

    if "main_loop" in _timers:
        total_time = _timers["main_loop"]
        real_fps = total_frames_processed / total_time if total_time > 0 else 0
        avg_ms = (total_time / total_frames_processed) * 1000
        
        lines.append(f"TOTAL REAL TIME: {total_time:.2f} s")
        lines.append(f"AVG FRAME TIME : {avg_ms:.2f} ms")
        lines.append(f"REAL FPS       : {real_fps:.2f}")
    
    lines.append(f"{'='*95}")
    return "\n".join(lines)

def save_report(path, total_frames):
    if not _ENABLED: return
    content = get_report(total_frames)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[PROFILER] Rapor kaydedildi: {path}")
    except Exception as e:
        print(f"[PROFILER] Hata: {e}")