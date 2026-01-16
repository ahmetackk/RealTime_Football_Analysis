import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import os

class FootballAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Football Analysis - Video Analyzer")
        self.root.geometry("700x800")  # ✅ Optimized for smaller console
        
        self.source_video_path = tk.StringVar(value="")
        self.target_video_path = tk.StringVar(value="")
        self.device = tk.StringVar(value="cuda")
        self.mode = tk.StringVar(value="stream")
        self.action_mode = tk.StringVar(value="supervised")  # ✅ Changed: supervised
        self.execution_mode = tk.StringVar(value="async")  # ✅ Changed: async
        self.display_backend = tk.StringVar(value="pyqt5")  # ✅ Changed: pyqt5
        self.debug = tk.BooleanVar(value=True)  # ✅ Changed: True
        self.inference_size = tk.IntVar(value=960)
        
        self.setup_ui()
        
    def setup_ui(self):
        main_frame = tk.Frame(self.root, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        title_label = tk.Label(main_frame, text="Football Video Analysis", 
                               font=("Helvetica", 18, "bold"))
        title_label.pack(pady=(0, 20))
        
        video_frame = tk.LabelFrame(main_frame, text="Video Selection", 
                                   font=("Helvetica", 12, "bold"), padx=10, pady=10)
        video_frame.pack(fill=tk.X, pady=5)
        
        source_frame = tk.Frame(video_frame)
        source_frame.pack(fill=tk.X, pady=5)
        tk.Label(source_frame, text="Source Video:", width=15, anchor="w").pack(side=tk.LEFT)
        tk.Entry(source_frame, textvariable=self.source_video_path, 
                state="readonly", width=40).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.source_browse_button = tk.Button(source_frame, text="Browse...", 
                 command=self.browse_source_video)
        self.source_browse_button.pack(side=tk.LEFT, padx=5)
        
        target_frame = tk.Frame(video_frame)
        target_frame.pack(fill=tk.X, pady=5)
        tk.Label(target_frame, text="Output Video:", width=15, anchor="w").pack(side=tk.LEFT)
        self.target_entry = tk.Entry(target_frame, textvariable=self.target_video_path, 
                                     width=40)
        self.target_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.target_browse_button = tk.Button(target_frame, text="Browse...", 
                 command=self.browse_target_video)
        self.target_browse_button.pack(side=tk.LEFT, padx=5)
        
        settings_frame = tk.LabelFrame(main_frame, text="Settings", 
                                     font=("Helvetica", 12, "bold"), padx=10, pady=10)
        settings_frame.pack(fill=tk.X, pady=5)
        
        device_frame = tk.Frame(settings_frame)
        device_frame.pack(fill=tk.X, pady=5)
        tk.Label(device_frame, text="Device:", width=15, anchor="w").pack(side=tk.LEFT)
        device_combo = ttk.Combobox(device_frame, textvariable=self.device, 
                                   values=["cuda", "cpu"], state="readonly", width=37)
        device_combo.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        mode_frame = tk.Frame(settings_frame)
        mode_frame.pack(fill=tk.X, pady=5)
        tk.Label(mode_frame, text="Run Mode:", width=15, anchor="w").pack(side=tk.LEFT)
        mode_combo = ttk.Combobox(mode_frame, textvariable=self.mode, 
                                 values=["stream", "save", "unsupervised_analysis"], state="readonly", width=37)
        mode_combo.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        mode_combo.bind("<<ComboboxSelected>>", self.on_mode_change)
        
        mode_help = tk.Label(settings_frame, 
                            text="  stream: Live view | save: Record video | unsupervised_analysis: AI Discovery (no display)", 
                            font=("Helvetica", 8), fg="gray")
        mode_help.pack(anchor=tk.W)
        
        # ✅ Unsupervised Speed Profile Selector
        self.unsupervised_speed = tk.StringVar(value="medium")
        speed_frame = tk.Frame(settings_frame)
        speed_frame.pack(fill=tk.X, pady=5)
        tk.Label(speed_frame, text="Unsupervised Speed:", width=15, anchor="w").pack(side=tk.LEFT)
        speed_combo = ttk.Combobox(speed_frame, textvariable=self.unsupervised_speed,
                                   values=["fast", "medium", "slow"], state="readonly", width=37)
        speed_combo.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        speed_help = tk.Label(settings_frame,
                             text="  fast: ~2min | medium: ~5min | slow: ~15min (only for unsupervised_analysis mode)",
                             font=("Helvetica", 8), fg="gray")
        speed_help.pack(anchor=tk.W)

        # ✅ Action mode removed - always supervised
        # ✅ Inference size removed - always 640
        
        exec_mode_frame = tk.Frame(settings_frame)
        exec_mode_frame.pack(fill=tk.X, pady=5)
        tk.Label(exec_mode_frame, text="Execution Mode:", width=15, anchor="w").pack(side=tk.LEFT)
        
        exec_combo = ttk.Combobox(exec_mode_frame, textvariable=self.execution_mode,
                                   values=["sync", "async"], state="readonly", width=37)
        exec_combo.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        exec_help = tk.Label(settings_frame, text="  sync: Synchronous (stable) | async: Asynchronous (parallel/fast)", 
                            font=("Helvetica", 8), fg="gray")
        exec_help.pack(anchor=tk.W)

        display_frame = tk.Frame(settings_frame)
        display_frame.pack(fill=tk.X, pady=5)
        tk.Label(display_frame, text="Display Backend:", width=15, anchor="w").pack(side=tk.LEFT)
        
        display_combo = ttk.Combobox(display_frame, textvariable=self.display_backend,
                                    values=["opencv", "pyqt5"], state="readonly", width=37)
        display_combo.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        display_help = tk.Label(settings_frame, text="  opencv: Stable | pyqt5: High quality (stream mode only)", 
                               font=("Helvetica", 8), fg="gray")
        display_help.pack(anchor=tk.W)
        
        debug_frame = tk.Frame(settings_frame)
        debug_frame.pack(fill=tk.X, pady=5)
        tk.Label(debug_frame, text="Debug:", width=15, anchor="w").pack(side=tk.LEFT)
        tk.Checkbutton(debug_frame, text="Enable Performance Profiling", 
                      variable=self.debug).pack(side=tk.LEFT, padx=5)
        
        progress_frame = tk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, pady=10)
        
        self.progress_label = tk.Label(progress_frame, text="", 
                                      font=("Helvetica", 9))
        self.progress_label.pack(anchor=tk.W)
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate', 
                                           length=400)
        self.progress_bar.pack(fill=tk.X, pady=(5, 0))
        
        # ✅ Console Output Window
        console_frame = tk.LabelFrame(main_frame, text="Console Output", 
                                     font=("Helvetica", 10, "bold"), padx=5, pady=5)
        console_frame.pack(fill=tk.BOTH, pady=10, expand=True)
        
        # Scrollbar
        console_scroll = tk.Scrollbar(console_frame)
        console_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Console text widget
        self.console_text = tk.Text(
            console_frame, 
            height=8,  # ✅ Reduced from 12 to 8
            bg="#1e1e1e",      # Dark background
            fg="#00ff00",      # Green text
            font=("Consolas", 9),
            wrap=tk.WORD,
            yscrollcommand=console_scroll.set,
            state=tk.DISABLED
        )
        self.console_text.pack(fill=tk.BOTH, expand=True)
        console_scroll.config(command=self.console_text.yview)
        
        # Initial message
        self.log_to_console("Ready. Select mode and click START ANALYSIS.")
        
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=20)
        
        # ✅ Much larger and more visible start button
        self.start_button = tk.Button(
            button_frame, 
            text="▶ START ANALYSIS", 
            font=("Helvetica", 16, "bold"),
            bg="#4CAF50", 
            fg="white",
            command=self.start_analysis, 
            height=3,
            relief=tk.RAISED,
            bd=5,
            cursor="hand2"
        )
        self.start_button.pack(fill=tk.BOTH, padx=30, pady=10, ipady=15)
        
        self.status_label = tk.Label(main_frame, text="Ready", 
                                    relief=tk.SUNKEN, anchor=tk.W, 
                                    font=("Helvetica", 9))
        self.status_label.pack(fill=tk.X, pady=(10, 0))
        
        self.on_mode_change()
        
        self.progress_callback = None
        self.total_frames = 0
        self.current_frame = 0
        
    def browse_source_video(self):
        filename = filedialog.askopenfilename(
            title="Select Source Video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("MP4 files", "*.mp4"),
                ("AVI files", "*.avi"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.source_video_path.set(filename)
            self.update_output_path()
    
    def update_output_path(self):
        source = self.source_video_path.get()
        if source:
            source_dir = os.path.dirname(os.path.abspath(source))
            parent_dir = os.path.dirname(source_dir)
            results_dir = os.path.join(parent_dir, "results")
            os.makedirs(results_dir, exist_ok=True)
            
            base_name = os.path.splitext(os.path.basename(source))[0]
            default_output = os.path.join(results_dir, f"{base_name}_analyzed.mp4")  # ✅ Always supervised
            self.target_video_path.set(default_output)
    
    def browse_target_video(self):
        filename = filedialog.asksaveasfilename(
            title="Save Output Video As",
            defaultextension=".mp4",
            filetypes=[
                ("MP4 files", "*.mp4"),
                ("AVI files", "*.avi"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.target_video_path.set(filename)
    
    def on_mode_change(self, event=None):
        mode = self.mode.get()
        
        if mode == "unsupervised_analysis":
            # ✅ Unsupervised mode: disable video selection, enable start button
            self.source_browse_button.config(state="disabled")
            self.target_entry.config(state="disabled")
            self.target_browse_button.config(state="disabled")
            
            # Clear paths (not needed for unsupervised)
            self.source_video_path.set("N/A (uses dataset)")
            self.target_video_path.set("N/A (outputs to unsupervised/results)")
            
            # Enable start button
            self.start_button.config(state="normal")
            
        elif mode == "save":
            # Save mode: both enabled
            self.source_browse_button.config(state="normal")
            self.target_entry.config(state="normal")
            self.target_browse_button.config(state="normal")
            
            # Restore paths if cleared
            if self.source_video_path.get() == "N/A (uses dataset)":
                self.source_video_path.set("")
            if self.target_video_path.get().startswith("N/A"):
                self.target_video_path.set("")
            
            if self.source_video_path.get():
                self.update_output_path()
                
        else:  # stream mode
            # Stream mode: source enabled, target disabled
            self.source_browse_button.config(state="normal")
            self.target_entry.config(state="readonly")
            self.target_browse_button.config(state="disabled")
            
            # Restore paths if cleared
            if self.source_video_path.get() == "N/A (uses dataset)":
                self.source_video_path.set("")
            if self.target_video_path.get().startswith("N/A"):
                self.target_video_path.set("")
    
    def validate_inputs(self):
        mode = self.mode.get()
        
        # ✅ Unsupervised mode: no video input needed (uses dataset)
        if mode == "unsupervised_analysis":
            # Check if unsupervised directory exists
            unsupervised_dir = os.path.join(os.getcwd(), "unsupervised")
            if not os.path.exists(unsupervised_dir):
                messagebox.showerror("Error", 
                    "Unsupervised directory not found!\n\n"
                    "Please ensure 'unsupervised/' folder exists with:\n"
                    "• data/ (val_tactical_data.h5)\n"
                    "• models/ (action_recognition.pt)\n"
                    "• videos/\n"
                    "• utils/, models/ scripts")
                return False
            return True
        
        # ✅ Normal modes: validate video paths
        if not self.source_video_path.get():
            messagebox.showerror("Error", "Please select a source video file.")
            return False
        
        if not os.path.exists(self.source_video_path.get()):
            messagebox.showerror("Error", "Source video file does not exist.")
            return False
        
        if mode == "save":
            if not self.target_video_path.get():
                messagebox.showerror("Error", "Please specify an output path for save mode.")
                return False
            
            output_dir = os.path.dirname(self.target_video_path.get())
            if output_dir and not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir, exist_ok=True)
                except Exception as e:
                    messagebox.showerror("Error", f"Cannot create output directory: {e}")
                    return False
        
        return True
    
    def update_progress(self, current, total):
        self.current_frame = current
        self.total_frames = total
        if total > 0:
            progress = int((current / total) * 100)
            self.progress_bar['value'] = progress
            self.progress_label.config(
                text=f"Processing: {current}/{total} frames ({progress}%)"
            )
        self.root.update_idletasks()
    
    def reset_progress(self):
        self.progress_bar['value'] = 0
        self.progress_label.config(text="")
        self.current_frame = 0
        self.total_frames = 0
    
    def log_to_console(self, message, color=None):
        """Add message to console output"""
        self.console_text.config(state=tk.NORMAL)
        
        # Add timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        
        # Insert message
        self.console_text.insert(tk.END, f"[{timestamp}] {message}\n")
        
        # Auto-scroll to bottom
        self.console_text.see(tk.END)
        
        self.console_text.config(state=tk.DISABLED)
        self.root.update_idletasks()
    
    def clear_console(self):
        """Clear console output"""
        self.console_text.config(state=tk.NORMAL)
        self.console_text.delete(1.0, tk.END)
        self.console_text.config(state=tk.DISABLED)
    
    def start_analysis(self):
        if not self.validate_inputs():
            return
        
        self.start_button.config(state="disabled")
        self.status_label.config(text="Starting analysis...")
        self.reset_progress()
        self.root.update()
        
        mode = self.mode.get()
        
        # ✅ For unsupervised mode, source/target are not needed
        if mode == "unsupervised_analysis":
            source = None  # Not needed
            target = None  # Not needed
        else:
            source = self.source_video_path.get()
            target = self.target_video_path.get() if mode == "save" else None
        
        device = self.device.get()
        debug = self.debug.get()
        inference_size = 640  # ✅ Fixed: always 640
        action_mode = "supervised"  # ✅ Fixed: always supervised
        execution_mode = self.execution_mode.get()
        display_backend = self.display_backend.get()
        
        def progress_callback(current, total):
            self.root.after(0, lambda: self.update_progress(current, total))
        
        def run_in_thread():
            try:
                self.status_label.config(text=f"Analysis running ({execution_mode}/{display_backend})...")
                self.root.update()
                
                # ✅ UNSUPERVISED ANALYSIS MODE - 4-Stage Pipeline
                if mode == "unsupervised_analysis":
                    import subprocess
                    import sys
                    
                    self.root.after(0, lambda: self.clear_console())
                    self.root.after(0, lambda: self.log_to_console("="*60))
                    self.root.after(0, lambda: self.log_to_console("UNSUPERVISED ANALYSIS PIPELINE"))
                    self.root.after(0, lambda: self.log_to_console("="*60))
                    
                    unsupervised_dir = os.path.join(os.getcwd(), "unsupervised")
                    
                    if not os.path.exists(unsupervised_dir):
                        raise Exception("Unsupervised directory not found! Please ensure 'unsupervised/' folder exists.")
                    
                    # ✅ Save speed profile config for scripts to read
                    from unsupervised_speed_config import save_current_profile
                    speed_profile = save_current_profile(self.unsupervised_speed.get())
                    
                    self.root.after(0, lambda: self.log_to_console(f"Speed Profile: {speed_profile['name']}"))
                    self.root.after(0, lambda: self.log_to_console(f"  {speed_profile['description']}"))
                    
                    # Save config to unsupervised directory
                    import json
                    config_path = os.path.join(unsupervised_dir, "speed_config.json")
                    with open(config_path, "w") as f:
                        json.dump(speed_profile, f, indent=2)
                    
                    # ====================================================================
                    # STAGE 1: Unsupervised Clustering
                    # ====================================================================
                    self.root.after(0, lambda: self.log_to_console(""))
                    self.root.after(0, lambda: self.log_to_console("STAGE 1/4: Unsupervised Clustering"))
                    self.root.after(0, lambda: self.log_to_console("-"*60))
                    self.root.after(0, lambda: self.status_label.config(
                        text="Stage 1/4: Clustering analysis..."))
                    self.root.after(0, lambda: self.progress_bar.config(value=10))
                    
                    stage1_script = os.path.join(unsupervised_dir, "unsupervised.py")
                    
                    if not os.path.exists(stage1_script):
                        raise Exception(f"unsupervised.py not found at {stage1_script}")
                    
                    self.root.after(0, lambda: self.log_to_console("Loading model and dataset..."))
                    
                    # Run subprocess with real-time output
                    process1 = subprocess.Popen(
                        [sys.executable, stage1_script],
                        cwd=unsupervised_dir,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        universal_newlines=True
                    )
                    
                    # Read output line by line
                    for line in process1.stdout:
                        line = line.strip()
                        if line:
                            self.root.after(0, lambda l=line: self.log_to_console(l))
                    
                    process1.wait()
                    
                    if process1.returncode != 0:
                        raise Exception(f"Stage 1 failed with return code {process1.returncode}")
                    
                    self.root.after(0, lambda: self.log_to_console("[OK] Stage 1 Complete!"))
                    
                    # ====================================================================
                    # STAGE 2: Known-Unknown Clip Generation
                    # ====================================================================
                    self.root.after(0, lambda: self.log_to_console(""))
                    self.root.after(0, lambda: self.log_to_console("STAGE 2/4: Known-Unknown Clips"))
                    self.root.after(0, lambda: self.log_to_console("-"*60))
                    self.root.after(0, lambda: self.status_label.config(
                        text="Stage 2/4: Generating clips..."))
                    self.root.after(0, lambda: self.progress_bar.config(value=35))
                    
                    stage2_script = os.path.join(unsupervised_dir, "unsupervised_clip_generate.py")
                    
                    if not os.path.exists(stage2_script):
                        raise Exception(f"unsupervised_clip_generate.py not found at {stage2_script}")
                    
                    process2 = subprocess.Popen(
                        [sys.executable, stage2_script],
                        cwd=unsupervised_dir,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        universal_newlines=True
                    )
                    
                    for line in process2.stdout:
                        line = line.strip()
                        if line:
                            self.root.after(0, lambda l=line: self.log_to_console(l))
                    
                    process2.wait()
                    if process2.returncode != 0:
                        raise Exception(f"Stage 2 failed with return code {process2.returncode}")
                    
                    self.root.after(0, lambda: self.log_to_console("[OK] Stage 2 Complete!"))
                    
                    # ====================================================================
                    # STAGE 3: New Action Extraction
                    # ====================================================================
                    self.root.after(0, lambda: self.log_to_console(""))
                    self.root.after(0, lambda: self.log_to_console("STAGE 3/4: New Action Extraction"))
                    self.root.after(0, lambda: self.log_to_console("-"*60))
                    self.root.after(0, lambda: self.status_label.config(
                        text="Stage 3/4: Extracting new actions..."))
                    self.root.after(0, lambda: self.progress_bar.config(value=60))
                    
                    stage3_script = os.path.join(unsupervised_dir, "unsupervised_new_action_full_match.py")
                    
                    if not os.path.exists(stage3_script):
                        raise Exception(f"unsupervised_new_action_full_match.py not found at {stage3_script}")
                    
                    process3 = subprocess.Popen(
                        [sys.executable, stage3_script],
                        cwd=unsupervised_dir,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        universal_newlines=True
                    )
                    
                    for line in process3.stdout:
                        line = line.strip()
                        if line:
                            self.root.after(0, lambda l=line: self.log_to_console(l))
                    
                    process3.wait()
                    if process3.returncode != 0:
                        raise Exception(f"Stage 3 failed with return code {process3.returncode}")
                    
                    self.root.after(0, lambda: self.log_to_console("[OK] Stage 3 Complete!"))
                    
                    # ====================================================================
                    # STAGE 4: New Action Clip Generation
                    # ====================================================================
                    self.root.after(0, lambda: self.log_to_console(""))
                    self.root.after(0, lambda: self.log_to_console("STAGE 4/4: New Action Clips"))
                    self.root.after(0, lambda: self.log_to_console("-"*60))
                    self.root.after(0, lambda: self.status_label.config(
                        text="Stage 4/4: Generating new action clips..."))
                    self.root.after(0, lambda: self.progress_bar.config(value=85))
                    
                    stage4_script = os.path.join(unsupervised_dir, "unsupervised_new_action_clip_generate.py")
                    
                    if not os.path.exists(stage4_script):
                        raise Exception(f"unsupervised_new_action_clip_generate.py not found at {stage4_script}")
                    
                    process4 = subprocess.Popen(
                        [sys.executable, stage4_script],
                        cwd=unsupervised_dir,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        universal_newlines=True
                    )
                    
                    for line in process4.stdout:
                        line = line.strip()
                        if line:
                            self.root.after(0, lambda l=line: self.log_to_console(l))
                    
                    process4.wait()
                    if process4.returncode != 0:
                        raise Exception(f"Stage 4 failed with return code {process4.returncode}")
                    
                    self.root.after(0, lambda: self.log_to_console("[OK] Stage 4 Complete!"))
                    
                    # ====================================================================
                    # COMPLETE
                    # ====================================================================
                    self.root.after(0, lambda: self.log_to_console(""))
                    self.root.after(0, lambda: self.log_to_console("="*60))
                    self.root.after(0, lambda: self.log_to_console("[SUCCESS] ALL STAGES COMPLETED SUCCESSFULLY!"))
                    self.root.after(0, lambda: self.log_to_console("="*60))
                    self.root.after(0, lambda: self.progress_bar.config(value=100))
                    self.root.after(0, lambda: self.status_label.config(
                        text="✓ Unsupervised analysis complete!"))
                    
                    results_msg = (
                        "✓ Unsupervised Analysis Complete!\n\n"
                        "Results saved in unsupervised/results/ folder:\n\n"
                        "📊 Graphs:\n"
                        "  • discovery_cm_counts.png (Confusion matrix)\n"
                        "  • discovery_cm_normalized.png (Normalized)\n"
                        "  • discovery_tsne.png (t-SNE visualization)\n"
                        "  • discovery_class_distribution.png\n\n"
                        "🎬 Video Clips:\n"
                        "  • final_demo_detailed_videos_all/ (Known-unknown actions)\n"
                        "  • NEW_ACTION_CLASSES/ (Newly discovered actions)\n\n"
                        "📁 Discovery State:\n"
                        "  • discovery_state.pkl (PCA + K-Means models)"
                    )
                    
                    self.root.after(0, lambda: messagebox.showinfo("Success", results_msg))
                    
                else:
                    # ✅ NORMAL MODE (stream/save)
                    # Lazy import to avoid circular import
                    from footballanalysis import main as run_analysis
                    
                    run_analysis(source, target, device, mode, debug, inference_size, 
                               action_mode=action_mode,
                               execution_mode=execution_mode,
                               display_backend=display_backend,
                               progress_callback=progress_callback)
                    
                    self.root.after(0, lambda: self.progress_bar.config(value=100))
                    self.root.after(0, lambda: self.status_label.config(
                        text="Analysis completed successfully!"))
                    self.root.after(0, lambda: messagebox.showinfo(
                        "Success", "Analysis completed successfully!"))
                
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda: self.status_label.config(
                    text=f"Error: {error_msg[:50]}..."))
                self.root.after(0, lambda: messagebox.showerror(
                    "Error", f"Analysis failed:\n{error_msg}"))
            
            finally:
                self.root.after(0, lambda: self.start_button.config(state="normal"))
        
        thread = threading.Thread(target=run_in_thread, daemon=True)
        thread.start()

def launch_gui():
    root = tk.Tk()
    app = FootballAnalysisGUI(root)
    root.mainloop()

if __name__ == "__main__":
    launch_gui()