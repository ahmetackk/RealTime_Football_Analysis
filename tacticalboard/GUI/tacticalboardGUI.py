import supervision as sv
import cv2
import numpy as np
import tkinter as tk
from tkinter import simpledialog, messagebox, Listbox, Scrollbar, Frame, Canvas, Button, Entry, Label, ttk
from PIL import Image, ImageTk
import uuid
import pickle
from tkinter import filedialog

from tacticalboard.configs.soccer import SoccerPitchConfiguration
from tacticalboard.annotators.soccer import draw_pitch
from tacticalboard.simulation.soccer import Simulation, Player, Ball, Action


class TacticalBoardGUI:
    
    PLAYER_RADIUS = 16
    BALL_RADIUS = 10
    CONFIG = SoccerPitchConfiguration()
    PADDING = 50 
    DRAW_SCALE = 0.1
    SIM_SCALE = 1.0 / DRAW_SCALE
    TEAM_GUI_COLORS = {
        0: sv.Color.BLUE.as_hex(),
        1: sv.Color.RED.as_hex(),
        9: '#FFFF00',
    }
    CLASS_GUI_COLORS = {
        1: '#FF8C00',
        3: '#FFFF00',
    }
    
    def __init__(self, root):
        self.root = root
        self.root.title("Tactical Board")

        self.scenes = []  
        self.actions = []
        self.current_canvas_items = {} 
        self._drag_data = {"x": 0, "y": 0, "item": None}
        self.default_jersey_number = [1, 1]
        self.main_frame = Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas_frame = Frame(self.main_frame, relief=tk.SUNKEN, borderwidth=1)
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.controls_frame = Frame(self.main_frame, width=400)
        self.controls_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        self.controls_frame.pack_propagate(False)
        self.pitch_image_np = draw_pitch(config=self.CONFIG, padding=self.PADDING, scale=self.DRAW_SCALE)
        self.canvas_height, self.canvas_width = self.pitch_image_np.shape[:2]
        self.canvas = Canvas(self.canvas_frame, width=self.canvas_width, height=self.canvas_height, bg="black")
        self.canvas.pack(anchor=tk.CENTER, expand=True)
        self.setup_canvas_background()
        self.setup_canvas_bindings()
        
        self.player_frame = Frame(self.controls_frame, relief=tk.RIDGE, borderwidth=1)
        self.player_frame.pack(fill=tk.X, pady=5)
        Label(self.player_frame, text="Player Controls", font=("Helvetica", 12, "bold")).pack(pady=2)
        Button(self.player_frame, text="Add Player (Blue Team)", 
               command=lambda: self.add_player(team_id=0)).pack(fill=tk.X, padx=5, pady=2)
        Button(self.player_frame, text="Add Player (Red Team)", 
               command=lambda: self.add_player(team_id=1)).pack(fill=tk.X, padx=5, pady=2)
        
        self.scene_frame = Frame(self.controls_frame, relief=tk.RIDGE, borderwidth=1)
        self.scene_frame.pack(fill=tk.X, pady=5)
        Label(self.scene_frame, text="Scene Controls", font=("Helvetica", 12, "bold")).pack(pady=2)
        self.time_frame = Frame(self.scene_frame)
        Label(self.time_frame, text="Time (s):").pack(side=tk.LEFT, padx=5)
        self.time_entry = Entry(self.time_frame, width=10)
        self.time_entry.pack(side=tk.LEFT, padx=5)
        self.time_frame.pack(pady=5)
        Button(self.scene_frame, text="Add Scene", 
               command=self.add_scene).pack(fill=tk.X, padx=5, pady=2)
        Button(self.scene_frame, text="Update Selected Scene", 
               command=self.update_scene).pack(fill=tk.X, padx=5, pady=2)
        Button(self.scene_frame, text="Delete Selected Scene", 
               command=self.delete_scene).pack(fill=tk.X, padx=5, pady=2)
        
        self.list_frame = Frame(self.controls_frame, relief=tk.RIDGE, borderwidth=1)
        self.list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        Label(self.list_frame, text="Scene List", font=("Helvetica", 12, "bold")).pack(pady=2)
        self.list_scrollbar = Scrollbar(self.list_frame, orient=tk.VERTICAL)
        self.scene_listbox = Listbox(self.list_frame, yscrollcommand=self.list_scrollbar.set, exportselection=False, height=6)
        self.list_scrollbar.config(command=self.scene_listbox.yview)
        self.list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.scene_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.scene_listbox.bind("<<ListboxSelect>>", self.load_scene_from_listbox)
        
        self.action_frame = Frame(self.controls_frame, relief=tk.RIDGE, borderwidth=1)
        self.action_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        Label(self.action_frame, text="Action Controls", font=("Helvetica", 12, "bold")).pack(pady=2)
        
        self.action_input_frame = Frame(self.action_frame)
        self.action_input_frame.pack(fill=tk.X, padx=5, pady=2)
        
        self.action_time_frame = Frame(self.action_input_frame)
        self.action_time_frame.pack(fill=tk.X, pady=1)
        Label(self.action_time_frame, text="Time (s):", width=10, anchor='w').pack(side=tk.LEFT)
        self.action_time_entry = Entry(self.action_time_frame, width=8)
        self.action_time_entry.pack(side=tk.LEFT, padx=2)
        self.action_time_entry.insert(0, "0.0")
        
        Label(self.action_time_frame, text="Duration:", width=8).pack(side=tk.LEFT)
        self.action_duration_entry = Entry(self.action_time_frame, width=5)
        self.action_duration_entry.pack(side=tk.LEFT, padx=2)
        self.action_duration_entry.insert(0, "3.0")
        
        # Type dropdown
        self.action_type_frame = Frame(self.action_input_frame)
        self.action_type_frame.pack(fill=tk.X, pady=1)
        Label(self.action_type_frame, text="Type:", width=10, anchor='w').pack(side=tk.LEFT)
        self.action_type_var = tk.StringVar(value="pass")
        self.action_type_combo = ttk.Combobox(self.action_type_frame, textvariable=self.action_type_var, 
                                               values=["pass", "shot", "goal", "tackle", "save", "foul", "corner", "offside"],
                                               state="readonly", width=12)
        self.action_type_combo.pack(side=tk.LEFT, padx=2)
        
        # Text input
        self.action_text_frame = Frame(self.action_input_frame)
        self.action_text_frame.pack(fill=tk.X, pady=1)
        Label(self.action_text_frame, text="Text:", width=10, anchor='w').pack(side=tk.LEFT)
        self.action_text_entry = Entry(self.action_text_frame, width=20)
        self.action_text_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        # Action buttons
        self.action_buttons_frame = Frame(self.action_frame)
        self.action_buttons_frame.pack(fill=tk.X, padx=5, pady=2)
        Button(self.action_buttons_frame, text="Add", width=6,
               command=self.add_action).pack(side=tk.LEFT, padx=2)
        Button(self.action_buttons_frame, text="Update", width=6,
               command=self.update_action).pack(side=tk.LEFT, padx=2)
        Button(self.action_buttons_frame, text="Delete", width=6,
               command=self.delete_action).pack(side=tk.LEFT, padx=2)
        
        # Action listbox
        self.action_list_frame = Frame(self.action_frame)
        self.action_list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=2)
        self.action_scrollbar = Scrollbar(self.action_list_frame, orient=tk.VERTICAL)
        self.action_listbox = Listbox(self.action_list_frame, yscrollcommand=self.action_scrollbar.set, 
                                       exportselection=False, height=4, font=("Helvetica", 9))
        self.action_scrollbar.config(command=self.action_listbox.yview)
        self.action_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.action_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.action_listbox.bind("<<ListboxSelect>>", self.on_action_select)
        
        self.sim_frame = Frame(self.controls_frame)
        self.sim_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        Button(self.sim_frame, text="Run Simulation", font=("Helvetica", 14, "bold"), 
               bg="green", fg="white", command=self.run_simulation).pack(fill=tk.X, padx=5, ipady=10)
        
        self.io_frame = Frame(self.controls_frame, relief=tk.RIDGE, borderwidth=1)
        self.io_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        Label(self.io_frame, text="Scenario I/O", font=("Helvetica", 12, "bold")).pack(pady=2)
        
        self.io_buttons_frame = Frame(self.io_frame)
        Button(self.io_buttons_frame, text="Save Scenario", 
               command=self.save_scenario).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=2)
        Button(self.io_buttons_frame, text="Load Scenario", 
               command=self.load_scenario).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=2)
        self.io_buttons_frame.pack(fill=tk.X)
        
        Button(self.io_frame, text="RESET ALL", font=("Helvetica", 10, "bold"),
               bg="#FF6347", fg="white", command=self.reset_all).pack(fill=tk.X, padx=5, pady=5)
        
        self._clear_canvas_items()
        self.time_entry.insert(0, "0.0")
    
    def setup_canvas_background(self):
        img_rgb = cv2.cvtColor(self.pitch_image_np, cv2.COLOR_BGR2RGB)
        self.pitch_photo = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))
        self.canvas.create_image(0, 0, image=self.pitch_photo, anchor=tk.NW)

    def setup_canvas_bindings(self):
        self.canvas.tag_bind("draggable", "<ButtonPress-1>", self.on_press)
        self.canvas.tag_bind("draggable", "<B1-Motion>", self.on_drag)
        self.canvas.tag_bind("draggable", "<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<Button-2>", self.on_right_click)

    def _clear_canvas_items(self):
        for item_id in self.current_canvas_items:
            self.canvas.delete(item_id)
        self.current_canvas_items = {}
        x, y = self.canvas_width / 2, self.canvas_height / 2
        r = self.BALL_RADIUS
        ball_id = self.canvas.create_oval(
            x - r, y - r, x + r, y + r, 
            fill="white", outline="black", width=2, tags=("draggable", "ball")
        )
        self.current_canvas_items[ball_id] = {'type': 'ball', 'player_id': 'ball', 'pair_id': None}

    def _draw_player_on_canvas(self, player_id, team_id, jersey, pos, class_id=2, tracker_id=None):
        x, y = pos
        r = self.PLAYER_RADIUS
        
        if class_id == 3:
            color = self.CLASS_GUI_COLORS.get(3, '#FFFF00')
        elif class_id == 1:
            color = self.CLASS_GUI_COLORS.get(1, '#FF8C00')
        else:
            color = self.TEAM_GUI_COLORS.get(team_id, "gray")
        
        if jersey and jersey > 0:
            display_text = str(jersey)
        elif tracker_id is not None:
            display_text = f"ID{tracker_id}"
        else:
            display_text = str(jersey) if jersey else ""
        
        oval_id = self.canvas.create_oval(
            x - r, y - r, x + r, y + r,
            fill=color, outline="black", width=2, tags=("draggable", "player")
        )
        text_id = self.canvas.create_text(
            x, y, text=display_text, 
            font=("Helvetica", 8 if len(display_text) > 2 else 10, "bold"), 
            fill="white", tags=("draggable", "player_text")
        )
        self.current_canvas_items[oval_id] = {
            'type': 'player', 'player_id': player_id, 'pair_id': text_id,
            'team': team_id, 'jersey': jersey, 'class_id': class_id, 'tracker_id': tracker_id
        }
        self.current_canvas_items[text_id] = {
            'type': 'text', 'player_id': player_id, 'pair_id': oval_id
        }
        ball_id = self.get_ball_canvas_id()
        if ball_id:
            self.canvas.tag_raise(ball_id)

    def _find_player_id(self, team_id, jersey_number):
        for scene in self.scenes:
            for player_data in scene['state']['players']:
                try:
                    p_team = int(player_data.get('team', -1)) if player_data.get('team') is not None else -1
                    p_jersey = int(player_data.get('jersey', -1)) if player_data.get('jersey') is not None else -1
                except (ValueError, TypeError):
                    continue
                if p_team == team_id and p_jersey == jersey_number:
                    return player_data['id']
        return None

    def add_player(self, team_id: int):
        jersey_number = simpledialog.askinteger("Input", "Enter Jersey Number:", 
                                                parent=self.root, minvalue=1, maxvalue=99, 
                                                initialvalue=self.default_jersey_number[team_id])
        if jersey_number is None:
            return
        for item_data in self.current_canvas_items.values():
            if item_data['type'] == 'player':
                if item_data['team'] == team_id and item_data['jersey'] == jersey_number:
                    messagebox.showerror("Duplicate Jersey", 
                                         f"Jersey {jersey_number} (Team {team_id}) is already on the pitch for this scene.")
                    return
        player_id = self._find_player_id(team_id, jersey_number)
        if player_id is None:
            player_id = str(uuid.uuid4())
        x, y = (self.canvas_width / 2) + np.random.randint(-50, 50), \
               (self.canvas_height / 2) + np.random.randint(-50, 50)
        self._draw_player_on_canvas(player_id, team_id, jersey_number, (x, y))
        team_jerseys = []
        for d in self.current_canvas_items.values():
            if d['type'] == 'player' and d['team'] == team_id:
                try:
                    team_jerseys.append(int(d['jersey']) if d['jersey'] is not None else 0)
                except (ValueError, TypeError):
                    team_jerseys.append(0)
        if team_jerseys:
            self.default_jersey_number[team_id] = max(team_jerseys) + 1
        else:
            self.default_jersey_number[team_id] = jersey_number + 1

    def on_press(self, event):
        item_id = self.canvas.find_closest(event.x, event.y)[0]
        if item_id not in self.current_canvas_items:
            return
        tags = self.canvas.gettags(item_id)
        if "draggable" not in tags:
            return
        item_info = self.current_canvas_items[item_id]
        drag_target_id = item_id
        if item_info['type'] == 'text':
            drag_target_id = item_info['pair_id']
        if drag_target_id:
            self._drag_data["item"] = drag_target_id
            self._drag_data["x"] = event.x
            self._drag_data["y"] = event.y
            self.canvas.tag_raise(drag_target_id)
            if self.current_canvas_items[drag_target_id]['type'] == 'player':
                text_id = self.current_canvas_items[drag_target_id]['pair_id']
                self.canvas.tag_raise(text_id)
            ball_id = self.get_ball_canvas_id()
            if ball_id:
                self.canvas.tag_raise(ball_id)

    def on_drag(self, event):
        oval_id = self._drag_data["item"]
        if oval_id is None:
            return
        dx = event.x - self._drag_data["x"]
        dy = event.y - self._drag_data["y"]
        self.canvas.move(oval_id, dx, dy)
        item_info = self.current_canvas_items.get(oval_id)
        if item_info and item_info['type'] == 'player':
            text_id = item_info['pair_id']
            self.canvas.move(text_id, dx, dy)
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y

    def on_release(self, event):
        self._drag_data["item"] = None
        self._drag_data["x"] = 0
        self._drag_data["y"] = 0

    def on_right_click(self, event):
        item_id = self.canvas.find_closest(event.x, event.y)[0]
        if item_id not in self.current_canvas_items:
            return
        item_info = self.current_canvas_items[item_id]
        if item_info['type'] == 'player':
            oval_id_to_del = item_id
            text_id_to_del = item_info['pair_id']
        elif item_info['type'] == 'text':
            text_id_to_del = item_id
            oval_id_to_del = item_info['pair_id']
        else:
            return
        self.canvas.delete(oval_id_to_del)
        self.canvas.delete(text_id_to_del)
        del self.current_canvas_items[oval_id_to_del]
        del self.current_canvas_items[text_id_to_del]

    def _get_current_timestamp(self):
        try:
            timestamp = float(self.time_entry.get())
            if timestamp < 0:
                raise ValueError
            return timestamp
        except ValueError:
            messagebox.showerror("Invalid Time", "Please enter a valid, non-negative number for the time.")
            return None

    def get_ball_canvas_id(self):
        for item_id, info in self.current_canvas_items.items():
            if info['type'] == 'ball':
                return item_id
        return None

    def _get_current_scene_state(self):
        state = {
            'ball_pos': (self.canvas_width / 2, self.canvas_height / 2),
            'players': []
        }
        for item_id in list(self.current_canvas_items.keys()):
            info = self.current_canvas_items[item_id]
            if info['type'] == 'text':
                continue
            if not self.canvas.find_withtag(item_id):
                try:
                    del self.current_canvas_items[item_id]
                except KeyError:
                    pass 
                continue 
            coords = self.canvas.coords(item_id)
            if not coords or len(coords) < 4:
                continue
            x_center = (coords[0] + coords[2]) / 2
            y_center = (coords[1] + coords[3]) / 2
            pos = (x_center, y_center)
            if info['type'] == 'ball':
                state['ball_pos'] = pos
            elif info['type'] == 'player':
                player_data = {
                    'id': info['player_id'],
                    'team': info['team'],
                    'jersey': info['jersey'],
                    'pos': pos,
                    'class_id': info.get('class_id', 2)
                }
                if info.get('tracker_id') is not None:
                    player_data['tracker_id'] = info['tracker_id']
                state['players'].append(player_data)
        return state

    def _update_scene_listbox(self):
        self.scene_listbox.delete(0, tk.END)
        self.scenes.sort(key=lambda s: s['timestamp'])
        for scene in self.scenes:
            listbox_id = f"Time: {scene['timestamp']:.2f}s"
            scene['listbox_id'] = listbox_id
            self.scene_listbox.insert(tk.END, listbox_id)

    def add_scene(self):
        timestamp = self._get_current_timestamp()
        if timestamp is None:
            return
        if any(scene['timestamp'] == timestamp for scene in self.scenes):
            messagebox.showerror("Duplicate Scene", 
                                 f"A scene at timestamp {timestamp}s already exists.")
            return
        current_state = self._get_current_scene_state()
        new_scene = {
            'timestamp': timestamp,
            'state': current_state,
            'listbox_id': ""
        }
        self.scenes.append(new_scene)
        self._update_scene_listbox()
        self.time_entry.delete(0, tk.END)
        self.time_entry.insert(0, str(round(timestamp + 1.0, 2)))

    def _get_selected_scene_index(self):
        try:
            return self.scene_listbox.curselection()[0]
        except IndexError:
            return None

    def load_scene(self, scene_index: int):
        if scene_index is None or scene_index >= len(self.scenes):
            self._clear_canvas_items()
            self.time_entry.delete(0, tk.END)
            self.time_entry.insert(0, "0.0")
            return
        scene = self.scenes[scene_index]
        state = scene['state']
        self.time_entry.delete(0, tk.END)
        self.time_entry.insert(0, str(scene['timestamp']))
        self._clear_canvas_items()
        for player_data in state['players']:
            try:
                jersey_val = int(player_data.get('jersey', 0)) if player_data.get('jersey') is not None else 0
            except (ValueError, TypeError):
                jersey_val = 0
            try:
                team_val = int(player_data.get('team', 0)) if player_data.get('team') is not None else 0
            except (ValueError, TypeError):
                team_val = 0
            try:
                class_val = int(player_data.get('class_id', 2)) if player_data.get('class_id') is not None else 2
            except (ValueError, TypeError):
                class_val = 2
            try:
                tracker_val = int(player_data.get('tracker_id')) if player_data.get('tracker_id') is not None else None
            except (ValueError, TypeError):
                tracker_val = None
            
            self._draw_player_on_canvas(
                player_id=player_data['id'],
                team_id=team_val,
                jersey=jersey_val,
                pos=player_data['pos'],
                class_id=class_val,
                tracker_id=tracker_val
            )
        ball_id = self.get_ball_canvas_id()
        if ball_id and state.get('ball_pos'):
            x, y = state['ball_pos']
            r = self.BALL_RADIUS
            self.canvas.coords(ball_id, x - r, y - r, x + r, y + r)
        if ball_id:
            self.canvas.tag_raise(ball_id)

    def load_scene_from_listbox(self, event):
        selected_index = self._get_selected_scene_index()
        if selected_index is not None:
            self.load_scene(selected_index)

    def update_scene(self):
        selected_index = self._get_selected_scene_index()
        if selected_index is None:
            messagebox.showwarning("No Scene Selected", "Please select a scene to update.")
            return
        timestamp = self._get_current_timestamp()
        if timestamp is None:
            return
        for i, scene in enumerate(self.scenes):
            if i != selected_index and scene['timestamp'] == timestamp:
                messagebox.showerror("Duplicate Scene", 
                                     f"Another scene at timestamp {timestamp}s already exists.")
                return
        scene_to_update = self.scenes[selected_index]
        scene_to_update['timestamp'] = timestamp
        scene_to_update['state'] = self._get_current_scene_state()
        self._update_scene_listbox()
        self.scene_listbox.selection_set(selected_index)
        
    def delete_scene(self):
        selected_index = self._get_selected_scene_index()
        if selected_index is None:
            messagebox.showwarning("No Scene Selected", "Please select a scene to delete.")
            return
        if messagebox.askyesno("Confirm Delete", "Are you sure you want to delete the selected scene?"):
            del self.scenes[selected_index]
            self._update_scene_listbox()
            if not self.scenes:
                self._clear_canvas_items()
                self.time_entry.delete(0, tk.END)
                self.time_entry.insert(0, "0.0")
                return
            index_to_load = max(0, selected_index - 1)
            if index_to_load < len(self.scenes):
                self.load_scene(index_to_load)
                self.scene_listbox.selection_set(index_to_load)
            else:
                self.load_scene(0)
                self.scene_listbox.selection_set(0)

    def _recalculate_default_jerseys(self):
        max_team_0 = 0
        max_team_1 = 0
        
        for scene in self.scenes:
            for player_data in scene['state']['players']:
                try:
                    jersey_val = int(player_data.get('jersey', 0)) if player_data.get('jersey') is not None else 0
                except (ValueError, TypeError):
                    jersey_val = 0
                
                try:
                    team_val = int(player_data.get('team', 0)) if player_data.get('team') is not None else 0
                except (ValueError, TypeError):
                    team_val = 0
                
                if team_val == 0:
                    max_team_0 = max(max_team_0, jersey_val)
                elif team_val == 1:
                    max_team_1 = max(max_team_1, jersey_val)
                    
        self.default_jersey_number[0] = max_team_0 + 1
        self.default_jersey_number[1] = max_team_1 + 1
        print(f"Default jerseys recalculated: Team 0 -> {self.default_jersey_number[0]}, Team 1 -> {self.default_jersey_number[1]}")

    def _update_action_listbox(self):
        self.action_listbox.delete(0, tk.END)
        self.actions.sort(key=lambda a: a['timestamp'])
        for action in self.actions:
            display_text = f"[{action['timestamp']:.1f}s] [{action['type']}] {action['text']}"
            self.action_listbox.insert(tk.END, display_text)
    
    def _get_selected_action_index(self):
        try:
            return self.action_listbox.curselection()[0]
        except IndexError:
            return None
    
    def on_action_select(self, event):
        selected_index = self._get_selected_action_index()
        if selected_index is None:
            return
        
        action = self.actions[selected_index]
        
        self.action_time_entry.delete(0, tk.END)
        self.action_time_entry.insert(0, str(action['timestamp']))
        
        self.action_duration_entry.delete(0, tk.END)
        self.action_duration_entry.insert(0, str(action.get('duration', 3.0)))
        
        self.action_type_var.set(action.get('type', 'pass'))
        
        self.action_text_entry.delete(0, tk.END)
        self.action_text_entry.insert(0, action.get('text', ''))
    
    def add_action(self):
        try:
            timestamp = float(self.action_time_entry.get())
            if timestamp < 0:
                raise ValueError("Time must be non-negative")
        except ValueError:
            messagebox.showerror("Invalid Time", "Please enter a valid, non-negative number for time.")
            return
        
        try:
            duration = float(self.action_duration_entry.get())
            if duration <= 0:
                raise ValueError("Duration must be positive")
        except ValueError:
            messagebox.showerror("Invalid Duration", "Please enter a valid, positive number for duration.")
            return
        
        action_type = self.action_type_var.get()
        text = self.action_text_entry.get().strip()
        
        if not text:
            messagebox.showwarning("Empty Text", "Please enter action text.")
            return
        
        new_action = {
            'timestamp': timestamp,
            'type': action_type,
            'text': text,
            'duration': duration
        }
        
        self.actions.append(new_action)
        self._update_action_listbox()
        
        self.action_text_entry.delete(0, tk.END)
        self.action_time_entry.delete(0, tk.END)
        self.action_time_entry.insert(0, str(round(timestamp + duration, 1)))
        
        print(f"Action added: [{action_type}] {text} @ {timestamp}s")
    
    def update_action(self):
        selected_index = self._get_selected_action_index()
        if selected_index is None:
            messagebox.showwarning("No Action Selected", "Please select an action to update.")
            return
        
        try:
            timestamp = float(self.action_time_entry.get())
            if timestamp < 0:
                raise ValueError("Time must be non-negative")
        except ValueError:
            messagebox.showerror("Invalid Time", "Please enter a valid, non-negative number for time.")
            return
        
        try:
            duration = float(self.action_duration_entry.get())
            if duration <= 0:
                raise ValueError("Duration must be positive")
        except ValueError:
            messagebox.showerror("Invalid Duration", "Please enter a valid, positive number for duration.")
            return
        
        action_type = self.action_type_var.get()
        text = self.action_text_entry.get().strip()
        
        if not text:
            messagebox.showwarning("Empty Text", "Please enter action text.")
            return
        
        self.actions[selected_index] = {
            'timestamp': timestamp,
            'type': action_type,
            'text': text,
            'duration': duration
        }
        
        self._update_action_listbox()
        self.action_listbox.selection_set(selected_index)
        
        print(f"Action updated: [{action_type}] {text} @ {timestamp}s")
    
    def delete_action(self):
        selected_index = self._get_selected_action_index()
        if selected_index is None:
            messagebox.showwarning("No Action Selected", "Please select an action to delete.")
            return
        
        action = self.actions[selected_index]
        if messagebox.askyesno("Confirm Delete", 
                               f"Delete action?\n[{action['type']}] {action['text']}"):
            del self.actions[selected_index]
            self._update_action_listbox()
            
            self.action_text_entry.delete(0, tk.END)
            
            print(f"Action deleted: [{action['type']}] {action['text']}")

    def reset_all(self):
        if not messagebox.askyesno("Confirm Reset", 
                                   "Are you sure you want to reset everything?\n"
                                   "All unsaved scenes will be lost."):
            return
        
        self.scenes = []
        self.actions = []
        self.default_jersey_number = [1, 1]
        self._update_scene_listbox()
        self._update_action_listbox()
        
        self.action_time_entry.delete(0, tk.END)
        self.action_time_entry.insert(0, "0.0")
        self.action_duration_entry.delete(0, tk.END)
        self.action_duration_entry.insert(0, "3.0")
        self.action_text_entry.delete(0, tk.END)
        self.action_type_var.set("pass")
        
        self.load_scene(None) 
        
        print("--- Scenario Reset ---")

    def save_scenario(self):
        if not self.scenes:
            messagebox.showwarning("Nothing to Save", "There are no scenes to save.")
            return
            
        filepath = filedialog.asksaveasfilename(
            defaultextension=".tcb",
            filetypes=[("Tactical Board Scenario", "*.tcb"), ("All Files", "*.*")],
            title="Save Scenario As..."
        )
        
        if not filepath:
            return

        try:
            scenario_data = {
                'version': 2,
                'scenes': self.scenes,
                'actions': self.actions
            }
            with open(filepath, 'wb') as f:
                pickle.dump(scenario_data, f)
            
            action_info = f" ({len(self.actions)} action(s))" if self.actions else ""
            messagebox.showinfo("Save Successful", f"Scenario saved to:\n{filepath}{action_info}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save scenario.\nError: {e}")

    def load_scenario(self):
        if self.scenes:
            if not messagebox.askyesno("Confirm Load",
                                       "Loading a new scenario will overwrite your current unsaved work.\n"
                                       "Are you sure you want to continue?"):
                return

        filepath = filedialog.askopenfilename(
            filetypes=[("Tactical Board Scenario", "*.tcb"), ("All Files", "*.*")],
            title="Load Scenario"
        )
        
        if not filepath:
            return

        try:
            with open(filepath, 'rb') as f:
                loaded_data = pickle.load(f)
            
            if isinstance(loaded_data, dict) and loaded_data.get('version', 1) >= 2:
                loaded_scenes = loaded_data.get('scenes', [])
                loaded_actions = loaded_data.get('actions', [])
                print(f"New format detected (v{loaded_data.get('version')}): {len(loaded_scenes)} scenes, {len(loaded_actions)} actions")
            elif isinstance(loaded_data, list):
                loaded_scenes = loaded_data
                loaded_actions = []
                print(f"Old format detected: {len(loaded_scenes)} scenes")
            else:
                raise TypeError("File is not a valid scenario.")
            
            if loaded_scenes and not isinstance(loaded_scenes[0], dict):
                raise TypeError("File is not a valid scenario (scenes must be dicts).")

            self.scenes = loaded_scenes
            self.actions = loaded_actions
            self._update_scene_listbox()
            self._update_action_listbox()
            self._recalculate_default_jerseys()
            
            self.load_scene(0)
            self.scene_listbox.selection_set(0)
            
            action_info = f"\n{len(loaded_actions)} action(s) loaded." if loaded_actions else ""
            messagebox.showinfo("Load Successful", f"Successfully loaded scenario:\n{filepath}{action_info}")

        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load scenario file.\nError: {e}")

    def run_simulation(self):
        if len(self.scenes) < 2:
            messagebox.showwarning("Not Enough Scenes", 
                                   "You must have at least two scenes to run a simulation.")
            return
            
        self.scenes.sort(key=lambda s: s['timestamp'])
        
        SAMPLE_INTERVAL = 0.5
        
        player_data_map = {}
        ball_data_list = []
        last_sampled_time = -SAMPLE_INTERVAL
        
        for scene in self.scenes:
            ts = scene['timestamp']
            
            if ts - last_sampled_time < SAMPLE_INTERVAL:
                continue
            last_sampled_time = ts
            
            state = scene['state']
            
            if state.get('ball_pos'):
                x_canvas, y_canvas = state['ball_pos']
                sim_x = (x_canvas - self.PADDING) * self.SIM_SCALE
                sim_y = (y_canvas - self.PADDING) * self.SIM_SCALE
                ball_data_list.append((ts, int(sim_x), int(sim_y)))
            
            for p_data in state['players']:
                player_id = p_data['id']
                if player_id not in player_data_map:
                    try:
                        jersey_val = int(p_data.get('jersey', 0)) if p_data.get('jersey') is not None else 0
                    except (ValueError, TypeError):
                        jersey_val = 0
                    try:
                        team_val = int(p_data.get('team', 0)) if p_data.get('team') is not None else 0
                    except (ValueError, TypeError):
                        team_val = 0
                    try:
                        class_val = int(p_data.get('class_id', 2)) if p_data.get('class_id') is not None else 2
                    except (ValueError, TypeError):
                        class_val = 2
                    try:
                        tracker_val = int(p_data.get('tracker_id')) if p_data.get('tracker_id') is not None else None
                    except (ValueError, TypeError):
                        tracker_val = None
                    
                    player_data_map[player_id] = {
                        'team': team_val,
                        'jersey': jersey_val,
                        'class_id': class_val,
                        'tracker_id': tracker_val,
                        'positions': []
                    }
                x_canvas, y_canvas = p_data['pos']
                sim_x = (x_canvas - self.PADDING) * self.SIM_SCALE
                sim_y = (y_canvas - self.PADDING) * self.SIM_SCALE
                player_data_map[player_id]['positions'].append((ts, int(sim_x), int(sim_y)))
        
        print(f"Sampling: {SAMPLE_INTERVAL}s intervals, total {int(last_sampled_time / SAMPLE_INTERVAL) + 1} frames")

        sim_players = []
        print(f"Total unique players: {len(player_data_map)}")
        for p_id, p_info in player_data_map.items():
            if not p_info['positions']: continue
            tracker_id = p_info.get('tracker_id')
            display_info = f"#{p_info['jersey']}" if p_info['jersey'] > 0 else f"ID {tracker_id}" if tracker_id else "?"
            print(f"  - {display_info} Team: {p_info['team']} Class: {p_info.get('class_id', 2)} Positions: {len(p_info['positions'])}")
            try:
                p = Player(team_id=p_info['team'], 
                           jersey_number=p_info['jersey'], 
                           positions=p_info['positions'],
                           class_id=p_info.get('class_id', 2),
                           player_id=p_id,
                           tracker_id=tracker_id)
                sim_players.append(p)
            except ValueError as e:
                print(f"Skipping player {display_info}: {e}")
                
        if not ball_data_list:
            messagebox.showerror("No Ball Data", "The ball was not found in the scenes.")
            return
            
        try:
            unique_ball_timestamps = {ts for ts, x, y in ball_data_list}
            if not unique_ball_timestamps:
                 raise ValueError("Ball has no positions.")
            sim_ball = Ball(positions=ball_data_list)
        except ValueError as e:
            messagebox.showerror("Ball Error", f"Could not create ball: {e}")
            return
            
        if not sim_players:
            messagebox.showerror("No Player Data", "No players were found in the scenes.")
            return
        
        sim_actions = []
        for action_data in self.actions:
            try:
                sim_action = Action(
                    timestamp=action_data['timestamp'],
                    action_type=action_data.get('type', 'default'),
                    text=action_data.get('text', ''),
                    duration=action_data.get('duration', 3.0)
                )
                sim_actions.append(sim_action)
            except Exception as e:
                print(f"Skipping action: {e}")
            
        self.root.withdraw()
        try:
            print("--- Starting Simulation ---")
            for player in sim_players:
                print(f"Player {player.jersey_number} (Team {player.team_id}):")
                print(f"  {len(player.positions)} positions")
                for pos in player.positions:
                    print(f"    Time: {pos.timestamp}s -> ({pos.x}, {pos.y})")
            print("Ball Positions:")
            for pos in sim_ball.positions:
                print(f"    Time: {pos.timestamp}s -> ({pos.x}, {pos.y})")
            
            if sim_actions:
                print(f"Actions: {len(sim_actions)}")
                for action in sim_actions:
                    print(f"    [{action.action_type}] {action.text} @ {action.timestamp}s")
                
            sim = Simulation(ball=sim_ball, players=sim_players, actions=sim_actions)
            sim.run()
        except Exception as e:
            messagebox.showerror("Simulation Error", f"An error occurred during simulation: {e}")
            print(f"Simulation Error: {e}")
        
        self.root.deiconify()