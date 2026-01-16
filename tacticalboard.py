import tkinter as tk
from tacticalboard.GUI.tacticalboardGUI import TacticalBoardGUI

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = TacticalBoardGUI(root)
        root.mainloop()
    except NameError as e:
        if "SoccerPitchConfiguration" in str(e):
            print("Exiting due to missing libraries.")
        else:
            raise e