import tkinter as tk
from src.graph_view import TFPSGUI

def main():
    root = tk.Tk()
    app = TFPSGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()