import tkinter as tk

def main():
    root = tk.Tk()
    label = tk.Label(root, text="Hello, Tkinter!")
    label.pack()
    root.mainloop()

if __name__ == "__main__":
    main()