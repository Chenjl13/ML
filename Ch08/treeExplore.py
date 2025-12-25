'''
Tree-Based Regression GUI Visualization Tool (Python3 Compatible)
'''
import numpy as np
import tkinter as tk
from tkinter import ttk
import regTrees  # Import core tree regression module (must be in same directory)

import matplotlib
matplotlib.use('TkAgg')  # Configure matplotlib backend for Tkinter compatibility
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

def reDraw(tolS, tolN):
    """
    Redraw the plot with new parameters (regression tree/model tree fitting).
    Fixes: Convert matrix to 1D array for matplotlib scatter plot compatibility.
    Args:
        tolS (float): Minimum error reduction for splitting
        tolN (int): Minimum sample count for splitting
    """
    # Clear previous plot
    reDraw.f.clf()
    reDraw.ax = reDraw.f.add_subplot(111)
    
    if chkBtnVar.get():
        # Use Model Tree (better fitting for non-linear data)
        if tolN < 2:
            tolN = 2  # Ensure minimum sample count for model tree
        myTree = regTrees.createTree(
            reDraw.rawDat,
            leafType=regTrees.modelLeaf,
            errType=regTrees.modelErr,
            ops=(tolS, tolN)
        )
        yHat = regTrees.createForeCast(
            myTree,
            reDraw.testDat,
            modelEval=regTrees.modelTreeEval
        )
    else:
        # Use Regression Tree (simpler, mean-based leaves)
        myTree = regTrees.createTree(reDraw.rawDat, ops=(tolS, tolN))
        yHat = regTrees.createForeCast(myTree, reDraw.testDat)
    
    # Fix: Convert matrix to 1D array for matplotlib scatter plot
    x_data = reDraw.rawDat[:, 0].A1  # 1D array of x-features
    y_data = reDraw.rawDat[:, 1].A1  # 1D array of target values
    
    # Plot original data (blue scatter) and fitted curve (red line)
    reDraw.ax.scatter(x_data, y_data, s=5, color='blue', label='Original Data')
    reDraw.ax.plot(reDraw.testDat, yHat, linewidth=2.0, color='red', label='Fitted Curve')
    reDraw.ax.legend(loc='best')
    reDraw.ax.set_xlabel('X Feature')
    reDraw.ax.set_ylabel('Target Value')
    reDraw.ax.set_title('Tree Regression Fitting Result')
    
    # Update canvas (replace plt.show() for Tkinter compatibility)
    reDraw.canvas.draw()

def getInputs():
    """
    Get and validate input parameters from entry boxes.
    Returns default values if input is invalid.
    Returns:
        tuple: (tolN, tolS) - Validated parameters
    """
    # Validate tolN (must be integer)
    try:
        tolN = int(tolN_entry.get())
    except ValueError:
        tolN = 10
        print("Warning: tolN must be an integer. Automatically set to 10.")
        tolN_entry.delete(0, tk.END)
        tolN_entry.insert(0, '10')
    
    # Validate tolS (must be float)
    try:
        tolS = float(tolS_entry.get())
    except ValueError:
        tolS = 1.0
        print("Warning: tolS must be a float. Automatically set to 1.0.")
        tolS_entry.delete(0, tk.END)
        tolS_entry.insert(0, '1.0')
    
    return tolN, tolS

def drawNewTree():
    """
    Triggered by "ReDraw" button: Get parameters and redraw the plot.
    """
    tolN, tolS = getInputs()
    reDraw(tolS, tolN)

# Initialize main GUI window
root = tk.Tk()
root.title("Tree Regression Fitting Tool (Python3)")
root.geometry("600x500")  # Set initial window size

# Initialize matplotlib figure and canvas
reDraw.f = Figure(figsize=(5, 3.5), dpi=100)
reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master=root)
reDraw.canvas.draw()  # Render canvas (replace show() for Python3)
reDraw.canvas.get_tk_widget().grid(row=0, columnspan=3, padx=10, pady=10, sticky='nsew')

# Create and arrange GUI widgets (labels, entries, buttons)
# tolN Label and Entry
tolN_label = ttk.Label(root, text="tolN (Min Sample Count):")
tolN_label.grid(row=1, column=0, padx=10, pady=5, sticky='w')
tolN_entry = ttk.Entry(root, width=10)
tolN_entry.grid(row=1, column=1, padx=5, pady=5, sticky='w')
tolN_entry.insert(0, '10')  # Default value

# tolS Label and Entry
tolS_label = ttk.Label(root, text="tolS (Min Error Reduction):")
tolS_label.grid(row=2, column=0, padx=10, pady=5, sticky='w')
tolS_entry = ttk.Entry(root, width=10)
tolS_entry.grid(row=2, column=1, padx=5, pady=5, sticky='w')
tolS_entry.insert(0, '1.0')  # Default value

# ReDraw Button
redraw_btn = ttk.Button(root, text="ReDraw Fitting", command=drawNewTree)
redraw_btn.grid(row=1, column=2, rowspan=2, padx=10, pady=5, sticky='nsew')

# Model Tree Checkbutton
chkBtnVar = tk.IntVar()
model_tree_chk = ttk.Checkbutton(root, text="Use Model Tree (Better Non-linear Fit)", variable=chkBtnVar)
model_tree_chk.grid(row=3, column=0, columnspan=2, padx=10, pady=5, sticky='w')

# Load dataset (sine.txt must be in the same directory)
reDraw.rawDat = np.mat(regTrees.loadDataSet('sine.txt'))

# Generate test data (evenly spaced for smooth fitting curve)
min_x = min(reDraw.rawDat[:, 0].A.flatten())
max_x = max(reDraw.rawDat[:, 0].A.flatten())
reDraw.testDat = np.arange(min_x, max_x, 0.01)

# Initial fitting with default parameters
reDraw(1.0, 10)

# Configure grid resizing (make plot responsive)
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)
root.grid_columnconfigure(2, weight=1)

# Start GUI main loop
root.mainloop()