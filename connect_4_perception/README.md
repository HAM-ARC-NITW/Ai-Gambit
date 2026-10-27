# Connect 4 simulation

Double-click **run_simulation.cmd**, or run this from the repository root in
PowerShell:

```powershell
.\connect_4\run_simulation.cmd
```

To run the Python file directly, use the project's interpreter:

```powershell
.\.venv\Scripts\python.exe .\connect_4\run_simulation.py

# Quick physics check without opening a window.
.\.venv\Scripts\python.exe .\connect_4\run_simulation.py --no-viewer --seconds 2
```

The launcher shows the SO-101 arm, Connect 4 board and sample red token. Mouse
dragging rotates/pans the view; scrolling zooms. Close the viewer or press Ctrl+C
in its terminal to stop. The default viewer refresh is 30 FPS; `--fps 60` changes
the refresh rate, and `--seconds 5` closes after five simulated seconds.

The scene reuses `../simulation/so101_new_calib.xml` and its meshes. The Connect 4
STLs stay in this folder. The scene uses radians, including its 90-degree board
rotation (`pi/2`). Python resolves the XML beside itself so the launch also works
from another working directory.

This is the initial environment viewer. The token is currently a fixed sample
body in the supplied XML. Human-vs-arm gameplay, dropping tokens, perception and
ROS control are future steps; this launcher runs MuJoCo directly.
