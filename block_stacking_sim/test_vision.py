"""Compatibility camera check: parks the arm, checks detections, saves results.

Run directly with Python. This is a rendered diagnostic, not a pytest collection
hook: importing it must not allocate a simulator or open a renderer.
"""
from run_simulation import main

if __name__=="__main__":main(["--check"])
