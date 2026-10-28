"""Compatibility entry point for the RGB engine, now using the live launcher."""
import sys
from run_simulation import main

if __name__=="__main__":main(["--stack",*sys.argv[1:]])
