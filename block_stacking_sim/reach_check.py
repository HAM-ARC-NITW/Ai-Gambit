import numpy as np
from env import RowEnv
from engine import StackEngine, PLACE_TILTS, PLACE_CLEARANCE

env = RowEnv(render=False)
eng = StackEngine(env)
bh = eng.bh
dzh = max(0.008, max(bh / 2, eng.protr + 0.003)) - bh / 2
ROW_X = 0.19


def reachable(x, y):
    for lv in (0, 3):                                    # lowest and highest tower level
        z = eng.sq_top + bh * (lv + 0.5) + 0.002 + dzh
        for zz in (z, z + PLACE_CLEARANCE):
            if not eng._solve(np.array([x,y,zz]),0.,PLACE_TILTS[-1])[3]:
                return False
    return True


xs = np.arange(0.10, 0.34, 0.02)
ys = np.arange(-0.20, 0.201, 0.04)
print("R = the arm can reach all tower heights there, . = it cannot.   (columns: x in m, rows: y in m)")
print("        " + " ".join(f"{x:.2f}" for x in xs))
good = []
for y in ys[::-1]:
    row = ""
    for x in xs:
        r = reachable(x, y)
        row += "  R  " if r else "  .  "
        if r:
            good.append((x, y))
    print(f"y={y:+.2f} {row}")

blocks_zone = [g for g in good if not (abs(g[0] - ROW_X) < 0.06 and abs(g[1]) < 0.14)]
blocks_zone.sort(key=lambda g: np.hypot(g[0] - 0.19, g[1] - 0.17))
print("\nbest spots outside the block row:")
for x, y in blocks_zone[:6]:
    print(f"   SQUARE_X = {x:.2f}   SQUARE_Y = {y:+.2f}")
if not blocks_zone:
    print("   none found. Send me this whole printout.")
