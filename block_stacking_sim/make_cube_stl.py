import numpy as np, os
SIZE_MM = 30.0
h = SIZE_MM / 2
V = np.array([[x, y, z] for x in (-h, h) for y in (-h, h) for z in (-h, h)], dtype=np.float32)
quads = [(0,1,3,2),(4,6,7,5),(0,4,5,1),(2,3,7,6),(0,2,6,4),(1,5,7,3)]
tris = []
for a, b, c, d in quads:
    tris += [(a, b, c), (a, c, d)]
out = bytearray(80) + np.uint32(len(tris)).tobytes()
for a, b, c in tris:
    p, q, r = V[a], V[b], V[c]
    n = np.cross(q - p, r - p)
    if np.dot(n, (p + q + r) / 3) < 0:
        q, r = r, q
        n = -n
    n = n / np.linalg.norm(n)
    out += n.astype("<f4").tobytes() + p.astype("<f4").tobytes() + q.astype("<f4").tobytes() + r.astype("<f4").tobytes() + b"\x00\x00"
open(os.path.join("meshes", "block1.stl"), "wb").write(out)
print("wrote meshes/block1.stl,", SIZE_MM, "mm cube")
