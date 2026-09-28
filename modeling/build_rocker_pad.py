"""Convex rocker foot pad (collision mesh) for sole_shape:=rocker.

Same footprint as the flat pad (0.15 x 0.10 m, 0.035 m thick at the centre). The top is flat; the bottom
is a downward paraboloid z = -t/2 + x^2/(2 Rx) + y^2/(2 Ry), so contact rolls heel-to-toe (Rx) and side
to side (Ry) instead of landing on an edge. Convex, so MuJoCo (convex hull) and Gazebo see the same shape.
Writes a binary STL (MuJoCo reads only binary) centred on the pad centre, metres.
Usage: python3 modeling/build_rocker_pad.py [--rx 0.30 --ry 0.20]
"""
import argparse
import struct
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]/'src/raptor_description/meshes/foot/rocker_pad.stl'


def surface(L, W, t, rx, ry, n=12):
    xs = [-L/2+L*i/n for i in range(n+1)]
    ys = [-W/2+W*j/n for j in range(n+1)]
    bottom = [[(x, y, -t/2+x*x/(2*rx)+y*y/(2*ry)) for y in ys] for x in xs]
    top = [[(x, y, t/2) for y in ys] for x in xs]
    return bottom, top


def quads_to_tris(grid, flip):
    tris = []
    for i in range(len(grid)-1):
        for j in range(len(grid[0])-1):
            a, b, c, d = grid[i][j], grid[i+1][j], grid[i+1][j+1], grid[i][j+1]
            tris += [(a, c, b), (a, d, c)] if flip else [(a, b, c), (a, c, d)]
    return tris


def normal(a, b, c):
    u = [b[k]-a[k] for k in range(3)]; v = [c[k]-a[k] for k in range(3)]
    n = [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]]
    s = sum(x*x for x in n)**.5 or 1.
    return [x/s for x in n]


def build(L=.15, W=.10, t=.035, rx=.30, ry=.20):
    bottom, top = surface(L, W, t, rx, ry)
    tris = quads_to_tris(bottom, flip=True)+quads_to_tris(top, flip=False)
    n = len(bottom)-1
    rings = [
        [bottom[i][0] for i in range(n+1)], [top[i][0] for i in range(n+1)],          # y = -W/2
        [bottom[i][n] for i in range(n+1)], [top[i][n] for i in range(n+1)],          # y = +W/2
        [bottom[0][j] for j in range(n+1)], [top[0][j] for j in range(n+1)],          # x = -L/2
        [bottom[n][j] for j in range(n+1)], [top[n][j] for j in range(n+1)],          # x = +L/2
    ]
    for k, flip in ((0, False), (2, True), (4, True), (6, False)):
        lo, hi = rings[k], rings[k+1]
        for i in range(n):
            a, b, c, d = lo[i], lo[i+1], hi[i+1], hi[i]
            tris += [(a, c, b), (a, d, c)] if flip else [(a, b, c), (a, c, d)]
    # orient every facet outward (the solid is convex and centred near the origin)
    out = []
    for a, b, c in tris:
        nrm = normal(a, b, c)
        centre = [(a[k]+b[k]+c[k])/3 for k in range(3)]
        if sum(nrm[k]*centre[k] for k in range(3)) < 0:
            b, c = c, b
            nrm = [-x for x in nrm]
        out.append((nrm, a, b, c))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rx', type=float, default=.30)
    p.add_argument('--ry', type=float, default=.20)
    a = p.parse_args()
    facets = build(rx=a.rx, ry=a.ry)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'wb') as f:
        f.write(b'rocker_pad'.ljust(80, b' ')+struct.pack('<I', len(facets)))
        for nrm, *v in facets:
            f.write(struct.pack('<12fH', *nrm, *v[0], *v[1], *v[2], 0))
    print(f'wrote {OUT} ({len(facets)} facets, Rx {a.rx} Ry {a.ry})')


if __name__ == '__main__':
    main()
