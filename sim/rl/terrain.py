"""Height-field terrains for Raptor RL (MuJoCo). Heights in metres, grid 0.05 m.

Kinds: flat, rough (random bumps), slope (fore-aft ramp), side_slope (cross slope everywhere, walking
along it), stairs (up), platform (drop-off edge), boxes (flat runs of 2-4 m joined by random ledges up or drops down,
for jumping), rough_hfield (the old height-field rough, kept to reproduce old runs).
rough is built from 0.2 m boxes in a strip |y| < ROUGH_HALF_WIDTH: with the height field the foot boxes sank 2-5 cm
and a policy fell even on a 0.5 cm field (evidence 173).
`level` in [0, 1] scales difficulty for a curriculum. The robot spawns at the grid centre (x = 0),
walking along +x.
"""
import mujoco
import numpy as np

SIZE = 8.  # m, square
RES = .05  # m per cell
KINDS = ('flat', 'rough', 'slope', 'side_slope', 'stairs', 'platform', 'boxes', 'rough_hfield')
BOX_UP, BOX_DOWN = .45, .8  # m, largest ledge up / drop down at level 1
ROUGH_HALF_WIDTH = 2.  # m


def heights(kind, level, rng):
    n = int(SIZE/RES)+1
    x = np.linspace(-SIZE/2, SIZE/2, n)
    X = np.tile(x, (n, 1))  # columns = x
    h = np.zeros((n, n))
    ahead = np.clip(X-.5, 0, None)  # keep the spawn pad flat
    if kind in ('rough', 'rough_hfield'):
        amp = .005+.045*level
        noise = rng.uniform(-amp, amp, (n//4+1, n//4+1))
        h = np.kron(noise, np.ones((4, 4)))[:n, :n]
        h[:, np.abs(x) < .5] = 0.
    elif kind == 'slope':
        deg = 3.+22.*level
        h = np.tan(np.radians(deg))*ahead*rng.choice([1., -1.])
    elif kind == 'side_slope':
        deg = 3.+17.*level
        Y = np.tile(x[:, None], (1, n))  # rows = y
        h = np.tan(np.radians(deg))*Y*rng.choice([1., -1.])
    elif kind == 'stairs':
        rise, run = .02+.10*level, .35
        h = np.floor(ahead/run)*rise
    elif kind == 'platform':  # spawn on a raised block, walk off the edge
        drop = .05+.35*level
        h = np.where(X < 1., drop, 0.)
    elif kind == 'boxes':  # from x = 1.5 m: a ledge up or a drop down (each 50 %) of 3-100 % of the level's size every 2-4 m
        row, x0, z = np.zeros(n), 1.5, 0.
        while x0 < SIZE/2:
            up = rng.random() < .5
            z += (1. if up else -1.)*rng.uniform(.03, 1.)*level*(BOX_UP if up else BOX_DOWN)
            row[x >= x0] = z
            x0 += rng.uniform(2., 4.)
        h = np.tile(row, (n, 1))
    return h - h[:, n//2].mean() if kind == 'slope' else h


def add_terrain(spec, kind, level, rng):
    """Add the terrain to the model spec; returns the height grid used for queries.

    flat keeps the plane. rough uses a height field; slope, stairs and platform use boxes
    (exact geometry and much cheaper contacts than a height field).
    """
    h = heights(kind, level, rng)
    world, n = spec.worldbody, h.shape[0]
    if kind == 'rough':
        x = np.linspace(-SIZE/2, SIZE/2, n)
        lo = min(h.min(), 0.)-.02
        keep = np.zeros_like(h, bool)
        for i in range(0, n-3, 4):  # rows = y
            for j in range(0, n-3, 4):  # columns = x; patch j covers cells j..j+3
                xc, yc = x[j]+1.5*RES, x[i]+1.5*RES
                if x[j] < .5 or abs(yc) > ROUGH_HALF_WIDTH:
                    continue
                keep[i:i+4, j:j+4] = True
                world.add_geom(name=f'patch{i}_{j}', type=mujoco.mjtGeom.mjGEOM_BOX, size=[2*RES, 2*RES, (h[i, j]-lo)/2],
                               pos=[xc, yc, (h[i, j]+lo)/2])
        x0 = x[np.flatnonzero(keep.any(0))[0]]-RES/2  # first patch edge: the spawn pad reaches it with no gap
        h[~keep] = lo
        h[np.tile(x, (n, 1)) < x0] = 0.
        world.add_geom(name='pad', type=mujoco.mjtGeom.mjGEOM_BOX, size=[(SIZE/2+x0)/2, SIZE/2, -lo/2], pos=[(-SIZE/2+x0)/2, 0, lo/2])
        spec.geom('floor').pos = [0, 0, lo]
    elif kind == 'rough_hfield':
        lo, hi = h.min(), h.max()
        span = max(hi-lo, 1e-3)
        spec.add_hfield(name='terrain', nrow=n, ncol=n, size=[SIZE/2, SIZE/2, span, .5],
                        userdata=((h-lo)/span).ravel().tolist())
        floor = spec.geom('floor')
        floor.type, floor.hfieldname, floor.pos = mujoco.mjtGeom.mjGEOM_HFIELD, 'terrain', [0, 0, lo]
    elif kind == 'slope':
        x = np.linspace(-SIZE/2, SIZE/2, n)
        k = int(np.argmin(np.abs(x-.5)))  # ramp foot (ahead = 0)
        grade = (h[0, -1]-h[0, k])/(x[-1]-x[k])
        ang = np.arctan(grade)
        L = (SIZE/2-.5)/np.cos(ang)
        cx, cz = .5+(SIZE/2-.5)/2, h[0, k]+grade*(SIZE/2-.5)/2
        world.add_geom(name='ramp', type=mujoco.mjtGeom.mjGEOM_BOX, size=[L/2, SIZE/2, .05],
                       pos=[cx+.05*np.sin(ang), 0, cz-.05*np.cos(ang)],
                       quat=[np.cos(-ang/2), 0, np.sin(-ang/2), 0])
        spec.geom('floor').pos = [0, 0, h[0, k]]
        if grade < 0:  # downhill: the ramp sinks below the spawn plane, lower the plane past the ramp
            spec.geom('floor').pos = [0, 0, h.min()]
            world.add_geom(name='pad', type=mujoco.mjtGeom.mjGEOM_BOX, size=[(SIZE/2+.5)/2, SIZE/2, .05],
                           pos=[(-SIZE/2+.5)/2, 0, h[0, k]-.05])
    elif kind == 'side_slope':  # one plane tilted about x through the origin
        grade = (h[-1, 0]-h[0, 0])/SIZE
        ang = np.arctan(grade)
        world.add_geom(name='cross', type=mujoco.mjtGeom.mjGEOM_BOX, size=[SIZE/2, SIZE/2/np.cos(ang), .05],
                       pos=[0, .05*np.sin(ang), -.05*np.cos(ang)], quat=[np.cos(ang/2), np.sin(ang/2), 0, 0])
        spec.geom('floor').pos = [0, 0, h.min()-.05]
    elif kind in ('stairs', 'platform', 'boxes'):  # blocks standing on the plane, lowered below the lowest top
        x = np.linspace(-SIZE/2, SIZE/2, n)
        row = h[0]
        lo = min(row.min(), 0.)
        edges = np.flatnonzero(np.diff(row) != 0)+1
        bounds = [0, *edges, n]
        for i in range(len(bounds)-1):
            top = row[bounds[i]]
            if top <= lo:
                continue
            x0, x1 = x[bounds[i]]-RES/2, x[bounds[i+1]-1]+RES/2
            world.add_geom(name=f'block{i}', type=mujoco.mjtGeom.mjGEOM_BOX,
                           size=[(x1-x0)/2, SIZE/2, (top-lo)/2], pos=[(x0+x1)/2, 0, (top+lo)/2])
        spec.geom('floor').pos = [0, 0, lo]
    return h


def height_at(h, x, y):
    n = h.shape[0]
    i = int(round((y+SIZE/2)/RES)); j = int(round((x+SIZE/2)/RES))
    return float(h[min(max(i, 0), n-1), min(max(j, 0), n-1)])
