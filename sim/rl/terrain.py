"""Height-field terrains for Raptor RL (MuJoCo). Heights in metres, grid 0.05 m.

Kinds: flat, rough (random bumps), slope (fore-aft ramp), stairs (up), platform (drop-off edge).
`level` in [0, 1] scales difficulty for a curriculum. The robot spawns at the grid centre (x = 0),
walking along +x.
"""
import mujoco
import numpy as np

SIZE = 8.  # m, square
RES = .05  # m per cell
KINDS = ('flat', 'rough', 'slope', 'stairs', 'platform')


def heights(kind, level, rng):
    n = int(SIZE/RES)+1
    x = np.linspace(-SIZE/2, SIZE/2, n)
    X = np.tile(x, (n, 1))  # columns = x
    h = np.zeros((n, n))
    ahead = np.clip(X-.5, 0, None)  # keep the spawn pad flat
    if kind == 'rough':
        amp = .005+.045*level
        noise = rng.uniform(-amp, amp, (n//4+1, n//4+1))
        h = np.kron(noise, np.ones((4, 4)))[:n, :n]
        h[:, np.abs(x) < .5] = 0.
    elif kind == 'slope':
        deg = 3.+22.*level
        h = np.tan(np.radians(deg))*ahead*rng.choice([1., -1.])
    elif kind == 'stairs':
        rise, run = .02+.10*level, .35
        h = np.floor(ahead/run)*rise
    elif kind == 'platform':  # spawn on a raised block, walk off the edge
        drop = .05+.35*level
        h = np.where(X < 1., drop, 0.)
    return h - h[:, n//2].mean() if kind == 'slope' else h


def add_terrain(spec, kind, level, rng):
    """Add the terrain to the model spec; returns the height grid used for queries.

    flat keeps the plane. rough uses a height field; slope, stairs and platform use boxes
    (exact geometry and much cheaper contacts than a height field).
    """
    h = heights(kind, level, rng)
    world, n = spec.worldbody, h.shape[0]
    if kind == 'rough':
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
    elif kind in ('stairs', 'platform'):
        x = np.linspace(-SIZE/2, SIZE/2, n)
        row = h[0]
        edges = np.flatnonzero(np.diff(row) != 0)+1
        bounds = [0, *edges, n]
        for i in range(len(bounds)-1):
            top = row[bounds[i]]
            if top <= 0:
                continue
            x0, x1 = x[bounds[i]]-RES/2, x[bounds[i+1]-1]+RES/2
            world.add_geom(name=f'block{i}', type=mujoco.mjtGeom.mjGEOM_BOX,
                           size=[(x1-x0)/2, SIZE/2, top/2], pos=[(x0+x1)/2, 0, top/2])
    return h


def height_at(h, x, y):
    n = h.shape[0]
    i = int(round((y+SIZE/2)/RES)); j = int(round((x+SIZE/2)/RES))
    return float(h[min(max(i, 0), n-1), min(max(j, 0), n-1)])
