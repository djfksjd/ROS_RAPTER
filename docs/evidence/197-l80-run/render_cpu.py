"""MuJoCo 장면을 NumPy/Matplotlib로 그리는 CPU 대체 렌더.
OpenGL이 없는 환경에서만 사용한다. 실제 장면의 box/sphere/plane만 투영하며
물리 계산·학습은 하지 않는다. 재질·그림자는 MuJoCo OpenGL과 다르다.
"""
from itertools import product

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure


def faces(geom):
    size = np.asarray(geom['size'])
    kind = geom['type']
    if kind == 6:  # box
        v = np.array(list(product((-1, 1), repeat=3))) * size
        ids = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
               (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
        polygons = [v[list(i)] for i in ids]
    elif kind == 2:  # sphere
        phi = np.linspace(0, np.pi, 9)
        theta = np.linspace(0, 2*np.pi, 17)
        v = np.array([[[np.sin(p)*np.cos(t), np.sin(p)*np.sin(t), np.cos(p)]
                       for t in theta] for p in phi]) * size[0]
        polygons = [v[[i, i+1, i+1, i], [j, j, j+1, j+1]]
                    for i in range(8) for j in range(16)]
    elif kind == 0:  # plane, a finite visible patch of the infinite floor
        polygons = [np.array([[-200, -200, 0], [200, -200, 0],
                              [200, 200, 0], [-200, 200, 0]], dtype=float)]
    else:
        raise ValueError(f'Unsupported MuJoCo scene geometry {kind}')
    rotation = np.asarray(geom['mat'])
    return [p @ rotation.T + geom['pos'] for p in polygons]


def clip_near(vertices, near=.01):
    """Sutherland-Hodgman clipping in camera coordinates, positive depth."""
    out = []
    for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
        ina, inb = a[2] >= near, b[2] >= near
        if ina:
            out.append(a)
        if ina != inb:
            out.append(a + (b-a)*((near-a[2])/(b[2]-a[2])))
    return np.asarray(out)


def render(frame):
    width, height = 420, 360
    forward = np.asarray(frame['camera_forward'], dtype=float)
    forward /= np.linalg.norm(forward)
    up = np.asarray(frame['camera_up'], dtype=float)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    basis = np.stack([right, up, forward], axis=1)
    position = np.asarray(frame['camera_pos'])
    focal = height / (2*np.tan(np.radians(frame['fovy'])/2))
    light = np.array([-.3, -.5, 1.])
    light /= np.linalg.norm(light)
    polygons = []
    for geom in frame['geoms']:
        for world in faces(geom):
            camera = (world-position) @ basis
            camera = clip_near(camera)
            if len(camera) < 3:
                continue
            projected = camera[:, :2] / camera[:, 2, None] * focal
            projected = projected * [1, -1] + [width/2, height/2]
            normal = np.cross(world[1]-world[0], world[2]-world[0])
            norm = np.linalg.norm(normal)
            # Faces are double-sided; shade only to make real geometry legible.
            shade = .6 + .4*abs(np.dot(normal/norm, light)) if norm else .6
            color = np.clip(np.asarray(geom['rgba'][:3])*shade, 0, 1)
            depth = float(camera[:, 2].mean()) if geom['type'] != 0 else float('inf')
            polygons.append((depth, projected, color))
    polygons.sort(key=lambda p: p[0], reverse=True)
    fig = Figure(figsize=(width/100, height/100), dpi=100, facecolor='#333333')
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, width), ylim=(height, 0))
    ax.axis('off')
    ax.add_collection(PolyCollection([p[1] for p in polygons],
                                    facecolors=[p[2] for p in polygons],
                                    edgecolors='none', antialiaseds=True))
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
