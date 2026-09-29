#!/usr/bin/env python3
"""Orthographic design drawing (side X-Z, front Y-Z) of the URDF collision boxes at zero pose.

Source evidence for the Morphloom job: dimensions are the Xacro's own values, 1 px = 1 mm.
"""
import itertools
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src/raptor_control/scripts'))
from lateral_feasibility import Model  # noqa: E402
from support_model import origin  # noqa: E402

urdf, out = sys.argv[1], sys.argv[2]
text = Path(urdf).read_text()
frames, root = Model(text).fk({}), ET.fromstring(text)
boxes = []
for link in root.findall('link'):
    for collision in link.findall('collision'):
        box, sphere = collision.find('geometry/box'), collision.find('geometry/sphere')
        if box is not None:
            half = np.array(list(map(float, box.get('size').split())))/2
        elif sphere is not None:  # R-02 MTP pad: drawn as its bounding cube
            half = np.full(3, float(sphere.get('radius')))
        else:
            continue
        t = frames[link.get('name')]@origin(collision.find('origin'))
        boxes.append(np.array([(t@np.r_[half*np.array(s), 1])[:3] for s in itertools.product([-1, 1], repeat=3)]))
pts = np.vstack(boxes)*1000
lo, hi = pts.min(0)-60, pts.max(0)+60
side_w, front_w, height = int(hi[0]-lo[0]), int(hi[1]-lo[1]), int(hi[2]-lo[2])
image = Image.new('RGB', (side_w+front_w+40, height+60), 'white')
draw = ImageDraw.Draw(image)


def hull(p):
    p = sorted(set(map(tuple, np.round(p, 1))))
    def half(seq):
        h = []
        for q in seq:
            while len(h) > 1 and (h[-1][0]-h[-2][0])*(q[1]-h[-2][1])-(h[-1][1]-h[-2][1])*(q[0]-h[-2][0]) <= 0:
                h.pop()
            h.append(q)
        return h
    return half(p)[:-1]+half(p[::-1])[:-1]


for box in boxes:
    b = box*1000
    side = [(x-lo[0], hi[2]-z) for x, z in b[:, [0, 2]]]
    front = [(side_w+40+(hi[1]-y), hi[2]-z) for y, z in b[:, [1, 2]]]
    for poly in (side, front):
        draw.polygon(hull(np.array(poly)), outline='black')
draw.line([(20, height+40), (520, height+40)], fill='black', width=3)
draw.text((20, height+45), '500 mm   side (X fwd right, Z up) | front (Y left on right)', fill='black')
image.save(out)
print(out, image.size)
