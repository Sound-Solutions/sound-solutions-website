"""Build the Sound Solutions cube logo in 3D and render it.

Usage: Blender -b --factory-startup -P build.py -- <option> <out.png> [view] [res]
  option: grille | rubik | mono
  view:   straight (default) | hero
"""
import bpy, bmesh, sys, math, numpy as np
from collections import defaultdict
from mathutils import Vector, Matrix

argv = sys.argv[sys.argv.index('--') + 1:]
OPTION = argv[0]
OUT = argv[1]
VIEW = argv[2] if len(argv) > 2 else 'straight'
RES = int(argv[3]) if len(argv) > 3 else 1000

LOGO = '/Users/ksellarsm4lt/Code/sound-solutions-site/logo.png'
S = 0.5503                     # sin(elevation): slope of the iso edges on screen
C = math.sqrt(1 - S * S)
K = math.sqrt(0.5)
CX, CY = 463.0, 456.0          # screen point of the top face's front corner
A = 541.0                      # top face edge, world units (= screen px)
U = 0.01                       # world unit -> blender metres

for o in list(bpy.data.objects): bpy.data.objects.remove(o)

# ---------------------------------------------------------------- source image
img = bpy.data.images.load(LOGO)
W, H = img.size
px = np.array(img.pixels[:], dtype=np.float32).reshape(H, W, 4)[::-1]
alpha = px[..., 3].copy()

def component_mask(seed):
    """Flood the >0.5 region from seed, dilate 3px so the antialiased rim stays."""
    a = alpha > 0.5
    lab = np.zeros(a.shape, bool)
    stack = [seed]; lab[seed] = True
    while stack:
        y, x = stack.pop()
        for yy, xx in ((y+1, x), (y-1, x), (y, x+1), (y, x-1)):
            if 0 <= yy < H and 0 <= xx < W and a[yy, xx] and not lab[yy, xx]:
                lab[yy, xx] = True; stack.append((yy, xx))
    m = lab.copy()
    for _ in range(3):
        m = m | np.roll(m, 1, 0) | np.roll(m, -1, 0) | np.roll(m, 1, 1) | np.roll(m, -1, 1)
    return alpha * m

LEFT = component_mask((616, 450))     # left S: rightmost point (452,616)
RIGHT = component_mask((290, 879))    # right S: (881,290)

def bilinear(img2d, x, y):
    x0 = np.clip(np.floor(x).astype(int), 0, W - 2); y0 = np.clip(np.floor(y).astype(int), 0, H - 2)
    fx = np.clip(x - x0, 0, 1); fy = np.clip(y - y0, 0, 1)
    v = (img2d[y0, x0] * (1 - fx) * (1 - fy) + img2d[y0, x0 + 1] * fx * (1 - fy)
         + img2d[y0 + 1, x0] * (1 - fx) * fy + img2d[y0 + 1, x0 + 1] * fx * fy)
    out = (x < 0) | (y < 0) | (x > W - 1) | (y > H - 1)
    v[out] = 0
    return v

# Face planes. Left face lies on y = YL (coords u=X, v=Z); right on x = XR (u=Y, v=Z).
YL, XR = -15.6, 17.0

def screen_left(u, v):
    sx = K * (u + YL); dn = K * S * (u - YL) - C * v
    return sx + CX, dn + CY

def screen_right(u, v):
    sx = K * (XR + u); dn = K * S * (XR - u) - C * v
    return sx + CX, dn + CY

def sample_face(src, to_screen, urange, vrange, step=0.5):
    us = np.arange(urange[0], urange[1], step); vs = np.arange(vrange[0], vrange[1], step)
    UU, VV = np.meshgrid(us, vs)
    sx, sy = to_screen(UU, VV)
    return us, vs, bilinear(src, sx, sy)

# ---------------------------------------------------------------- marching squares
def contours(grid, us, vs, level=0.5):
    """Closed iso-loops of grid (rows = v, cols = u), in (u, v) coords."""
    g = np.pad(grid, 1, constant_values=0.0)
    step = us[1] - us[0]
    u0, v0 = us[0] - step, vs[0] - step
    b = (g > level).astype(np.uint8)
    case = b[:-1, :-1] | (b[:-1, 1:] << 1) | (b[1:, 1:] << 2) | (b[1:, :-1] << 3)
    js, is_ = np.nonzero((case != 0) & (case != 15))

    def pt(edge, j, i):
        # edges: 0 bottom(j,i)-(j,i+1) 1 right(j,i+1)-(j+1,i+1) 2 top(j+1,i+1)-(j+1,i) 3 left(j+1,i)-(j,i)
        if edge == 0: (ja, ia), (jb, ib) = (j, i), (j, i + 1)
        elif edge == 1: (ja, ia), (jb, ib) = (j, i + 1), (j + 1, i + 1)
        elif edge == 2: (ja, ia), (jb, ib) = (j + 1, i + 1), (j + 1, i)
        else: (ja, ia), (jb, ib) = (j + 1, i), (j, i)
        a_, b_ = g[ja, ia], g[jb, ib]
        t = (level - a_) / (b_ - a_) if b_ != a_ else 0.5
        return (u0 + (ia + t * (ib - ia)) * step, v0 + (ja + t * (jb - ja)) * step)

    def key(edge, j, i):
        if edge == 0: return ('h', j, i)
        if edge == 2: return ('h', j + 1, i)
        if edge == 3: return ('v', j, i)
        return ('v', j, i + 1)

    # segment table: inside is corners set; segments go so inside is on the left
    table = {1: [(3, 0)], 2: [(0, 1)], 3: [(3, 1)], 4: [(1, 2)], 5: [(3, 0), (1, 2)],
             6: [(0, 2)], 7: [(3, 2)], 8: [(2, 3)], 9: [(2, 0)], 10: [(0, 1), (2, 3)],
             11: [(2, 1)], 12: [(1, 3)], 13: [(1, 0)], 14: [(0, 3)]}
    nxt, pos = {}, {}
    for j, i in zip(js, is_):
        for ea, eb in table[int(case[j, i])]:
            ka, kb = key(ea, j, i), key(eb, j, i)
            nxt[ka] = kb
            pos.setdefault(ka, pt(ea, j, i)); pos.setdefault(kb, pt(eb, j, i))
    loops, seen = [], set()
    for start in nxt:
        if start in seen: continue
        loop, k = [], start
        while k not in seen and k in nxt:
            seen.add(k); loop.append(pos[k]); k = nxt[k]
        if len(loop) > 8: loops.append(np.array(loop))
    return loops

def rdp(points, eps):
    if len(points) < 3: return points
    a, b = points[0], points[-1]
    d = b - a; n = np.hypot(*d)
    if n == 0: dist = np.hypot(*(points - a).T)
    else: dist = np.abs(d[0] * (points[:, 1] - a[1]) - d[1] * (points[:, 0] - a[0])) / n
    k = int(np.argmax(dist))
    if dist[k] > eps:
        return np.vstack([rdp(points[:k + 1], eps)[:-1], rdp(points[k:], eps)])
    return np.array([a, b])

def simplify_loop(loop, eps=0.12):
    k = int(np.argmax(loop[:, 0]))
    loop = np.roll(loop, -k, axis=0)
    closed = np.vstack([loop, loop[:1]])
    return rdp(closed, eps)[:-1]

def area(loop):
    x, y = loop[:, 0], loop[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)

# ---------------------------------------------------------------- face shapes
# Left face: u = X in [-A-60, 40], v = Z in roughly [-800, 100]
LU, LV, LG = sample_face(LEFT, screen_left, (-A - 80, 60), (-900, 120))
RU, RV, RG = sample_face(RIGHT, screen_right, (-60, A + 80), (-900, 120))

def bbox_of(grid, us, vs):
    js, is_ = np.nonzero(grid > 0.5)
    return us[is_.min()], us[is_.max()], vs[js.min()], vs[js.max()]

L_BB = bbox_of(LG, LU, LV); R_BB = bbox_of(RG, RU, RV)
print('left face bbox', L_BB); print('right face bbox', R_BB)

def pieces(grid, us, vs, bb, cells=None, groove=7.0):
    """Loops for the face; with cells=(nu,nv) cut it into a grid of tiles."""
    if not cells:
        return [[simplify_loop(l) for l in contours(grid, us, vs)]]
    nu, nv = cells
    out = []
    ue = np.linspace(bb[0], bb[1], nu + 1); ve = np.linspace(bb[2], bb[3], nv + 1)
    UU, VV = np.meshgrid(us, vs)
    for a_ in range(nu):
        for b_ in range(nv):
            m = ((UU > ue[a_] + groove / 2) & (UU < ue[a_ + 1] - groove / 2) &
                 (VV > ve[b_] + groove / 2) & (VV < ve[b_ + 1] - groove / 2))
            ls = [simplify_loop(l) for l in contours(grid * m, us, vs)]
            ls = [l for l in ls if abs(area(l)) > 60]
            if ls: out.append(((a_, b_), ls))
    return out

def top_grid(cells=None, groove=7.0):
    """Top plate: covers the whole top, flush with the S panels' outer faces."""
    lo = 0.0
    us = np.arange(-CUBE - 20, 20, 0.5); vs = np.arange(-20, CUBE + 20, 0.5)
    UU, VV = np.meshgrid(us, vs)
    g = ((UU > -CUBE) & (UU < -lo) & (VV > lo) & (VV < CUBE)).astype(np.float32)
    return us, vs, g, (-CUBE, -lo, lo, CUBE)

# ---------------------------------------------------------------- materials
def node_tree(mat):
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes): nt.nodes.remove(n)
    return nt, nt.nodes, nt.links

def add(nodes, kind, **props):
    n = nodes.new(kind)
    for k, v in props.items():
        if k.startswith('in_'):
            n.inputs[k[3:].replace('_', ' ')].default_value = v
        else:
            setattr(n, k, v)
    return n

def hex_holes(nodes, links, coord, pitch, radius):
    """Returns a node output that is 1 inside a hex-lattice hole, 0 on the metal."""
    sq3 = math.sqrt(3)
    cell = (pitch, pitch * sq3, 1.0)
    def lattice(offset):
        sub = add(nodes, 'ShaderNodeVectorMath', operation='SUBTRACT')
        sub.inputs[1].default_value = offset
        links.new(coord, sub.inputs[0])
        div = add(nodes, 'ShaderNodeVectorMath', operation='DIVIDE'); div.inputs[1].default_value = cell
        links.new(sub.outputs[0], div.inputs[0])
        fr = add(nodes, 'ShaderNodeVectorMath', operation='FRACTION'); links.new(div.outputs[0], fr.inputs[0])
        c = add(nodes, 'ShaderNodeVectorMath', operation='SUBTRACT'); c.inputs[1].default_value = (0.5, 0.5, 0.0)
        links.new(fr.outputs[0], c.inputs[0])
        mul = add(nodes, 'ShaderNodeVectorMath', operation='MULTIPLY'); mul.inputs[1].default_value = (cell[0], cell[1], 0.0)
        links.new(c.outputs[0], mul.inputs[0])
        ln = add(nodes, 'ShaderNodeVectorMath', operation='LENGTH'); links.new(mul.outputs[0], ln.inputs[0])
        return ln.outputs['Value']
    d1 = lattice((0, 0, 0)); d2 = lattice((pitch / 2, pitch * sq3 / 2, 0))
    mn = add(nodes, 'ShaderNodeMath', operation='MINIMUM'); links.new(d1, mn.inputs[0]); links.new(d2, mn.inputs[1])
    # soft edge so holes do not alias
    sm = add(nodes, 'ShaderNodeMapRange', interpolation_type='SMOOTHSTEP')
    sm.inputs['From Min'].default_value = radius * 1.08; sm.inputs['From Max'].default_value = radius * 0.92
    links.new(mn.outputs[0], sm.inputs['Value'])
    return sm.outputs['Result']

def make_material(name, kind, tint=0.026):
    mat = bpy.data.materials.new(name)
    nt, nodes, links = node_tree(mat)
    out = add(nodes, 'ShaderNodeOutputMaterial')
    bsdf = add(nodes, 'ShaderNodeBsdfPrincipled')
    links.new(bsdf.outputs[0], out.inputs['Surface'])
    tc = add(nodes, 'ShaderNodeTexCoord')
    coord = tc.outputs['Object']
    base = (tint, tint, tint, 1)
    bsdf.inputs['Base Color'].default_value = base
    if kind == 'satin':          # anodized black, soft sheen
        bsdf.inputs['Metallic'].default_value = 0.0
        bsdf.inputs['Roughness'].default_value = 0.42
        bsdf.inputs['Specular IOR Level'].default_value = 0.6
        nz = add(nodes, 'ShaderNodeTexNoise', in_Scale=900.0, in_Detail=2.0)
        links.new(coord, nz.inputs['Vector'])
        bump = add(nodes, 'ShaderNodeBump', in_Strength=0.04, in_Distance=0.002)
        links.new(nz.outputs['Fac'], bump.inputs['Height']); links.new(bump.outputs[0], bsdf.inputs['Normal'])
    elif kind == 'gloss':        # piano black
        bsdf.inputs['Roughness'].default_value = 0.12
        bsdf.inputs['Coat Weight'].default_value = 1.0
        bsdf.inputs['Coat Roughness'].default_value = 0.05
    elif kind == 'brushed':      # brushed dark metal
        bsdf.inputs['Metallic'].default_value = 1.0
        bsdf.inputs['Base Color'].default_value = (0.09, 0.09, 0.09, 1)
        mp = add(nodes, 'ShaderNodeMapping'); mp.inputs['Scale'].default_value = (3.0, 400.0, 3.0)
        links.new(coord, mp.inputs['Vector'])
        nz = add(nodes, 'ShaderNodeTexNoise', in_Scale=12.0, in_Detail=8.0)
        links.new(mp.outputs[0], nz.inputs['Vector'])
        rr = add(nodes, 'ShaderNodeMapRange'); rr.inputs['To Min'].default_value = 0.22; rr.inputs['To Max'].default_value = 0.38
        links.new(nz.outputs['Fac'], rr.inputs['Value']); links.new(rr.outputs['Result'], bsdf.inputs['Roughness'])
        bsdf.inputs['Anisotropic'].default_value = 0.35
        bump = add(nodes, 'ShaderNodeBump', in_Strength=0.08, in_Distance=0.002)
        links.new(nz.outputs['Fac'], bump.inputs['Height']); links.new(bump.outputs[0], bsdf.inputs['Normal'])
    elif kind == 'rubber':       # fine stipple, almost no shine
        bsdf.inputs['Roughness'].default_value = 0.75
        bsdf.inputs['Base Color'].default_value = (0.015, 0.015, 0.015, 1)
        nz = add(nodes, 'ShaderNodeTexNoise', in_Scale=2600.0, in_Detail=1.0)
        links.new(coord, nz.inputs['Vector'])
        bump = add(nodes, 'ShaderNodeBump', in_Strength=0.35, in_Distance=0.003)
        links.new(nz.outputs['Fac'], bump.inputs['Height']); links.new(bump.outputs[0], bsdf.inputs['Normal'])
    elif kind in ('grille', 'grille_fine'):  # perforated speaker-grille metal
        pitch, radius = (0.17, 0.064) if kind == 'grille' else (0.10, 0.038)
        holes_all = hex_holes(nodes, links, coord, pitch, radius)
        sep = add(nodes, 'ShaderNodeSeparateXYZ'); links.new(tc.outputs['Normal'], sep.inputs[0])
        cap = add(nodes, 'ShaderNodeMath', operation='GREATER_THAN'); cap.inputs[1].default_value = 0.95
        links.new(sep.outputs['Z'], cap.inputs[0])
        hm = add(nodes, 'ShaderNodeMath', operation='MULTIPLY')
        links.new(holes_all, hm.inputs[0]); links.new(cap.outputs[0], hm.inputs[1])
        holes = hm.outputs[0]
        bsdf.inputs['Metallic'].default_value = 1.0
        mix = add(nodes, 'ShaderNodeMix', data_type='RGBA')
        g_ = 0.16 if kind == 'grille' else 0.11
        mix.inputs['A'].default_value = (g_, g_, g_, 1); mix.inputs['B'].default_value = (0.0, 0.0, 0.0, 1)
        links.new(holes, mix.inputs['Factor']); links.new(mix.outputs['Result'], bsdf.inputs['Base Color'])
        rr = add(nodes, 'ShaderNodeMapRange'); rr.inputs['To Min'].default_value = 0.38; rr.inputs['To Max'].default_value = 1.0
        links.new(holes, rr.inputs['Value']); links.new(rr.outputs['Result'], bsdf.inputs['Roughness'])
        inv = add(nodes, 'ShaderNodeMath', operation='SUBTRACT'); inv.inputs[0].default_value = 1.0
        links.new(holes, inv.inputs[1])
        bump = add(nodes, 'ShaderNodeBump', in_Strength=1.0, in_Distance=0.01)
        links.new(inv.outputs[0], bump.inputs['Height']); links.new(bump.outputs[0], bsdf.inputs['Normal'])
    return mat

MATS = {k: make_material(k, k) for k in ('satin', 'gloss', 'brushed', 'rubber', 'grille', 'grille_fine')}

# ---------------------------------------------------------------- geometry
def face_object(name, loops, basis, origin, depth, bevel, mat):
    cu = bpy.data.curves.new(name, 'CURVE')
    cu.dimensions = '2D'; cu.fill_mode = 'BOTH'
    cu.extrude = (depth / 2 - bevel) * U   # bevel adds its own depth on both caps
    cu.bevel_depth = bevel * U; cu.bevel_resolution = 4
    cu.offset = -bevel * U if hasattr(cu, 'offset') else 0
    for l in loops:
        sp = cu.splines.new('POLY'); sp.points.add(len(l) - 1)
        for p, (x, y) in zip(sp.points, l):
            p.co = (x * U, y * U, 0.0, 1.0)
        sp.use_cyclic_u = True
    ob = bpy.data.objects.new(name, cu)
    bpy.context.collection.objects.link(ob)
    bx, by, bz = basis
    m = Matrix((bx, by, bz)).transposed()
    ob.matrix_world = Matrix.Translation(Vector(origin) * U) @ m.to_4x4()
    ob.data.materials.append(mat)
    # convert so object-space texture coords are the face plane (u, v)
    return ob

def plane_frames():
    # (basis vectors for local x, y, z) and the world point that local (0,0,0) sits on,
    # pushed half a slab inward so the visible face is exactly on the plane.
    return {
        'top':   ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        'left':  ((1, 0, 0), (0, 0, 1), (0, -1, 0)),
        'right': ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
    }

DEPTH, BEVEL = 30.0, 4.0
CUBE = 574.5                    # cube edge; the S panels fill each side face edge to edge
GAP = 8.0                       # seam between the top plate and the S panels
frames = plane_frames()

def fit(loops, bb, uoff):
    # stretch the S so its outer edges land exactly on the cube face edges
    out = []
    for l in loops:
        u = (l[:, 0] - bb[0]) / (bb[1] - bb[0]) * CUBE + uoff
        top = -(DEPTH + GAP)       # S panels hang below the plate, seam GAP under it
        v = (l[:, 1] - bb[2]) / (bb[3] - bb[2]) * (CUBE + top) - CUBE
        out.append(np.stack([u, v], 1))
    return out

STROKE = 120.0                  # block S: bar and upright thickness

def block_s(uoff):
    """Square-cornered S filling the face under the plate: 3 bars, 2 uprights."""
    Wd = CUBE; top = -(DEPTH + GAP); Ht = CUBE + top; w = STROKE; c = (Ht - 3 * w) / 2
    y = [0, w, w + c, 2 * w + c, 2 * w + 2 * c, Ht]
    pts = [(0, y[0]), (Wd, y[0]), (Wd, y[3]), (w, y[3]), (w, y[4]), (Wd, y[4]), (Wd, y[5]),
           (0, y[5]), (0, y[2]), (Wd - w, y[2]), (Wd - w, y[1]), (0, y[1])]
    return [np.array([(u + uoff, v - CUBE) for u, v in pts], dtype=float)]

def s_rects():
    """Block S as rectangles (u0, u1, v0, v1) in face coords, v measured up from the cube bottom."""
    Wd = CUBE; Ht = CUBE - DEPTH - GAP; w = STROKE; c = (Ht - 3 * w) / 2
    y = [0, w, w + c, 2 * w + c, 2 * w + 2 * c, Ht]
    return [(0, Wd, y[0], y[1]), (Wd - w, Wd, y[1], y[2]), (0, Wd, y[2], y[3]),
            (0, w, y[3], y[4]), (0, Wd, y[4], y[5])]

def wedge(name, tri, z0, z1):
    """Triangular prism cutter (never rendered)."""
    bm = bmesh.new()
    lo = [bm.verts.new((x * U, y * U, z0 * U)) for x, y in tri]
    hi = [bm.verts.new((x * U, y * U, z1 * U)) for x, y in tri]
    bm.faces.new(lo[::-1]); bm.faces.new(hi)
    for i in range(3):
        j = (i + 1) % 3; bm.faces.new([lo[i], lo[j], hi[j], hi[i]])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free()
    ob = bpy.data.objects.new(name, me); bpy.context.collection.objects.link(ob)
    ob.hide_render = True; ob.display_type = 'WIRE'
    return ob

def box_union(name, boxes, mat, cutters=()):
    """One solid mesh from axis-aligned boxes: shared volume merges, nothing overlaps."""
    ax = [sorted({b[2 * k] for b in boxes} | {b[2 * k + 1] for b in boxes}) for k in range(3)]
    n = [len(a) - 1 for a in ax]
    occ = np.zeros(n, bool)
    for i in range(n[0]):
        for j in range(n[1]):
            for k in range(n[2]):
                c = [(ax[0][i] + ax[0][i + 1]) / 2, (ax[1][j] + ax[1][j + 1]) / 2, (ax[2][k] + ax[2][k + 1]) / 2]
                occ[i, j, k] = any(b[0] < c[0] < b[1] and b[2] < c[1] < b[3] and b[4] < c[2] < b[5] for b in boxes)
    bm = bmesh.new(); vmap = {}
    def V(i, j, k):
        key = (i, j, k)
        if key not in vmap: vmap[key] = bm.verts.new((ax[0][i] * U, ax[1][j] * U, ax[2][k] * U))
        return vmap[key]
    def filled(i, j, k):
        return 0 <= i < n[0] and 0 <= j < n[1] and 0 <= k < n[2] and occ[i, j, k]
    for i in range(n[0]):
        for j in range(n[1]):
            for k in range(n[2]):
                if not occ[i, j, k]: continue
                if not filled(i + 1, j, k): bm.faces.new([V(i+1, j, k), V(i+1, j+1, k), V(i+1, j+1, k+1), V(i+1, j, k+1)])
                if not filled(i - 1, j, k): bm.faces.new([V(i, j, k), V(i, j, k+1), V(i, j+1, k+1), V(i, j+1, k)])
                if not filled(i, j + 1, k): bm.faces.new([V(i, j+1, k), V(i, j+1, k+1), V(i+1, j+1, k+1), V(i+1, j+1, k)])
                if not filled(i, j - 1, k): bm.faces.new([V(i, j, k), V(i+1, j, k), V(i+1, j, k+1), V(i, j, k+1)])
                if not filled(i, j, k + 1): bm.faces.new([V(i, j, k+1), V(i+1, j, k+1), V(i+1, j+1, k+1), V(i, j+1, k+1)])
                if not filled(i, j, k - 1): bm.faces.new([V(i, j, k), V(i, j+1, k), V(i+1, j+1, k), V(i+1, j, k)])
    bmesh.ops.dissolve_limit(bm, angle_limit=math.radians(1), verts=bm.verts[:], edges=bm.edges[:])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free()
    ob = bpy.data.objects.new(name, me); bpy.context.collection.objects.link(ob)
    me.materials.append(mat)
    for ct in cutters:
        bo = ob.modifiers.new('miter', 'BOOLEAN'); bo.operation = 'DIFFERENCE'; bo.solver = 'EXACT'; bo.object = ct
    bv = ob.modifiers.new('bevel', 'BEVEL'); bv.width = BEVEL * U; bv.segments = 4
    bv.limit_method = 'ANGLE'; bv.harden_normals = True
    for f in me.polygons: f.use_smooth = True
    return ob

def place(face, loops, mat, name):
    bx, by, bz = frames[face]
    half = DEPTH / 2
    if face == 'top': origin = (0, 0, -half)
    elif face == 'left': origin = (0, half, 0)
    else: origin = (-half, 0, 0)
    return face_object(name, loops, (bx, by, bz), origin, DEPTH, BEVEL, mat)

DEPTH, BEVEL, CUBE, GAP = 30.0, 4.0, 574.5, 0.0
tu, tv, tg, t_bb = top_grid()

PLAN = {
    # option: (top material, side material, tiled?)
    'grille': ('grille', 'satin', None),
    'mono':   ('gloss', 'satin', None),
    'rubik':  (None, None, (3, 3)),
}
top_mat, side_mat, cells = PLAN[OPTION]

# hand-picked tile materials, like the reference cube: mostly matte, a few grilles and metals
TILE_TOP = {(0, 0): 'satin', (1, 0): 'grille', (2, 0): 'satin', (0, 1): 'brushed', (1, 1): 'satin',
            (2, 1): 'grille_fine', (0, 2): 'rubber', (1, 2): 'satin', (2, 2): 'brushed'}
TILE_LEFT = {(0, 0): 'rubber', (1, 0): 'satin', (2, 0): 'satin', (0, 1): 'satin', (1, 1): 'brushed',
             (2, 1): 'satin', (0, 2): 'grille', (1, 2): 'satin', (2, 2): 'rubber'}
TILE_RIGHT = {(0, 0): 'satin', (1, 0): 'rubber', (2, 0): 'satin', (0, 1): 'grille_fine', (1, 1): 'satin',
              (2, 1): 'brushed', (0, 2): 'satin', (1, 2): 'grille', (2, 2): 'satin'}

objs = []
if cells is None:
    objs.append(box_union('top', [(-CUBE, 0, 0, CUBE, -DEPTH, 0)], MATS[top_mat]))
    rects = s_rects(); t, e = DEPTH, 20.0
    left = [(u0 - CUBE, u1 - CUBE, 0, t, v0 - CUBE, v1 - CUBE) for u0, u1, v0, v1 in rects]
    right = [(-t, 0, u0, u1, v0 - CUBE, v1 - CUBE) for u0, u1, v0, v1 in rects]
    # 45-degree miter down the whole front edge: each S stays entirely on its own face,
    # two separate pieces that touch only where both reach the corner
    z0, z1 = -CUBE - e, -t + e
    lcut = [wedge('lcut', [(-t - e, t + e), (e, t + e), (e, -e)], z0, z1)]
    rcut = [wedge('rcut', [(-t - e, t + e), (-t - e, -e), (e, -e)], z0, z1)]
    objs.append(box_union('left', left, MATS[side_mat], lcut))
    objs.append(box_union('right', right, MATS[side_mat], rcut))
else:
    for face, (g, us, vs, bb), table in (('top', (tg, tu, tv, t_bb), TILE_TOP),
                                         ('left', (LG, LU, LV, L_BB), TILE_LEFT),
                                         ('right', (RG, RU, RV, R_BB), TILE_RIGHT)):
        for (a_, b_), ls in pieces(g, us, vs, bb, cells):
            if face != 'top': ls = fit(ls, bb, -CUBE if face == 'left' else 0.0)
            objs.append(place(face, ls, MATS[table[(a_, b_)]], f'{face}_{a_}{b_}'))

# group under an empty so the hero view can tilt the whole mark
root = bpy.data.objects.new('mark', None); bpy.context.collection.objects.link(root)
for o in objs: o.parent = root

# ---------------------------------------------------------------- camera + light
SC = math.sin(math.radians(24)); CC = math.sqrt(1 - SC * SC)   # camera 24 deg above the cube
r = Vector((K, K, 0)); cdir = Vector((K * CC, -K * CC, SC)); up = Vector((-K * SC, K * SC, CC))
target = Vector((-CUBE / 2, CUBE / 2, -CUBE / 2)) * U

if VIEW == 'hero':
    # tilt like the reference: turned and rolled, floating
    pivot = target.copy()
    root.location = -pivot
    holder = bpy.data.objects.new('holder', None); bpy.context.collection.objects.link(holder)
    root.parent = holder; holder.location = pivot
    holder.rotation_euler = (math.radians(-14), math.radians(10), math.radians(-18))

cam_data = bpy.data.cameras.new('cam')
cam = bpy.data.objects.new('cam', cam_data); bpy.context.collection.objects.link(cam)
cam.location = target + cdir * 60
cam.rotation_euler = (math.radians(90) - math.asin(SC), 0, math.radians(45))
if VIEW == 'hero':
    cam_data.type = 'PERSP'; cam_data.lens = 85
    cam.location = target + cdir * 64
    cam_data.clip_end = 500
else:
    cam_data.type = 'ORTHO'; cam_data.ortho_scale = 2 * CUBE * 1.12 * U
    cam_data.clip_end = 500
bpy.context.scene.camera = cam

def area_light(name, pos, size, power, size_y=None):
    ld = bpy.data.lights.new(name, 'AREA'); ld.energy = power
    ld.shape = 'RECTANGLE'; ld.size = size; ld.size_y = size_y or size
    lo = bpy.data.objects.new(name, ld); bpy.context.collection.objects.link(lo)
    lo.location = pos
    d = (target - pos).normalized()
    lo.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
    return lo

# key: big softbox high and to the left of camera; rim: behind, upper right; kicker low right
area_light('key', target + (-r * 13 + up * 8 + cdir * 6), 9, 8000, 5)
area_light('rim', target + (r * 10 + up * 7 - cdir * 4), 4, 4000, 10)
area_light('kick', target + (r * 9 - up * 6 + cdir * 6), 6, 1400)
area_light('top', target + (up * 16 + cdir * 2), 12, 3500, 3)

world = bpy.data.worlds.new('w'); bpy.context.scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes['Background']
bg.inputs['Color'].default_value = (0.03, 0.03, 0.03, 1); bg.inputs['Strength'].default_value = 1.0

# ---------------------------------------------------------------- render
sc = bpy.context.scene
sc.render.engine = 'CYCLES'
prefs = bpy.context.preferences.addons['cycles'].preferences
try:
    prefs.compute_device_type = 'METAL'; prefs.get_devices()
    for d in prefs.devices: d.use = True
    sc.cycles.device = 'GPU'
except Exception as e:
    print('gpu fallback', e)
sc.cycles.samples = 160
sc.cycles.use_denoising = True
sc.render.film_transparent = True
aspect = 1.0
sc.render.resolution_x = RES; sc.render.resolution_y = int(RES * aspect)
sc.render.resolution_percentage = 100
sc.view_settings.view_transform = 'AgX'
try: sc.view_settings.look = 'AgX - Medium High Contrast'
except Exception: pass
sc.render.image_settings.file_format = 'PNG'; sc.render.image_settings.color_mode = 'RGBA'
sc.render.filepath = OUT
dg = bpy.context.evaluated_depsgraph_get()
for o in objs:
    ev = o.evaluated_get(dg); me = ev.to_mesh()
    vs_ = np.array([(o.matrix_world @ v.co)[:] for v in me.vertices]) / U
    print('BOUNDS', o.name, np.round(vs_.min(0), 1), np.round(vs_.max(0), 1))
    ev.to_mesh_clear()
bpy.ops.render.render(write_still=True)
print('WROTE', OUT)
