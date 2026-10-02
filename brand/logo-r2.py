"""Sound Solutions logo, round 2: three directions in the dark material style.

Usage: Blender -b --factory-startup -P logo-r2.py -- <option> <out.png> [res]
  option: rounded | solved | badge
"""
import bpy, bmesh, sys, math, numpy as np
from mathutils import Vector, Matrix

argv = sys.argv[sys.argv.index('--') + 1:]
OPTION, OUT = argv[0], argv[1]
RES = int(argv[2]) if len(argv) > 2 else 1200
U = 0.01                         # design unit -> metres
K = math.sqrt(0.5)

for o in list(bpy.data.objects): bpy.data.objects.remove(o)

# ---------------------------------------------------------------- materials
def add(nodes, kind, **props):
    n = nodes.new(kind)
    for k, v in props.items():
        if k.startswith('in_'): n.inputs[k[3:].replace('_', ' ')].default_value = v
        else: setattr(n, k, v)
    return n

def hex_holes(nodes, links, coord, pitch, radius):
    """1 inside a hex-lattice hole, 0 on the metal."""
    sq3 = math.sqrt(3); cell = (pitch, pitch * sq3, 1.0)
    def lattice(offset):
        sub = add(nodes, 'ShaderNodeVectorMath', operation='SUBTRACT'); sub.inputs[1].default_value = offset
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
    sm = add(nodes, 'ShaderNodeMapRange', interpolation_type='SMOOTHSTEP')
    sm.inputs['From Min'].default_value = radius * 1.08; sm.inputs['From Max'].default_value = radius * 0.92
    links.new(mn.outputs[0], sm.inputs['Value'])
    return sm.outputs['Result']

def material(name, kind, grey=0.16, pitch=0.17):
    mat = bpy.data.materials.new(name); mat.use_nodes = True
    nt = mat.node_tree; nodes, links = nt.nodes, nt.links
    for n in list(nodes): nodes.remove(n)
    out = add(nodes, 'ShaderNodeOutputMaterial'); bsdf = add(nodes, 'ShaderNodeBsdfPrincipled')
    links.new(bsdf.outputs[0], out.inputs['Surface'])
    tc = add(nodes, 'ShaderNodeTexCoord'); coord = tc.outputs['Object']
    if kind == 'satin':
        bsdf.inputs['Base Color'].default_value = (0.026, 0.026, 0.026, 1)
        bsdf.inputs['Roughness'].default_value = 0.42
        bsdf.inputs['Specular IOR Level'].default_value = 0.6
        nz = add(nodes, 'ShaderNodeTexNoise', in_Scale=900.0, in_Detail=2.0); links.new(coord, nz.inputs['Vector'])
        bump = add(nodes, 'ShaderNodeBump', in_Strength=0.04, in_Distance=0.002)
        links.new(nz.outputs['Fac'], bump.inputs['Height']); links.new(bump.outputs[0], bsdf.inputs['Normal'])
    elif kind == 'brushed':
        bsdf.inputs['Metallic'].default_value = 1.0
        bsdf.inputs['Base Color'].default_value = (0.07, 0.07, 0.07, 1)
        mp = add(nodes, 'ShaderNodeMapping'); mp.inputs['Scale'].default_value = (3.0, 3.0, 400.0)
        links.new(coord, mp.inputs['Vector'])
        nz = add(nodes, 'ShaderNodeTexNoise', in_Scale=12.0, in_Detail=8.0); links.new(mp.outputs[0], nz.inputs['Vector'])
        rr = add(nodes, 'ShaderNodeMapRange'); rr.inputs['To Min'].default_value = 0.24; rr.inputs['To Max'].default_value = 0.4
        links.new(nz.outputs['Fac'], rr.inputs['Value']); links.new(rr.outputs['Result'], bsdf.inputs['Roughness'])
        bump = add(nodes, 'ShaderNodeBump', in_Strength=0.06, in_Distance=0.002)
        links.new(nz.outputs['Fac'], bump.inputs['Height']); links.new(bump.outputs[0], bsdf.inputs['Normal'])
    elif kind == 'grille':
        holes_all = hex_holes(nodes, links, coord, pitch, pitch * 0.376)
        sep = add(nodes, 'ShaderNodeSeparateXYZ'); links.new(tc.outputs['Normal'], sep.inputs[0])
        cap = add(nodes, 'ShaderNodeMath', operation='GREATER_THAN'); cap.inputs[1].default_value = 0.95
        links.new(sep.outputs['Z'], cap.inputs[0])
        hm = add(nodes, 'ShaderNodeMath', operation='MULTIPLY')
        links.new(holes_all, hm.inputs[0]); links.new(cap.outputs[0], hm.inputs[1])
        holes = hm.outputs[0]
        bsdf.inputs['Metallic'].default_value = 1.0
        mix = add(nodes, 'ShaderNodeMix', data_type='RGBA')
        mix.inputs['A'].default_value = (grey, grey, grey, 1); mix.inputs['B'].default_value = (0, 0, 0, 1)
        links.new(holes, mix.inputs['Factor']); links.new(mix.outputs['Result'], bsdf.inputs['Base Color'])
        rr = add(nodes, 'ShaderNodeMapRange'); rr.inputs['To Min'].default_value = 0.38; rr.inputs['To Max'].default_value = 1.0
        links.new(holes, rr.inputs['Value']); links.new(rr.outputs['Result'], bsdf.inputs['Roughness'])
        inv = add(nodes, 'ShaderNodeMath', operation='SUBTRACT'); inv.inputs[0].default_value = 1.0
        links.new(holes, inv.inputs[1])
        bump = add(nodes, 'ShaderNodeBump', in_Strength=1.0, in_Distance=0.01)
        links.new(inv.outputs[0], bump.inputs['Height']); links.new(bump.outputs[0], bsdf.inputs['Normal'])
    return mat

# ---------------------------------------------------------------- shapes
def rounded_s(W, H, w, n=48):
    """Geometric S: three straight bars joined by two true half-circle turns.
    Outer edges land exactly on the box [0,W] x [0,H]; the bar ends are cut square."""
    yt, ym, yb = H - w / 2, H / 2, w / 2
    rho = (H - w) / 4
    xl, xr = w / 2 + rho, W - w / 2 - rho
    cl, cr = (yt + ym) / 2, (ym + yb) / 2
    cl_pts = [(W, yt), (xl, yt)]
    cl_pts += [(xl + rho * math.cos(a), cl + rho * math.sin(a))
               for a in np.linspace(math.pi / 2, 3 * math.pi / 2, n)[1:]]
    cl_pts += [(xr, ym)]
    cl_pts += [(xr + rho * math.cos(a), cr + rho * math.sin(a))
               for a in np.linspace(math.pi / 2, -math.pi / 2, n)[1:]]
    cl_pts += [(0, yb)]
    P = np.array(cl_pts, float)
    # offset both sides by w/2 along the left normal of the walking direction
    tang = np.gradient(P, axis=0)
    tang[0] = P[1] - P[0]; tang[-1] = P[-1] - P[-2]
    tang /= np.linalg.norm(tang, axis=1)[:, None]
    nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)
    a, b = P + nrm * w / 2, P - nrm * w / 2
    return np.vstack([a, b[::-1]])

def curve_slab(name, loop, depth, bevel, mat, matrix):
    """Extruded, bevelled slab from a 2D outline; front face on local z = +depth/2."""
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '2D'; cu.fill_mode = 'BOTH'
    cu.extrude = (depth / 2 - bevel) * U; cu.bevel_depth = bevel * U; cu.bevel_resolution = 4
    cu.offset = -bevel * U
    sp = cu.splines.new('POLY'); sp.points.add(len(loop) - 1)
    for p, (x, y) in zip(sp.points, loop): p.co = (x * U, y * U, 0, 1)
    sp.use_cyclic_u = True
    ob = bpy.data.objects.new(name, cu); bpy.context.collection.objects.link(ob)
    ob.matrix_world = matrix; cu.materials.append(mat)
    return ob

def to_mesh(ob):
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
    m = bpy.data.objects.new(ob.name, me); bpy.context.collection.objects.link(m)
    m.matrix_world = ob.matrix_world.copy()
    bpy.data.objects.remove(ob)
    return m

def wedge(name, tri, z0, z1):
    bm = bmesh.new()
    lo = [bm.verts.new((x * U, y * U, z0 * U)) for x, y in tri]
    hi = [bm.verts.new((x * U, y * U, z1 * U)) for x, y in tri]
    bm.faces.new(lo[::-1]); bm.faces.new(hi)
    for i in range(3):
        j = (i + 1) % 3; bm.faces.new([lo[i], lo[j], hi[j], hi[i]])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free()
    ob = bpy.data.objects.new(name, me); bpy.context.collection.objects.link(ob)
    ob.hide_render = True
    return ob

def box(name, lo, hi, mats, bevel, seg=4):
    """Bevelled box; mats = one material, or (top, left(-y), right(+x), other)."""
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        v.co = Vector([(lo[i] + (hi[i] - lo[i]) * (v.co[i] + 0.5)) * U for i in range(3)])
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free()
    ob = bpy.data.objects.new(name, me); bpy.context.collection.objects.link(ob)
    if isinstance(mats, (list, tuple)):
        for m in mats: me.materials.append(m)
        for p in me.polygons:
            nz = p.normal
            p.material_index = 0 if nz.z > 0.5 else 1 if nz.y < -0.5 else 2 if nz.x > 0.5 else 3
    else:
        me.materials.append(mats)
    bv = ob.modifiers.new('bevel', 'BEVEL'); bv.width = bevel * U; bv.segments = seg
    bv.harden_normals = True
    for p in me.polygons: p.use_smooth = True
    return ob

# ---------------------------------------------------------------- scene builders
def iso_camera(target, extent, elev=24):
    s = math.sin(math.radians(elev)); c = math.cos(math.radians(elev))
    cdir = Vector((K * c, -K * c, s)); r = Vector((K, K, 0)); up = Vector((-K * s, K * s, c))
    cam_d = bpy.data.cameras.new('cam'); cam = bpy.data.objects.new('cam', cam_d)
    bpy.context.collection.objects.link(cam)
    cam.location = target + cdir * 60; cam.rotation_euler = (math.radians(90 - elev), 0, math.radians(45))
    cam_d.type = 'ORTHO'; cam_d.ortho_scale = extent * U; cam_d.clip_end = 500
    bpy.context.scene.camera = cam
    return r, up, cdir

def front_camera(target, extent):
    cam_d = bpy.data.cameras.new('cam'); cam = bpy.data.objects.new('cam', cam_d)
    bpy.context.collection.objects.link(cam)
    cam.location = target + Vector((0, 0, 60)); cam.rotation_euler = (0, 0, 0)
    cam_d.type = 'ORTHO'; cam_d.ortho_scale = extent * U; cam_d.clip_end = 500
    bpy.context.scene.camera = cam
    return Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((0, 0, 1))

def light(name, target, pos, size, power, size_y=None):
    ld = bpy.data.lights.new(name, 'AREA'); ld.energy = power
    ld.shape = 'RECTANGLE'; ld.size = size; ld.size_y = size_y or size
    lo = bpy.data.objects.new(name, ld); bpy.context.collection.objects.link(lo)
    lo.location = pos
    lo.rotation_euler = (target - pos).normalized().to_track_quat('-Z', 'Y').to_euler()

def studio(target, r, up, cdir):
    light('key', target, target + (-r * 13 + up * 8 + cdir * 6), 9, 8000, 5)
    light('rim', target, target + (r * 10 + up * 7 - cdir * 4), 4, 4000, 10)
    light('kick', target, target + (r * 9 - up * 6 + cdir * 6), 6, 1400)
    light('top', target, target + (up * 16 + cdir * 2), 12, 3500, 3)

CUBE, T = 574.5, 30.0

if OPTION == 'rounded':
    # his cube: grille plate on top, one rounded S per side face, mitered where they meet
    satin, grille = material('satin', 'satin'), material('grille', 'grille')
    box('top', (-CUBE, 0, -T), (0, CUBE, 0), grille, 4)
    Ht = CUBE - T
    loop = rounded_s(CUBE, Ht, 124)
    e = 20.0
    lm = Matrix(((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))   # local x->X, y->Z, z->-Y
    rm = Matrix(((0, 0, 1, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 1)))    # local x->Y, y->Z, z->+X
    left = to_mesh(curve_slab('left', loop + [-CUBE, -CUBE], T, 4, satin,
                              Matrix.Translation(Vector((0, T / 2, 0)) * U) @ lm))
    right = to_mesh(curve_slab('right', loop + [0, -CUBE], T, 4, satin,
                               Matrix.Translation(Vector((-T / 2, 0, 0)) * U) @ rm))
    for ob, tri in ((left, [(-T - e, T + e), (e, T + e), (e, -e)]),
                    (right, [(-T - e, T + e), (-T - e, -e), (e, -e)])):
        bo = ob.modifiers.new('miter', 'BOOLEAN'); bo.operation = 'DIFFERENCE'; bo.solver = 'EXACT'
        bo.object = wedge('cut', tri, -CUBE - e, -T + e)
    target = Vector((-CUBE / 2, CUBE / 2, -CUBE / 2)) * U
    studio(target, *iso_camera(target, 2 * CUBE * 1.12))

elif OPTION == 'solved':
    # a solved 3x3 cube: every face one material, like the reference cube put in order
    grille, satin, brushed = material('grille', 'grille', 0.16, 0.12), material('satin', 'satin'), material('brushed', 'brushed')
    s, g = CUBE / 3, 9.0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                lo = (-CUBE + i * s + g / 2, j * s + g / 2, -CUBE + k * s + g / 2)
                hi = (-CUBE + (i + 1) * s - g / 2, (j + 1) * s - g / 2, -CUBE + (k + 1) * s - g / 2)
                box(f'c{i}{j}{k}', lo, hi, (grille, satin, brushed, satin), 10, 4)
    target = Vector((-CUBE / 2, CUBE / 2, -CUBE / 2)) * U
    studio(target, *iso_camera(target, 2 * CUBE * 1.12))

elif OPTION == 'badge':
    # one rounded S, raised in satin black on a square of speaker grille
    satin, grille = material('satin', 'satin'), material('grille', 'grille', 0.16, 0.30)
    P, R = 1000.0, 170.0
    th = np.linspace(0, math.pi / 2, 24)
    corners = [(P - R, P - R, 0), (R, P - R, 1), (R, R, 2), (P - R, R, 3)]
    plate = []
    for cx, cy, q in corners:
        plate += [(cx + R * math.cos(t + q * math.pi / 2), cy + R * math.sin(t + q * math.pi / 2)) for t in th]
    plate = np.array(plate) - P / 2
    curve_slab('plate', plate, 70, 10, grille, Matrix.Translation(Vector((0, 0, -35)) * U))
    sw, sh = 560.0, 690.0
    loop = rounded_s(sw, sh, 150) - [sw / 2, sh / 2]
    curve_slab('s', loop, 40, 6, satin, Matrix.Translation(Vector((0, 0, 20)) * U))
    target = Vector((0, 0, 0))
    r, up, cdir = front_camera(target, P * 1.16)
    light('key', target, Vector((-9, 9, 7)), 9, 9000, 5)
    light('rim', target, Vector((9, -6, 4)), 4, 2500, 10)
    light('fill', target, Vector((4, 8, 12)), 12, 500, 4)

world = bpy.data.worlds.new('w'); bpy.context.scene.world = world; world.use_nodes = True
bg = world.node_tree.nodes['Background']
bg.inputs['Color'].default_value = (0.03, 0.03, 0.03, 1); bg.inputs['Strength'].default_value = 1.0

sc = bpy.context.scene; sc.render.engine = 'CYCLES'
prefs = bpy.context.preferences.addons['cycles'].preferences
try:
    prefs.compute_device_type = 'METAL'; prefs.get_devices()
    for d in prefs.devices: d.use = True
    sc.cycles.device = 'GPU'
except Exception as ex: print('gpu fallback', ex)
sc.cycles.samples = 160; sc.cycles.use_denoising = True
sc.render.film_transparent = True
sc.render.resolution_x = sc.render.resolution_y = RES; sc.render.resolution_percentage = 100
sc.view_settings.view_transform = 'AgX'
try: sc.view_settings.look = 'AgX - Medium High Contrast'
except Exception: pass
sc.render.image_settings.file_format = 'PNG'; sc.render.image_settings.color_mode = 'RGBA'
sc.render.filepath = OUT
bpy.ops.render.render(write_still=True)
print('WROTE', OUT)
