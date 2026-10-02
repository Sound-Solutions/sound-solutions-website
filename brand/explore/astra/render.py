"""Sound Solutions logo exploration, v2. Blender 5.1, no external packages.

Render all five:
  /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup --python-exit-code 1 -P brand/explore/astra/render.py
Optional arguments after --: --only 1 3 --resolution 600 --samples 64
Final delivery defaults to 1200px. make_previews.py makes 252px copies and proofs.
All materials, geometry, lighting, and cameras are defined here. No source assets.
"""

import argparse
import math
from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parent
NAMES = {1: "soft-s", 2: "cone", 3: "fader", 4: "pulse", 5: "ripple"}
args = argparse.ArgumentParser()
args.add_argument("--only", type=int, nargs="+", default=list(NAMES))
args.add_argument("--resolution", type=int, default=1200)
args.add_argument("--samples", type=int, default=384)
OPTS = args.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])


def linear(hex_color):
    values = [int(hex_color[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return tuple(v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in values)


def material(name, gray, metal=0.0, roughness=0.45, texture=None):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    tree = mat.node_tree
    shader = tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (gray, gray, gray, 1)
    shader.inputs["Metallic"].default_value = metal
    shader.inputs["Roughness"].default_value = roughness
    if texture:
        coord = tree.nodes.new("ShaderNodeTexCoord")
        mapping = tree.nodes.new("ShaderNodeVectorMath")
        mapping.operation = "MULTIPLY"
        mapping.inputs[1].default_value = (4, 450, 4) if texture == "brushed" else (240, 240, 240)
        tree.links.new(coord.outputs["Object"], mapping.inputs[0])
        noise = tree.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 4
        noise.inputs["Detail"].default_value = 2
        tree.links.new(mapping.outputs[0], noise.inputs["Vector"])
        bump = tree.nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = 0.12 if texture == "brushed" else 0.16
        bump.inputs["Distance"].default_value = 0.005 if texture == "brushed" else 0.007
        tree.links.new(noise.outputs["Fac"], bump.inputs["Height"])
        tree.links.new(bump.outputs[0], shader.inputs["Normal"])
    return mat


def accent(name, hex_color):
    # Camera-only colored insert: exact saturated color, no tinted reflections,
    # white specular highlights, halo, or low-alpha wash.
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    tree = mat.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (*linear(hex_color), 1)
    emission.inputs["Strength"].default_value = 1
    diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
    diffuse.inputs["Color"].default_value = (0.03, 0.03, 0.03, 1)
    lightpath = tree.nodes.new("ShaderNodeLightPath")
    mix = tree.nodes.new("ShaderNodeMixShader")
    tree.links.new(lightpath.outputs["Is Camera Ray"], mix.inputs[0])
    tree.links.new(diffuse.outputs[0], mix.inputs[1])
    tree.links.new(emission.outputs[0], mix.inputs[2])
    tree.links.new(mix.outputs[0], output.inputs["Surface"])
    return mat


def slab(name, points, depth, z, mat, edge=0.03):
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "2D"
    curve.resolution_u = 24
    curve.fill_mode = "BOTH"
    curve.extrude = depth / 2 - edge
    curve.bevel_depth = edge
    curve.bevel_resolution = 5
    spline = curve.splines.new("POLY")
    spline.points.add(len(points) - 1)
    for dest, (x, y) in zip(spline.points, points):
        dest.co = (float(x), float(y), 0, 1)
    spline.use_cyclic_u = True
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    obj.location.z = z
    obj.data.materials.append(mat)
    return obj


def roundrect(name, x, y, width, height, radius, depth, z, mat, edge=0.02):
    points = []
    for cx, cy, start in ((width / 2 - radius, height / 2 - radius, 0),
                          (-width / 2 + radius, height / 2 - radius, 90),
                          (-width / 2 + radius, -height / 2 + radius, 180),
                          (width / 2 - radius, -height / 2 + radius, 270)):
        for angle in np.linspace(start, start + 90, 20):
            a = math.radians(angle)
            points.append((x + cx + radius * math.cos(a), y + cy + radius * math.sin(a)))
    return slab(name, points, depth, z, mat, edge)


def lathe(name, profile, mat, segments=192):
    vertices = [(r * math.cos(a), r * math.sin(a), z)
                for r, z in profile for a in np.linspace(0, 2 * math.pi, segments, endpoint=False)]
    faces = []
    for row in range(len(profile) - 1):
        for i in range(segments):
            j = (i + 1) % segments
            faces.append((row * segments + i, row * segments + j,
                          (row + 1) * segments + j, (row + 1) * segments + i))
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(mat)
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    return obj


def torus(name, major, minor, z, mat):
    bpy.ops.mesh.primitive_torus_add(major_radius=major, minor_radius=minor,
                                    major_segments=192, minor_segments=32, location=(0, 0, z))
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(mat)
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    return obj


def ribbon(name, path, width, depth, z, mat, edge=0.02):
    p = np.array(path)
    tangent = np.gradient(p, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    half = width / 2
    pts = list(p + normal * half)
    # Round caps at both ends, keeping the silhouette soft at signature size.
    a = math.atan2(normal[-1, 1], normal[-1, 0])
    pts += [p[-1] + half * np.array((math.cos(t), math.sin(t))) for t in np.linspace(a, a - math.pi, 32)]
    pts += list((p - normal * half)[::-1])
    a = math.atan2(-normal[0, 1], -normal[0, 0])
    pts += [p[0] + half * np.array((math.cos(t), math.sin(t))) for t in np.linspace(a, a - math.pi, 32)]
    return slab(name, pts, depth, z, mat, edge)


def bezier(points, steps=100):
    p = np.array(points)
    t = np.linspace(0, 1, steps, endpoint=False)[:, None]
    return (1 - t)**3 * p[0] + 3 * (1 - t)**2 * t * p[1] + 3 * (1 - t) * t**2 * p[2] + t**3 * p[3]


def soft_s(m):
    path = np.vstack((
        bezier(((0.61, 0.78), (0.27, 1.09), (-0.72, 1.03), (-0.72, 0.47))),
        bezier(((-0.72, 0.47), (-0.72, -0.09), (0.69, 0.06), (0.69, -0.53))),
        bezier(((0.69, -0.53), (0.69, -1.1), (-0.31, -1.1), (-0.66, -0.77))),
        [[-0.66, -0.77]],
    ))
    ribbon("One flowing S", path, 0.43, 0.25, 0, m["metal"], 0.035)
    ribbon("Cyan inlay", path, 0.046, 0.016, 0.129, m["cyan"], 0.003)


def cone(m):
    lathe("Outer speaker rim", [(0, -0.16), (1.16, -0.16), (1.23, -0.1),
          (1.26, 0), (1.25, 0.08), (1.21, 0.13), (1.10, 0.13), (1.08, 0.07)], m["metal"])
    torus("Rolled rubber surround", 0.988, 0.12, 0.055, m["rubber"])
    lathe("Paper speaker cone", [(0.90, 0.10), (0.86, 0.06), (0.80, 0.025),
          (0.70, -0.025), (0.57, -0.11), (0.44, -0.18), (0.31, -0.21), (0, -0.21)], m["paper"])
    torus("Orange center ring", 0.357, 0.027, -0.13, m["orange"])
    profile = [(0.345 * math.sin(t), -0.16 + 0.17 * math.cos(t))
               for t in np.linspace(0, math.pi / 2, 60)]
    lathe("Satin dust cap", profile, m["satin"])
    torus("Fine outside lip", 1.226, 0.017, 0.10, m["edge"])


def fader(m):
    roundrect("Fader body", 0, 0, 1.16, 2.75, 0.27, 0.23, -0.10, m["metal"], 0.035)
    roundrect("Rubber face", 0, 0, 1.045, 2.63, 0.23, 0.05, 0.03, m["satin"], 0.009)
    roundrect("Travel slot", 0, 0, 0.095, 2.21, 0.046, 0.021, 0.065, m["black"], 0.005)
    for y in np.linspace(-0.94, 0.94, 9):
        for x in (-0.36, 0.36):
            roundrect("Scale line", x, y, 0.15, 0.029, 0.013, 0.01, 0.062, m["edge"], 0.002)
    roundrect("Fader cap", 0, 0.36, 1.34, 0.48, 0.1, 0.26, 0.20, m["metal"], 0.031)
    for y in (0.225, 0.495):
        roundrect("Grip groove", 0, y, 1.08, 0.028, 0.012, 0.014, 0.334, m["black"], 0.002)
    roundrect("Green position line", 0, 0.36, 1.12, 0.050, 0.02, 0.014, 0.333, m["green"], 0.002)


def pulse(m):
    # Unequal, rounded columns read as audio amplitude; no diagonals or letters.
    heights = (0.73, 1.55, 2.40, 1.87, 0.94)
    for i, height in enumerate(heights):
        x = (i - 2) * 0.49
        roundrect("Pulse column", x, 0, 0.32, height, 0.16, 0.27, 0,
                  m["metal"] if i % 2 == 0 else m["satin"], 0.024)
        if i == 2:
            roundrect("Cyan pulse", x, 0, 0.09, height - 0.18, 0.045, 0.016, 0.14, m["cyan"], 0.003)


def ripple(m):
    # One offset source and two generous circular arcs, opening to the right.
    # Rounded ends make these sound-pressure waves, not heraldic chevrons.
    for radius in (0.67, 1.15):
        path = [(radius * math.cos(t), radius * math.sin(t))
                for t in np.linspace(math.radians(51), math.radians(309), 240)]
        ribbon("Sound wave", path, 0.245, 0.20, 0, m["metal"], 0.025)
    roundrect("Orange sound source", 0.055, 0, 0.43, 0.43, 0.215, 0.12, 0.04, m["orange"], 0.01)


def area(name, position, power, size, size_y=None):
    data = bpy.data.lights.new(name, "AREA")
    data.energy = power
    data.shape = "RECTANGLE"
    data.size = size
    data.size_y = size_y or size
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.location = position
    obj.rotation_euler = (-obj.location).to_track_quat("-Z", "Y").to_euler()


def setup_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    try:
        prefs.compute_device_type = "METAL"
        prefs.get_devices()
        gpu = False
        for device in prefs.devices:
            device.use = device.type == "METAL"
            gpu = gpu or device.use
        scene.cycles.device = "GPU" if gpu else "CPU"
    except Exception as error:
        print("Metal unavailable; using CPU:", error)
        scene.cycles.device = "CPU"
    scene.cycles.samples = OPTS.samples
    scene.cycles.seed = 29
    # Color denoising bleeds a faint tint from the insert onto neutral faces.
    # More samples preserve exactly neutral material pixels without that bleed.
    scene.cycles.use_denoising = False
    scene.render.film_transparent = True
    scene.render.resolution_x = OPTS.resolution
    scene.render.resolution_y = OPTS.resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.image_settings.compression = 70
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1
    world = bpy.data.worlds.new("Neutral studio")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.07, 0.07, 0.07, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.40
    scene.world = world
    return {
        "metal": material("Brushed graphite", 0.036, 0.72, 0.37, "brushed"),
        "edge": material("Neutral rim", 0.10, 0.6, 0.40),
        "satin": material("Satin black", 0.015, 0.20, 0.47, "matte"),
        "rubber": material("Soft black rubber", 0.009, 0, 0.78, "matte"),
        "paper": material("Pressed black cone", 0.018, 0.10, 0.62, "matte"),
        "black": material("Deep black", 0.004, 0, 0.82),
        "cyan": accent("Cyan 00D4FF", "00D4FF"),
        "green": accent("Green 39FF14", "39FF14"),
        "orange": accent("Orange FF9500", "FF9500"),
    }


def fit_camera(index):
    data = bpy.data.cameras.new("Logo camera")
    camera = bpy.data.objects.new("Logo camera", data)
    bpy.context.collection.objects.link(camera)
    camera.location = {1: (2.2, -2.5, 14), 2: (1.8, -2.5, 14),
                       3: (3.2, -2.8, 14), 4: (2.5, -2.5, 14), 5: (1.8, -2.3, 14)}[index]
    # Project world Y into the camera plane. Track-quat's usual world-Z up
    # introduces roll when looking down at an XY mark.
    backward = camera.location.normalized()
    right = Vector((0, 1, 0)).cross(backward).normalized()
    up = backward.cross(right)
    camera.rotation_euler = Matrix((right, up, backward)).transposed().to_euler()
    data.type = "ORTHO"
    bpy.context.scene.camera = camera
    bpy.context.view_layer.update()
    inverse = camera.matrix_world.inverted()
    graph = bpy.context.evaluated_depsgraph_get()
    points = []
    for obj in bpy.context.scene.objects:
        if obj.type not in ("MESH", "CURVE"):
            continue
        evaluated = obj.evaluated_get(graph)
        mesh = evaluated.to_mesh()
        points.extend(inverse @ evaluated.matrix_world @ v.co for v in mesh.vertices)
        evaluated.to_mesh_clear()
    low_x, high_x = min(p.x for p in points), max(p.x for p in points)
    low_y, high_y = min(p.y for p in points), max(p.y for p in points)
    data.ortho_scale = max(high_x - low_x, high_y - low_y) / 0.84
    camera.location += camera.rotation_euler.to_matrix() @ Vector(((low_x + high_x) / 2, (low_y + high_y) / 2, 0))


for number in OPTS.only:
    mats = setup_scene()
    {1: soft_s, 2: cone, 3: fader, 4: pulse, 5: ripple}[number](mats)
    fit_camera(number)
    area("Broad left softbox", (-3.5, 4.2, 6), 340, 4.0, 5.0)
    area("Right strip reflection", (3.8, 1, 3), 180, 2.0, 5.0)
    area("Lower edge light", (-1.2, -4, 3.4), 120, 4.0, 1.4)
    output = ROOT / f"{number}-{NAMES[number]}.png"
    bpy.context.scene.render.filepath = str(output)
    bpy.ops.render.render(write_still=True)
    print("WROTE", output, flush=True)
