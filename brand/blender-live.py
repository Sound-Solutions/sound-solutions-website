"""Live-reload for working on the logo together: when the open .blend changes on disk,
Blender reloads it and goes back to the camera view.

Launch: Blender brand/logo-cube.blend --python brand/blender-live.py
"""
import bpy, os

state = {'path': None, 'mtime': None}

def _mtime(p):
    try: return os.path.getmtime(p)
    except OSError: return None

def _camera_view():
    for win in bpy.context.window_manager.windows:
        for area in win.screen.areas:
            if area.type == 'VIEW_3D':
                for sp in area.spaces:
                    if sp.type == 'VIEW_3D':
                        sp.shading.type = 'RENDERED'
                        sp.region_3d.view_perspective = 'CAMERA'
                        sp.overlay.show_overlays = False

def _tick():
    path = bpy.data.filepath
    if not path:
        return 1.0
    m = _mtime(path)
    if state['path'] != path:
        state['path'], state['mtime'] = path, m
        return 1.0
    if m and state['mtime'] and m > state['mtime'] + 0.5:
        state['mtime'] = m
        win = bpy.context.window_manager.windows[0]
        with bpy.context.temp_override(window=win):
            bpy.ops.wm.revert_mainfile()
        bpy.app.timers.register(_camera_view, first_interval=0.5)
    return 1.0

bpy.app.timers.register(_tick, first_interval=1.0, persistent=True)
