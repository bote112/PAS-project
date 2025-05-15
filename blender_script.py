import bpy
import os
import trimesh

# === CONFIGURATION ===
ply_dir = "D:/VS_Projects/Python/PAS/SPH-Fluid-Simulation/Fluid Simulation_output"  # <-- update this path
ply_prefix = "frame_"
ply_suffix = ".ply"
frame_start = 0
frame_end = 275
use_geometry_nodes = True  # Set to False if you want raw mesh only

# Optional: reuse one Geometry Nodes modifier
gnode = None

def import_all_ply_and_keyframe():
    global gnode

    for frame in range(frame_start, frame_end + 1):
        frame_str = f"{frame:06d}"
        ply_path = os.path.join(ply_dir, f"{ply_prefix}{frame_str}{ply_suffix}")

        if not os.path.exists(ply_path):
            print(f"[!] Skipping missing file: {ply_path}")
            continue

        # Load mesh using trimesh
        try:
            mesh = trimesh.load(ply_path, process=False)
        except Exception as e:
            print(f"[!] Failed to load {ply_path}: {e}")
            continue

        verts = mesh.vertices
        faces = mesh.faces if hasattr(mesh, 'faces') else []

        blender_mesh = bpy.data.meshes.new(f"FluidMesh_{frame_str}")
        blender_mesh.from_pydata(verts.tolist(), [], faces)
        blender_mesh.update()

        # Create object
        obj = bpy.data.objects.new(f"Fluid_{frame_str}", blender_mesh)
        bpy.context.collection.objects.link(obj)

        # Add Geometry Nodes modifier if requested
        if use_geometry_nodes:
            if gnode is None:
                gnode = bpy.data.node_groups.get("GeometryNodes") or None
                if gnode is None:
                    print("[!] No GeometryNodes modifier found. Skipping.")
            if gnode:
                mod = obj.modifiers.new("GeoNodes", 'NODES')
                mod.node_group = gnode

        # Keyframe visibility
        for f in range(frame_start, frame_end + 1):
            show = (f == frame)
            obj.hide_render = not show
            obj.hide_viewport = not show
            obj.keyframe_insert(data_path="hide_render", frame=f)
            obj.keyframe_insert(data_path="hide_viewport", frame=f)

        print(f"[✔] Imported frame {frame}")

    print("✅ All frames imported and keyframed.")

import_all_ply_and_keyframe()