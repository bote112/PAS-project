import taichi as ti
import json
from FluidEnvironment import ParticleSystem
import numpy as np
import os

ti.init(arch=ti.gpu)

""" box_bounds = {
    "x_min": 0.0, "x_max": 4.0,
    "y_min": 0.0, "y_max": 4.0,
    "z_min": 0.0, "z_max": 4.0,
} """

with open('./data/scenes/mario.json', 'r') as f:
    simulation_config = json.load(f)

# Initialize box_bounds from the config
domain_start = simulation_config["Configuration"]["domainStart"]
domain_end = simulation_config["Configuration"]["domainEnd"]

box_bounds = {
    "x_min": domain_start[0], "x_max": domain_end[0],
    "y_min": domain_start[1], "y_max": domain_end[1],
    "z_min": domain_start[2], "z_max": domain_end[2],
}

def compute_box_vertices(bounds):
    return np.array([
        [bounds["x_min"], bounds["y_min"], bounds["z_min"]],
        [bounds["x_max"], bounds["y_min"], bounds["z_min"]],
        [bounds["x_max"], bounds["y_max"], bounds["z_min"]],
        [bounds["x_min"], bounds["y_max"], bounds["z_min"]],
        [bounds["x_min"], bounds["y_min"], bounds["z_max"]],
        [bounds["x_max"], bounds["y_min"], bounds["z_max"]],
        [bounds["x_max"], bounds["y_max"], bounds["z_max"]],
        [bounds["x_min"], bounds["y_max"], bounds["z_max"]],
    ], dtype=np.float32)

vertices_np = compute_box_vertices(box_bounds)

edges_np = np.array([
    0, 1, 1, 2, 2, 3, 3, 0,
    4, 5, 5, 6, 6, 7, 7, 4,
    0, 4, 1, 5, 2, 6, 3, 7
], dtype=np.int32)

# create Taichi fields
box_vertex_point = ti.Vector.field(3, dtype=ti.f32, shape=8)
box_edge_index = ti.field(dtype=ti.i32, shape=24)

# copy numpy data to taichi fields
for i in range(8):
    box_vertex_point[i] = vertices_np[i]

for i in range(24):
    box_edge_index[i] = edges_np[i]

win = ti.ui.Window("SPH - Dark Mode", (1500, 1000))
cnvs = win.get_canvas()
scn = ti.ui.Scene()
cam = ti.ui.Camera()
cam.position(3.0, 6.0, 7.5)  # New camera position and lookat for different view
cam.lookat(1.0, 0.0, -2.0)
scn.set_camera(cam)
cnvs.set_background_color((0.1, 0.12, 0.15))  # Dark blue-gray background


particle_sys = ParticleSystem(simulation_config)
particle_sys.memory_allocation_and_initialization_only_position()
num_substeps = simulation_config["Configuration"]["numberOfStepsPerRenderUpdate"]

render_mesh_object = False
simulation_active = False
fluid_domain_starts = [np.array(fluid['start']) for fluid in particle_sys.fluidBlocksConfig]
fluid_domain_ends = [np.array(fluid['end']) for fluid in particle_sys.fluidBlocksConfig]
num_fluid_boxes = len(fluid_domain_starts)


object_config = particle_sys.rigidBodiesConfig.copy()
boundary_start_safe = particle_sys.domain_start + np.array([particle_sys.padding + particle_sys.particle_radius]*3)
boundary_end_safe = particle_sys.domain_end - np.array([particle_sys.padding + particle_sys.particle_radius]*3)
need_memory_reallocation = False

scene_title = 'Fluid Simulation'
series_prefix = f"{scene_title}_output/frame"
should_output_frames = False
frame_output_interval = simulation_config["Configuration"]["outputInterval"]
should_output_ply = False
frame_count = 0
ply_frame_count = 0
first_time_second_phase = True
reset_flag = False
use_rigid_object = False
prev_use_rigid_object = use_rigid_object  # Needed for checkbox change detection


def hex_to_rgb_normalized(hex_color):
    r = ((hex_color >> 16) & 0xFF) / 255.0
    g = ((hex_color >> 8) & 0xFF) / 255.0
    b = (hex_color & 0xFF) / 255.0
    return (r, g, b)


while win.running:
    if simulation_active:
        for _ in range(num_substeps):
            solver.step()

    cam.track_user_inputs(win, movement_speed=0.02, hold_key=ti.ui.RMB)

    gui = win.get_gui()
    gui.begin('SPH Controls Panel', 0.82, 0.02, 0.16, 0.96)

    gui.text('', color=hex_to_rgb_normalized(0xFFFFFF))
    gui.text("SPH Fluid Simulator", color=hex_to_rgb_normalized(0xA0C8FF))
    gui.text('----------------------------', color=hex_to_rgb_normalized(0xFFFFFF))

    if not simulation_active:
        if gui.button('Start Simulation'):
            simulation_active = True
            particle_sys.memory_allocation_and_initialization()
            solver = particle_sys.build_solver()
            solver.initialize()
            render_mesh_object = True

        if gui.button('Insert Fluid Block'):
            latest_object_id = particle_sys.cur_obj_id
            last_fluid_cfg = particle_sys.fluidBlocksConfig[-1]
            new_fluid_cfg = last_fluid_cfg.copy()
            new_fluid_cfg['objectId'] = latest_object_id + 1
            particle_sys.fluidBlocksConfig.append(new_fluid_cfg)
            fluid_domain_starts = [np.array(fluid['start']) for fluid in particle_sys.fluidBlocksConfig]
            fluid_domain_ends = [np.array(fluid['end']) for fluid in particle_sys.fluidBlocksConfig]
            num_fluid_boxes = len(fluid_domain_starts)
            need_memory_reallocation = True

        if gui.button('Remove Last Fluid Block'):
            if particle_sys.fluidBlocksConfig:
                particle_sys.fluidBlocksConfig.pop()
                fluid_domain_starts = [np.array(fluid['start']) for fluid in particle_sys.fluidBlocksConfig]
                fluid_domain_ends = [np.array(fluid['end']) for fluid in particle_sys.fluidBlocksConfig]
                num_fluid_boxes = len(fluid_domain_starts)
                need_memory_reallocation = True

        use_rigid_object = gui.checkbox('Enable Rigid Object', use_rigid_object)
        if use_rigid_object != prev_use_rigid_object:
            prev_use_rigid_object = use_rigid_object
            need_memory_reallocation = True

        gui.text('----------------------------', color=hex_to_rgb_normalized(0xFFFFFF))
        gui.text("Edit Bounding Box (Wall Offsets)", color=hex_to_rgb_normalized(0xFFD580))

        box_changed = False
        gui.text("Domain Box Controls", color=hex_to_rgb_normalized(0xA0C8FF))
        for axis in ['x', 'y', 'z']:
            min_key = f'{axis}_min'
            max_key = f'{axis}_max'
            min_val = gui.slider_float(min_key, box_bounds[min_key], 0.0, box_bounds[max_key] - 0.01)
            max_val = gui.slider_float(max_key, box_bounds[max_key], box_bounds[min_key] + 0.01, 10.0)
            if min_val != box_bounds[min_key] or max_val != box_bounds[max_key]:
                box_bounds[min_key] = min_val
                box_bounds[max_key] = max_val
                box_changed = True

        if box_changed:
            vertices_np = compute_box_vertices(box_bounds)
            for i in range(8):
                box_vertex_point[i] = vertices_np[i]

            # Update simulation_config's domainStart and domainEnd with new box_bounds
            simulation_config["Configuration"]["domainStart"] = [
                box_bounds["x_min"], box_bounds["y_min"], box_bounds["z_min"]
            ]
            simulation_config["Configuration"]["domainEnd"] = [
                box_bounds["x_max"], box_bounds["y_max"], box_bounds["z_max"]
            ]

            # Recreate ParticleSystem from updated simulation_config
            #particle_sys.free_memory_allocation()
            del particle_sys
            particle_sys = ParticleSystem(simulation_config)

            # Re-assign rigid body config if needed
            if use_rigid_object:
                particle_sys.rigidBodiesConfig = object_config
            else:
                particle_sys.rigidBodiesConfig = []

            # Refresh domain bounds from the new particle_sys
            fluid_domain_starts = [np.array(fluid['start']) for fluid in particle_sys.fluidBlocksConfig]
            fluid_domain_ends = [np.array(fluid['end']) for fluid in particle_sys.fluidBlocksConfig]
            num_fluid_boxes = len(fluid_domain_starts)

            particle_sys.memory_allocation_and_initialization_only_position()

            # Also re-compute safe zones
            boundary_start_safe = particle_sys.domain_start + np.array([particle_sys.padding + particle_sys.particle_radius] * 3)
            boundary_end_safe = particle_sys.domain_end - np.array([particle_sys.padding + particle_sys.particle_radius] * 3)





        for idx in range(num_fluid_boxes):
            gui.text('----------------------------', color=hex_to_rgb_normalized(0xFFFFFF))
            gui.text(f'Fluid Box #{idx + 1}', color=hex_to_rgb_normalized(0xFFFFFF))
            gui.text('Start Position', color=hex_to_rgb_normalized(0xFFFFFF))
            s_x = gui.slider_float(f'start_x_{idx + 1}', fluid_domain_starts[idx][0],
                                   boundary_start_safe[0], fluid_domain_ends[idx][0] - particle_sys.particle_diameter)
            s_y = gui.slider_float(f'start_y_{idx + 1}', fluid_domain_starts[idx][1],
                                   boundary_start_safe[1], fluid_domain_ends[idx][1] - particle_sys.particle_diameter)
            s_z = gui.slider_float(f'start_z_{idx + 1}', fluid_domain_starts[idx][2],
                                   boundary_start_safe[2], fluid_domain_ends[idx][2] - particle_sys.particle_diameter)
            gui.text('', color=hex_to_rgb_normalized(0xFFFFFF))
            gui.text('End Position', color=hex_to_rgb_normalized(0xFFFFFF))
            e_x = gui.slider_float(f'end_x_{idx + 1}', fluid_domain_ends[idx][0],
                                   fluid_domain_starts[idx][0] + particle_sys.particle_diameter, boundary_end_safe[0])
            e_y = gui.slider_float(f'end_y_{idx + 1}', fluid_domain_ends[idx][1],
                                   fluid_domain_starts[idx][1] + particle_sys.particle_diameter, boundary_end_safe[1])
            e_z = gui.slider_float(f'end_z_{idx + 1}', fluid_domain_ends[idx][2],
                                   fluid_domain_starts[idx][2] + particle_sys.particle_diameter, boundary_end_safe[2])
            new_start = np.array([s_x, s_y, s_z]).round(2)
            new_end = np.array([e_x, e_y, e_z]).round(2)
            if (fluid_domain_starts[idx] != new_start).any() or (fluid_domain_ends[idx] != new_end).any():
                need_memory_reallocation = True
                fluid_domain_starts[idx] = new_start
                fluid_domain_ends[idx] = new_end

        if need_memory_reallocation:
            del particle_sys
            particle_sys = ParticleSystem(simulation_config)
            if use_rigid_object:
                particle_sys.rigidBodiesConfig = object_config
            else:
                particle_sys.rigidBodiesConfig = []
            for idx in range(num_fluid_boxes):
                particle_sys.fluidBlocksConfig[idx]['start'] = fluid_domain_starts[idx]
                particle_sys.fluidBlocksConfig[idx]['end'] = fluid_domain_ends[idx]
            particle_sys.memory_allocation_and_initialization_only_position()
            need_memory_reallocation = False

        gui.text('----------------------------', color=hex_to_rgb_normalized(0xFFFFFF))
        gui.text('Fluid Particles Count', color=hex_to_rgb_normalized(0xFFFFFF))
        gui.text(f'{particle_sys.total_fluid_particle_num}', color=hex_to_rgb_normalized(0xFFFFFF))
        gui.text('Rigid Particles Count', color=hex_to_rgb_normalized(0xFFFFFF))
        gui.text(f'{particle_sys.total_rigid_particle_num}', color=hex_to_rgb_normalized(0xFFFFFF))
        gui.text('Total Particles Count', color=hex_to_rgb_normalized(0xFFFFFF))
        gui.text(f'{particle_sys.total_particle_num}', color=hex_to_rgb_normalized(0xFFFFFF))

        gui.text('----------------------------', color=hex_to_rgb_normalized(0xFFFFFF))
        should_output_frames = gui.checkbox('Save Images', should_output_frames)
        should_output_ply = gui.checkbox('Save [.ply] Mesh Files', should_output_ply)

        gui.end()

    else:
        if gui.button('Restart Simulation'):
            particle_sys.reset_particle_system()
            reset_flag = True

        if gui.button('Reset Camera View'):
            cam.position(6.5, 3.5, 5)
            cam.lookat(-1, -1.5, -3)

        render_mesh_object = gui.checkbox('Show Mesh Object', render_mesh_object)

        gui.text('----------------------------', color=hex_to_rgb_normalized(0xFFFFFF))
        gui.text('Euler Step Interval', color=hex_to_rgb_normalized(0xFFFFFF))
        solver.dt[None] = gui.slider_float('Step time [ms]', solver.dt[None] * 1000, 0.2, 0.8) * 0.001
        gui.text('Viscosity Coefficient', color=hex_to_rgb_normalized(0xFFFFFF))
        solver.viscosity[None] = gui.slider_float('', solver.viscosity[None], 0.001, 0.5)
        gui.text('Surface Tension Coefficient', color=hex_to_rgb_normalized(0xFFFFFF))
        solver.surface_tension[None] = gui.slider_float('[N/m]', solver.surface_tension[None], 0.001, 5)

        if solver.viscosity[None] > 0.23 or solver.surface_tension[None] > 2.0:
            solver.dt[None] = ti.min(solver.dt[None], 0.0005)
        if solver.viscosity[None] > 0.23 and solver.surface_tension[None] > 2.0:
            solver.dt[None] = ti.min(solver.dt[None], 0.0004)

        gui.text('----------------------------', color=hex_to_rgb_normalized(0xFFFFFF))
        gui.text('Fluid Particles Count', color=hex_to_rgb_normalized(0xFFFFFF))
        gui.text(f'{particle_sys.total_fluid_particle_num}', color=hex_to_rgb_normalized(0xFFFFFF))
        gui.text('Rigid Particles Count', color=hex_to_rgb_normalized(0xFFFFFF))
        gui.text(f'{particle_sys.total_rigid_particle_num}', color=hex_to_rgb_normalized(0xFFFFFF))
        gui.text('Total Particles Count', color=hex_to_rgb_normalized(0xFFFFFF))
        gui.text(f'{particle_sys.total_particle_num}', color=hex_to_rgb_normalized(0xFFFFFF))

        gui.end()

    scn.set_camera(cam)
    scn.point_light((2, 2, 2), color=(1, 1, 1))
    scn.lines(box_vertex_point, width=3.5, indices=box_edge_index, color=(1, 1, 1))

    if render_mesh_object:
        particle_sys.update_fluid_position_info()
        particle_sys.update_fluid_color_info()
        scn.particles(particle_sys.fluid_only_position, radius=particle_sys.particle_radius, per_vertex_color=particle_sys.fluid_only_color)
        for i in range(len(particle_sys.mesh_vertices)):
            scn.mesh(particle_sys.mesh_vertices[i], particle_sys.mesh_indices[i])
    else:
        scn.particles(particle_sys.position, radius=particle_sys.particle_radius, per_vertex_color=particle_sys.color)

    cnvs.scene(scn)

    if simulation_active:
        if reset_flag:
            frame_count = 0
            ply_frame_count = 0
            reset_flag = False

        if first_time_second_phase:
            if should_output_frames:
                os.makedirs(f"{scene_title}_output_img", exist_ok=True)
            if should_output_ply:
                os.makedirs(f"{scene_title}_output", exist_ok=True)
            first_time_second_phase = False

        if frame_count % frame_output_interval == 0:
            if should_output_ply:
                particle_sys.update_fluid_position_info()
                np_positions = particle_sys.dump()
                writer = ti.tools.PLYWriter(num_vertices=particle_sys.total_fluid_particle_num)
                writer.add_vertex_pos(np_positions[:, 0], np_positions[:, 1], np_positions[:, 2])
                writer.export_frame_ascii(ply_frame_count, series_prefix.format(0))
                for rigid_id in particle_sys.rigid_object_id:
                    with open(f"{scene_title}output/obj{rigid_id}_{ply_frame_count:06}.obj", "w") as f:
                        exported_mesh = particle_sys.object_collection[rigid_id]["mesh"].export(file_type='obj')
                        f.write(exported_mesh)
                ply_frame_count += 1
            if should_output_frames:
                win.save_image(f"{scene_title}_output_img/{frame_count:06}.png")
        frame_count += 1

    win.show()