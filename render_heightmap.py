import bpy
import os

# === 🛠 CONFIG ===
HEIGHTMAP_PATH = "/Users/valentinweyer/Dropbox/Valentin/Projekte/AI/ML_based_erosion/erosion_snapshots/erosion_00000.png"
OUTPUT_IMAGE = "/Users/valentinweyer/Dropbox/Valentin/Projekte/AI/ML_based_erosion/blender_render.png"
DISPLACE_STRENGTH = 50
PLANE_SUBDIVISIONS = 512
RENDER_RESOLUTION = 1024

# === 🧼 Reset scene ===
bpy.ops.wm.read_factory_settings(use_empty=True)

# === 🟫 Create plane and subdivide ===
bpy.ops.mesh.primitive_plane_add(size=2)
plane = bpy.context.active_object
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.subdivide(number_cuts=PLANE_SUBDIVISIONS)
bpy.ops.object.mode_set(mode='OBJECT')

# === 📈 Displace with heightmap ===
bpy.ops.object.modifier_add(type='DISPLACE')
disp_mod = plane.modifiers["Displace"]
disp_mod.strength = DISPLACE_STRENGTH
disp_tex = bpy.data.textures.new("HeightTex", type='IMAGE')
disp_tex.image = bpy.data.images.load(HEIGHTMAP_PATH)
disp_mod.texture = disp_tex
disp_mod.texture_coords = 'UV'

# === 🧶 Add UVs ===
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.uv.smart_project()
bpy.ops.object.mode_set(mode='OBJECT')

# === 🎨 Create slope-based material ===
mat = bpy.data.materials.new(name="TerrainMaterial")
mat.use_nodes = True
nodes = mat.node_tree.nodes
links = mat.node_tree.links

nodes.clear()

# --- Nodes ---
tex_coord = nodes.new("ShaderNodeTexCoord")
separate = nodes.new("ShaderNodeSeparateXYZ")
gradient = nodes.new("ShaderNodeVectorMath")
gradient.operation = "LENGTH"
colorramp = nodes.new("ShaderNodeValToRGB")
diffuse = nodes.new("ShaderNodeBsdfDiffuse")
output = nodes.new("ShaderNodeOutputMaterial")

# --- Connect ---
links.new(tex_coord.outputs["Normal"], separate.inputs["Vector"])
links.new(separate.outputs["Z"], gradient.inputs[0])
links.new(gradient.outputs[0], colorramp.inputs["Fac"])
links.new(colorramp.outputs["Color"], diffuse.inputs["Color"])
links.new(diffuse.outputs["BSDF"], output.inputs["Surface"])

# --- Colors (Lague-inspired) ---
ramp = colorramp.color_ramp
ramp.elements[0].position = 0.0
ramp.elements[0].color = (0.4, 0.6, 0.3, 1)  # Grass
ramp.elements[1].position = 0.6
ramp.elements[1].color = (0.5, 0.4, 0.2, 1)  # Dirt
ramp.elements.new(1.0).color = (1.0, 1.0, 1.0, 1)  # Snow

plane.data.materials.append(mat)

# === 🎥 Camera ===
bpy.ops.object.camera_add(location=(2, -3, 2))
cam = bpy.context.active_object
cam.rotation_euler = (1.1, 0, 0.9)
bpy.context.scene.camera = cam

# === 💡 Light ===
bpy.ops.object.light_add(type='SUN', location=(10, 10, 10))

# === 🖼 Render settings ===
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.device = 'CPU'
scene.cycles.samples = 64
scene.render.resolution_x = RENDER_RESOLUTION
scene.render.resolution_y = RENDER_RESOLUTION
scene.render.filepath = OUTPUT_IMAGE

# === 📸 Render! ===
bpy.ops.render.render(write_still=True)