import numpy as np
from noise import pnoise2
from pyvistaqt import BackgroundPlotter
from PyQt5.QtCore import QThread, pyqtSignal, QObject
from scipy.ndimage import gaussian_filter
import pyvista as pv
from Lague_erosion_port import erode
from PIL import Image
import os

SNAPSHOT_DIR = "erosion_snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

def save_snapshot(hm, step):
    if hm.ndim == 1:
        hm = hm.reshape((size, size))  # Adjust if needed for dynamic size
    img = ((hm - hm.min()) / (hm.max() - hm.min()) * 255).astype(np.uint8)
    im = Image.fromarray(img, mode='L')  # Grayscale mode
    im.save(f"{SNAPSHOT_DIR}/erosion_{step:05}.png")

# --- Config ---
size = 2048
scale = 0.01
elevation_scale = 80
octaves = 6
persistence = 0.5
lacunarity = 2.0

# --- Generate Perlin heightmap ---
heightmap = np.zeros((size, size), dtype=np.float32)
for i in range(size):
    for j in range(size):
        heightmap[i][j] = pnoise2(i * scale, j * scale,
                                  octaves=octaves,
                                  persistence=persistence,
                                  lacunarity=lacunarity,
                                  repeatx=1024, repeaty=1024,
                                  base=42)

heightmap -= heightmap.min()
heightmap /= heightmap.max()

from matplotlib.colors import LinearSegmentedColormap

# --- Setup PyVista plot ---
x, y = np.meshgrid(np.arange(size), np.arange(size))
z = heightmap * elevation_scale

# Base mesh (used for updating during erosion)
base_mesh = pv.StructuredGrid()
base_mesh.points = np.c_[x.ravel(), y.ravel(), z.ravel()]
base_mesh.dimensions = size, size, 1

# Display mesh (smoothed for visualization only)
display_mesh = base_mesh.extract_surface().triangulate().subdivide(2, subfilter='loop')

# Earthy color map (fallback if needed)
custom_cmap = ["#3b2f2f", "#775533", "#c5b358", "#c5e87c"]

# Slope-based colormap
slope_cmap = LinearSegmentedColormap.from_list(
    "slope_map",
    [
        (0.0, "#c5e87c"),   # green (flat)
        (0.5, "#775533"),   # brown (moderate)
        (1.0, "#3b2f2f")    # rock (steep)
    ]
)
plotter = BackgroundPlotter()
plotter.add_mesh(
    display_mesh,
    cmap=custom_cmap,
    show_edges=False,
    smooth_shading=True,
    lighting=True,
    scalar_bar_args={"title": "Elevation"}
)
plotter.enable_eye_dome_lighting()
plotter.set_background("beige")
# plotter.enable_shadows()       # optional, GPU intensive
plotter.show_bounds()
plotter.view_xy()


def assign_lague_colors(hm, elevation_scale):
    if hm.ndim == 1:
        hm = hm.reshape((size, size))  # or use a global/map_size
    slope = compute_slope(hm)
    height_norm = (hm - hm.min()) / (hm.max() - hm.min())

    snow = np.array([1.0, 1.0, 1.0])      # snow
    rock = np.array([0.3, 0.3, 0.3])      # rock
    dirt = np.array([0.6, 0.5, 0.3])      # dirt
    grass = np.array([0.6, 0.8, 0.4])     # grass

    # Create empty RGB array
    color_map = np.zeros((hm.shape[0], hm.shape[1], 3), dtype=np.float32)

    # Build masks based on slope + height
    snow_mask = (height_norm > 0.8) & (slope < 0.03)
    rock_mask = slope > 0.05
    grass_mask = (slope < 0.015) & (height_norm < 0.6)

    # Default to dirt
    color_map[:, :] = dirt

    # Apply overrides
    color_map[snow_mask] = snow
    color_map[rock_mask] = rock
    color_map[grass_mask] = grass

    return color_map.reshape((-1, 3))


# --- Qt Erosion Worker Thread ---
class ErosionWorker(QObject):
    finished = pyqtSignal()
    update_mesh_signal = pyqtSignal(np.ndarray)

    def run(self):
        global heightmap

        def update_mesh_live(hm, step=None):
            if step is not None:
                print(f"📸 Saving snapshot at step {step}")
                save_snapshot(hm, step)
            self.update_mesh_signal.emit(hm.copy())

        try:
            print("Before erosion min/max:", heightmap.min(), heightmap.max())

            heightmap = erode(
                heightmap,
                map_size=heightmap.shape[0],
                iterations=250000,
                radius=3,
                inertia=0.05,
                sediment_capacity_factor=4,
                min_sediment_capacity=0.01,
                erode_speed=0.3,
                deposit_speed=0.3,
                evaporate_speed=0.01,
                gravity=4,
                max_lifetime=30,
                live_callback=update_mesh_live
            )

            heightmap = gaussian_filter(heightmap, sigma=0.8)
            update_mesh_live(heightmap)

            print("After erosion min/max:", heightmap.min(), heightmap.max())
            heightmap = np.nan_to_num(heightmap, nan=0.0)
            heightmap = np.clip(heightmap, 0.0, 1.0)
            update_mesh_live(heightmap)
            # Rebuild and show updated display mesh
            updated_display = base_mesh.extract_surface().triangulate().subdivide(2, subfilter='loop')

            plotter.clear()
            plotter.add_mesh(
                updated_display,
                cmap=custom_cmap,
                show_edges=False,
                smooth_shading=True,
                lighting=True,
                scalar_bar_args={"title": "Elevation"}
            )
            plotter.enable_eye_dome_lighting()
            plotter.set_background("beige")
            plotter.show_bounds()
            plotter.view_xy()
            plotter.render()

            self.finished.emit()

        except Exception as e:
            import traceback
            print("🔥 Erosion thread crashed:")
            traceback.print_exc()


# --- Run in thread ---
thread = QThread()
worker = ErosionWorker()
worker.moveToThread(thread)
def compute_slope(heightmap, scale=1.0):
    if heightmap.ndim == 1:
        heightmap = heightmap.reshape((size, size))
    gx, gy = np.gradient(heightmap, scale)
    return np.sqrt(gx**2 + gy**2)

def apply_mesh_update(hm):
    print("✅ Applying mesh update to base_mesh")
    z = hm * elevation_scale

    base_mesh.points[:, 2] = z.ravel()
    base_mesh.GetPoints().Modified()
    base_mesh.Modified()

    poly = base_mesh.extract_surface().triangulate()
    poly.compute_normals(inplace=True)

    # Assign colors based on height and slope (Lague style)
    vertex_colors = assign_lague_colors(hm, elevation_scale)
    poly.point_data['colors'] = vertex_colors

    camera = plotter.camera_position
    plotter.clear()
    plotter.add_mesh(
        poly,
        scalars='colors',
        rgb=True,
        show_edges=False,
        smooth_shading=True,
        lighting=True,
        scalar_bar_args={"title": "Lague Terrain"}
    )
    plotter.camera_position = camera
    plotter.render()

worker.update_mesh_signal.connect(apply_mesh_update)
thread.started.connect(worker.run)
worker.finished.connect(thread.quit)

thread.start()

# --- Keep GUI running ---
plotter.show()
plotter.app.exec_()




import imageio.v3 as iio
import os

frames = []
files = sorted(os.listdir("erosion_snapshots"))
for f in files:
    if f.endswith(".png"):
        frames.append(iio.imread(os.path.join("erosion_snapshots", f)))

iio.imwrite("erosion_timelapse.gif", frames, duration=10)  # ms per frame