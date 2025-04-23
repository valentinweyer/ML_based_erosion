import numpy as np
from noise import pnoise2
from pyvistaqt import BackgroundPlotter
from PyQt5.QtCore import QThread, pyqtSignal, QObject
from scipy.ndimage import gaussian_filter
import pyvista as pv

# --- Config ---
size = 1024
scale = 0.01
elevation_scale = 200  # Was 100 — now a bit softer, still dramatic
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

# --- Setup PyVista plot ---
x, y = np.meshgrid(np.arange(size), np.arange(size))
z = heightmap * elevation_scale
terrain_mesh = pv.StructuredGrid()
terrain_mesh.points = np.c_[x.ravel(), y.ravel(), z.ravel()]
terrain_mesh.dimensions = size, size, 1

plotter = BackgroundPlotter()
plotter.add_mesh(terrain_mesh, cmap='terrain', show_edges=False)
plotter.show_bounds()
plotter.view_xy()

# --- Erosion Function ---
def realistic_erosion(
    heightmap,
    droplets=50000,
    erosion_steps=30,
    inertia=0.05,
    capacity_constant=4.0,
    min_slope=0.01,
    deposit_speed=0.3,
    erode_speed=0.3,
    evaporate_speed=0.01,
    blur_every=1000,
    blur_sigma=0.5,
    live_callback=None
):
    size = heightmap.shape[0]
    hm = heightmap.copy()
    flow = np.zeros_like(hm, dtype=np.float32)

    for d in range(droplets):
        x = np.random.rand() * (size - 1)
        y = np.random.rand() * (size - 1)
        dir = np.array([0.0, 0.0])
        speed = 1.0
        water = 1.0
        sediment = 0.0

        for _ in range(erosion_steps):
            if not (1 <= x < size - 2 and 1 <= y < size - 2):
                break

            cx, cy = int(x), int(y)
            h = hm[cy, cx]

            grad_x = (hm[cy, cx + 1] - hm[cy, cx - 1]) * 0.5
            grad_y = (hm[cy + 1, cx] - hm[cy - 1, cx]) * 0.5
            gradient = np.array([grad_x, grad_y])

            dir = dir * inertia - gradient * (1 - inertia)
            norm = np.linalg.norm(dir)
            if norm == 0 or np.isnan(norm):
                break
            dir /= norm

            x += dir[0]
            y += dir[1]

            if not (1 <= x < size - 2 and 1 <= y < size - 2):
                break

            nx, ny = int(x), int(y)
            new_h = hm[ny, nx]
            delta_h = new_h - h
            slope = -delta_h

            capacity = max(slope * speed * water * capacity_constant, min_slope)

            if sediment > capacity:
                deposit = (sediment - capacity) * deposit_speed * (1 - slope)
                hm[ny, nx] += deposit
                sediment -= deposit
            else:
                erode_amount = min((capacity - sediment) * erode_speed, hm[ny, nx])
                hm[ny, nx] -= erode_amount
                sediment += erode_amount

            speed = np.sqrt(speed ** 2 + delta_h * 0.1)
            water *= (1 - evaporate_speed)
            flow[ny, nx] += water

            if water < 0.01:
                break

        if blur_every > 0 and d % blur_every == 0 and d != 0:
            hm = gaussian_filter(hm, sigma=blur_sigma)

        if live_callback and d % 500 == 0:
            live_callback(hm)

    return np.clip(hm, 0.0, 1.0), flow

# --- Qt Erosion Worker Thread ---
class ErosionWorker(QObject):
    finished = pyqtSignal()

    def run(self):
        global heightmap
        def update_mesh_live(hm):
            z = hm * elevation_scale
            terrain_mesh.points[:, 2] = z.ravel()
            terrain_mesh.Modified()

        heightmap, _ = realistic_erosion(
            heightmap,
            droplets=50000,
            erosion_steps=30,
            inertia=0.3,                  # smoother directional flow
            capacity_constant=8.0,        # more sediment capacity
            min_slope=0.01,               # don't erode flats too aggressively
            deposit_speed=0.2,            # more subtle deposits
            erode_speed=0.4,              # faster erosion in steeper spots
            evaporate_speed=0.01,         # as in Lague's code
            blur_every=1000,              # helps reduce spikes
            blur_sigma=0.7,               # visually smooth, but still sharp
            live_callback=update_mesh_live
        )
        self.finished.emit()

# --- Run in thread ---
thread = QThread()
worker = ErosionWorker()
worker.moveToThread(thread)
thread.started.connect(worker.run)
worker.finished.connect(thread.quit)

thread.start()

# --- Keep GUI running ---
plotter.show()
plotter.app.exec_()