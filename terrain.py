import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter



# Generate simple Perlin-style terrain
def generate_terrain(size=512, scale=20):
    base = np.random.rand(size, size)
    terrain = gaussian_filter(base, sigma=scale)
    return terrain

terrain = generate_terrain()
plt.imshow(terrain, cmap='terrain')
plt.title("Base Terrain")
plt.colorbar()
plt.show()

from erosion.simulator import erode

# Input: terrain as a NumPy array
eroded_terrain = erode(terrain, iterations=50)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(terrain, cmap='terrain')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title("Eroded")
plt.imshow(eroded_terrain, cmap='terrain')
plt.colorbar()
plt.show()