# This is a Python port of Sebastian Lague's erosion simulation
# Includes: erosion brush, bilinear interpolation, weighted erosion and deposition

import numpy as np

class HeightAndGradient:
    def __init__(self, height, grad_x, grad_y):
        self.height = height
        self.gradient_x = grad_x
        self.gradient_y = grad_y


def calculate_height_and_gradient(heightmap, map_size, pos_x, pos_y):
    x = int(pos_x)
    y = int(pos_y)
    cell_x = pos_x - x
    cell_y = pos_y - y

    idx = y * map_size + x
    height_nw = heightmap[idx]
    height_ne = heightmap[idx + 1]
    height_sw = heightmap[idx + map_size]
    height_se = heightmap[idx + map_size + 1]

    grad_x = (height_ne - height_nw) * (1 - cell_y) + (height_se - height_sw) * cell_y
    grad_y = (height_sw - height_nw) * (1 - cell_x) + (height_se - height_ne) * cell_x

    h = height_nw * (1 - cell_x) * (1 - cell_y) + \
        height_ne * cell_x * (1 - cell_y) + \
        height_sw * (1 - cell_x) * cell_y + \
        height_se * cell_x * cell_y

    return HeightAndGradient(h, grad_x, grad_y)


def initialize_brush(map_size, radius):
    brush_indices = [[] for _ in range(map_size * map_size)]
    brush_weights = [[] for _ in range(map_size * map_size)]

    for y in range(map_size):
        for x in range(map_size):
            weights = []
            indices = []
            weight_sum = 0
            for oy in range(-radius, radius + 1):
                for ox in range(-radius, radius + 1):
                    sx = x + ox
                    sy = y + oy
                    sqr_dist = ox * ox + oy * oy
                    if 0 <= sx < map_size and 0 <= sy < map_size and sqr_dist < radius * radius:
                        weight = 1 - np.sqrt(sqr_dist) / radius
                        indices.append(sy * map_size + sx)
                        weights.append(weight)
                        weight_sum += weight

            norm_weights = [w / weight_sum for w in weights]
            i = y * map_size + x
            brush_indices[i] = indices
            brush_weights[i] = norm_weights

    return brush_indices, brush_weights


def erode(
    heightmap,
    map_size,
    iterations=50000,
    radius=3,
    inertia=0.05,
    sediment_capacity_factor=4,
    min_sediment_capacity=0.01,
    erode_speed=0.3,
    deposit_speed=0.3,
    evaporate_speed=0.01,
    gravity=4,
    max_lifetime=30,
    initial_water_volume=1,
    initial_speed=1,
    live_callback=None
):
    brush_indices, brush_weights = initialize_brush(map_size, radius)
    heightmap = heightmap.ravel()  # use flat array like in Unity
    rng = np.random.default_rng()

    for d in range(iterations):
        pos_x = rng.uniform(0, map_size - 1)
        pos_y = rng.uniform(0, map_size - 1)
        dir_x, dir_y = 0.0, 0.0
        speed = initial_speed
        water = initial_water_volume
        sediment = 0.0

        for lifetime in range(max_lifetime):
            node_x = int(pos_x)
            node_y = int(pos_y)
            if node_x < 0 or node_x >= map_size - 1 or node_y < 0 or node_y >= map_size - 1:
                break

            droplet_index = node_y * map_size + node_x
            offset_x = pos_x - node_x
            offset_y = pos_y - node_y

            h_and_g = calculate_height_and_gradient(heightmap, map_size, pos_x, pos_y)

            dir_x = dir_x * inertia - h_and_g.gradient_x * (1 - inertia)
            dir_y = dir_y * inertia - h_and_g.gradient_y * (1 - inertia)
            norm = np.hypot(dir_x, dir_y)
            if norm == 0 or np.isnan(norm):
                break
            dir_x /= norm
            dir_y /= norm

            pos_x += dir_x
            pos_y += dir_y

            if np.isnan(pos_x) or np.isnan(pos_y):
                break

            if pos_x < 0 or pos_x >= map_size - 1 or pos_y < 0 or pos_y >= map_size - 1:
                break

            new_h = calculate_height_and_gradient(heightmap, map_size, pos_x, pos_y).height
            delta_h = new_h - h_and_g.height
            sediment_capacity = max(-delta_h * speed * water * sediment_capacity_factor, min_sediment_capacity)

            if sediment > sediment_capacity or delta_h > 0:
                amount_to_deposit = delta_h if delta_h > 0 else (sediment - sediment_capacity) * deposit_speed
                sediment -= amount_to_deposit
                deposit = np.clip(amount_to_deposit, 0, 0.01)
                heightmap[droplet_index] += deposit * (1 - offset_x) * (1 - offset_y)
                heightmap[droplet_index + 1] += deposit * offset_x * (1 - offset_y)
                heightmap[droplet_index + map_size] += deposit * (1 - offset_x) * offset_y
                heightmap[droplet_index + map_size + 1] += deposit * offset_x * offset_y
            else:
                amount_to_erode = min((sediment_capacity - sediment) * erode_speed, -delta_h)
                for i, weight in zip(brush_indices[droplet_index], brush_weights[droplet_index]):
                    delta_sediment = min(heightmap[i], np.clip(amount_to_erode * weight, 0, 0.01))
                    heightmap[i] -= delta_sediment
                    sediment += delta_sediment

            speed = np.sqrt(max(0.0, speed ** 2 + delta_h * gravity))
            water *= (1 - evaporate_speed)

            if water < 0.01:
                break

        heightmap = np.clip(heightmap, 0.0, 1.0)

        if live_callback and d % 1000 == 0:
            live_callback(heightmap, d)

    heightmap = np.nan_to_num(heightmap, nan=0.0)
    heightmap = np.clip(heightmap, 0.0, 1.0)
    return heightmap.reshape((map_size, map_size))
