import time
import numpy as np

import viser

def slider_changed(event):
    print("Slider changed:", event.target.value)

def main():
    server = viser.ViserServer(host="0.0.0.0")

    # Generate a spiral point cloud.
    num_points = 200
    t = np.linspace(0, 10, num_points)
    spiral_positions = np.column_stack(
        [
            np.sin(t) * (1 + t / 10),
            np.cos(t) * (1 + t / 10),
            t / 5,
        ]
    )
    print(f"t shape: {t.shape}")
    print(f"spiral_positions shape: {spiral_positions.shape}")
    # Create colors based on height (z-coordinate).
    z_min, z_max = spiral_positions[:, 2].min(), spiral_positions[:, 2].max()
    normalized_z = (spiral_positions[:, 2] - z_min) / (z_max - z_min)

    # Color gradient from blue (bottom) to red (top).
    colors = np.zeros((num_points, 3), dtype=np.uint8)
    colors[:, 0] = (normalized_z * 255).astype(np.uint8)  # Red channel.
    colors[:, 2] = ((1 - normalized_z) * 255).astype(np.uint8)  # Blue channel.
    print(f"shape of the colors: {colors.shape} ")

    # Add the point cloud to the scene.
    server.scene.add_point_cloud(
        name="/spiral_cloud",
        points=spiral_positions,
        colors=colors,
        point_size=0.05,
    )

    print("Open your browser to http://localhost:8080")
    print("Press Ctrl+C to exit")

    with server.gui.add_folder("Read-only"):
        gui_slider = server.gui.add_slider(
            "Timestep_slider",
            min=0,
            max=23,
            step=1,
            initial_value=0,
            disabled=False,
        )
    gui_slider.on_update(slider_changed)
    while True:
        time.sleep(10.0)


if __name__ == "__main__":
    main()


