from contextlib import contextmanager
from typing import Literal
import pyvista as pv
from stpyvista import stpyvista

Axis = Literal["X","Y","Z"]

@contextmanager
def themed_plotter(height: int = 600):
    # Control canvas size via PyVista, not stpyvista kwargs
    p = pv.Plotter(window_size=[900, height])
    p.background_color = "white"
    try:
        yield p
    finally:
        p.close()

def _is_empty_dataset(ds) -> bool:
    # Compatible across PyVista versions: property, method, or fallback to n_points
    try:
        attr = getattr(ds, "is_empty", None)
        if callable(attr):
            return bool(attr())
        if isinstance(attr, bool):
            return attr
    except Exception:
        pass
    try:
        return getattr(ds, "n_points", 0) == 0
    except Exception:
        return True

def show_stl_with_fill(
    stl_path: str,
    fill_fraction: float = 0.0,
    axis: Axis = "Z",
    foam_color: str = "#FF9900",
    mold_color: str = "#AAAAAA",
    mold_opacity: float = 0.3,
    auto_rotate: bool = False,
    height: int = 600,
    key: str = "pv",
):
    mesh = pv.read(stl_path)

    # Clamp and compute plane location along chosen axis
    t = max(0.0, min(1.0, float(fill_fraction)))
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    if axis == "Z":
        z_cut = zmin + t * (zmax - zmin)
        foam_part = mesh.clip(normal=(0, 0, 1), origin=(0, 0, z_cut), invert=True)
    elif axis == "Y":
        y_cut = ymin + t * (ymax - ymin)
        foam_part = mesh.clip(normal=(0, 1, 0), origin=(0, y_cut, 0), invert=True)
    else:
        x_cut = xmin + t * (xmax - xmin)
        foam_part = mesh.clip(normal=(1, 0, 0), origin=(x_cut, 0, 0), invert=True)

    with themed_plotter(height=height) as p:
        # Base mesh (mold shell)
        p.add_mesh(mesh, color=mold_color, opacity=mold_opacity, lighting=True, smooth_shading=True)
        # Filled region
        if not _is_empty_dataset(foam_part):
            p.add_mesh(foam_part, color=foam_color, opacity=0.95, lighting=True, smooth_shading=True)

        p.view_isometric()
        if auto_rotate:
            try:
                p.camera.azimuth(10)
            except Exception:
                pass
        # Embed existing plotter; no height kwarg here
        stpyvista(p, key=key)
