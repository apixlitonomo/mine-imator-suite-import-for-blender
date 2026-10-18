"""Blender-independent geometry and bend math for Mine-imator .mimodel parts.

Everything here is plain Python (no bpy / mathutils) so it can be unit-tested
outside Blender.  blender_importer.py supplies the thin bpy glue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence
import math

Vec3 = tuple[float, float, float]
Mat3 = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]

EPS = 1e-6
DEFAULT_BEND_SIZE = 4.0   # Mine-imator units, used when a bend has no "size"
MAX_BEND_SEGMENTS = 16


# --------------------------------------------------------------------------
# Plane geometry
# --------------------------------------------------------------------------

def _vec3(value: Any, default: float = 0.0) -> list[float]:
    values = list(value) if isinstance(value, (list, tuple)) else []
    values = [float(v) for v in values[:3]]
    while len(values) < 3:
        values.append(default)
    return values


# Cyclic permutations keep handedness, so face winding / normals stay valid.
# canonical (u, v, n) -> model axis indices, keyed by the collapsed (normal) axis.
_AXIS_MAP = {
    2: (0, 1, 2),  # normal Z: spans X,Y   (legacy behaviour, unchanged)
    0: (1, 2, 0),  # normal X: spans Y,Z
    1: (2, 0, 1),  # normal Y: spans Z,X
}


def plane_normal_axis(start: Sequence[float], end: Sequence[float]) -> int:
    """Return the model axis (0=X,1=Y,2=Z) a plane is flat along.

    The collapsed axis (from == to) is the normal.  If several are collapsed,
    prefer Z (legacy), then the axis with the smallest extent.  If none are
    collapsed the plane is treated as lying on Z, like before.
    """
    extents = [abs(float(end[i]) - float(start[i])) for i in range(3)]
    collapsed = [i for i in range(3) if extents[i] < EPS]
    if not collapsed:
        return 2
    if len(collapsed) == 1:
        return collapsed[0]
    # Degenerate (two or more collapsed): keep the axis with the most extent
    # out of the plane so something visible remains.
    return 2 if 2 in collapsed else collapsed[0]


def plane_geometry(
    shape: dict[str, Any],
    texture_size: tuple[float, float],
    to_blender: Callable[[Sequence[float]], Vec3],
) -> tuple[list[Vec3], list[tuple[int, ...]], list[tuple[float, float]], list[str]]:
    """Build a Mine-imator plane on whichever axis it is flat along.

    Returns (vertices, faces, loop_uvs, warnings).  loop_uvs has one entry per
    face corner in face order.
    """
    warnings: list[str] = []
    start = _vec3(shape.get("from"), -8.0)
    end = _vec3(shape.get("to"), 8.0)
    if shape.get("from") is None:
        start[2] = 0.0
    if shape.get("to") is None:
        end[2] = 0.0

    n_axis = plane_normal_axis(start, end)
    ax_u, ax_v, ax_n = _AXIS_MAP[n_axis]

    # Canonical frame: u/v span the plane, n is the normal.
    u0, u1 = sorted((start[ax_u], end[ax_u]))
    v0, v1 = sorted((start[ax_v], end[ax_v]))
    n_a, n_b = start[ax_n], end[ax_n]
    n_mid = (n_a + n_b) * 0.5
    width, height = u1 - u0, v1 - v0

    if width < EPS or height < EPS:
        warnings.append("plane has zero area on its spanning axes; widened to 1 unit")
        if width < EPS:
            u0, u1, width = u0 - 0.5, u1 + 0.5, 1.0
        if height < EPS:
            v0, v1, height = v0 - 0.5, v1 + 0.5, 1.0

    def to_mi(u: float, v: float, n: float) -> Vec3:
        out = [0.0, 0.0, 0.0]
        out[ax_u], out[ax_v], out[ax_n] = u, v, n
        return to_blender(out)

    uv_origin = _vec3(shape.get("uv"))[:2]
    ou, ov = uv_origin
    # model_file_load_part: texture_size is made square (max of both axes)
    tex_w = tex_h = max(float(texture_size[0]), float(texture_size[1]))
    invert = bool(shape.get("invert", False))
    is_3d = bool(shape.get("3d", False))

    thickness = abs(n_b - n_a)
    if is_3d and thickness < EPS:
        thickness = 1.0

    def px(x: float, y: float) -> tuple[float, float]:
        return (x / tex_w, 1.0 - y / tex_h)

    if is_3d:
        half = thickness * 0.5
        lo = (u0, v0, n_mid - half)
        hi = (u1, v1, n_mid + half)
        corners = [
            (lo[0], lo[1], lo[2]), (hi[0], lo[1], lo[2]), (hi[0], hi[1], lo[2]), (lo[0], hi[1], lo[2]),
            (lo[0], lo[1], hi[2]), (hi[0], lo[1], hi[2]), (hi[0], hi[1], hi[2]), (lo[0], hi[1], hi[2]),
        ]
        vertices = [to_mi(*c) for c in corners]
        faces: list[tuple[int, ...]] = [
            (0, 3, 2, 1), (5, 6, 7, 4), (4, 7, 3, 0), (1, 2, 6, 5), (3, 7, 6, 2), (4, 0, 1, 5),
        ]
        d = thickness
        pts = {
            "front": [(ou, ov), (ou + width, ov), (ou + width, ov + height), (ou, ov + height)],
            "back": [(ou + width, ov), (ou, ov), (ou, ov + height), (ou + width, ov + height)],
            "east": [(ou + width, ov), (ou + width + d, ov), (ou + width + d, ov + height), (ou + width, ov + height)],
            "west": [(ou - d, ov), (ou, ov), (ou, ov + height), (ou - d, ov + height)],
            "up": [(ou, ov - d), (ou + width, ov - d), (ou + width, ov), (ou, ov)],
            "down": [(ou + width, ov), (ou + width * 2, ov), (ou + width * 2, ov - d), (ou + width, ov - d)],
        }
        loops = [
            [pts["front"][2], pts["front"][1], pts["front"][0], pts["front"][3]],
            [pts["back"][2], pts["back"][1], pts["back"][0], pts["back"][3]],
            [pts["west"][2], pts["west"][1], pts["west"][0], pts["west"][3]],
            [pts["east"][2], pts["east"][1], pts["east"][0], pts["east"][3]],
            [pts["up"][0], pts["up"][3], pts["up"][2], pts["up"][1]],
            [pts["down"][3], pts["down"][0], pts["down"][1], pts["down"][2]],
        ]
        uvs = [px(x, y) for face in loops for x, y in face]
        if invert:
            faces = [tuple(reversed(f)) for f in faces]
            reordered: list[tuple[float, float]] = []
            for i in range(0, len(uvs), 4):
                reordered.extend(reversed(uvs[i:i + 4]))
            uvs = reordered
        return vertices, faces, uvs, warnings

    # Flat plane.  Front and back get their own vertices: sharing four verts
    # between two opposed faces averages their normals to zero (broken
    # shading) and made the old back-face UVs land on the wrong corners.
    front = [to_mi(u0, v0, n_mid), to_mi(u1, v0, n_mid), to_mi(u1, v1, n_mid), to_mi(u0, v1, n_mid)]
    quad = [px(ou, ov + height), px(ou + width, ov + height), px(ou + width, ov), px(ou, ov)]
    if shape.get("texture_mirror", False):  # model_shape_generate_plane: swap left/right points
        quad = [quad[1], quad[0], quad[3], quad[2]]
    vertices: list[Vec3] = []
    faces = []
    uvs: list[tuple[float, float]] = []
    if not shape.get("hide_front", False):
        vertices += front
        faces.append((0, 1, 2, 3))
        uvs += quad
    if not shape.get("hide_back", False):
        base = len(vertices)
        vertices += [front[3], front[2], front[1], front[0]]
        faces.append((base, base + 1, base + 2, base + 3))
        # Back corner k is front corner 3-k, so it keeps that corner's UV.
        uvs += [quad[3], quad[2], quad[1], quad[0]]
    if invert:
        vertices_new = list(vertices)
        faces = [tuple(reversed(f)) for f in faces]
        uvs_new: list[tuple[float, float]] = []
        for i in range(0, len(uvs), 4):
            uvs_new.extend(reversed(uvs[i:i + 4]))
        vertices, uvs = vertices_new, uvs_new
    return vertices, faces, uvs, warnings


# --------------------------------------------------------------------------
# Bend math
# --------------------------------------------------------------------------

def ease_in_out_quint(t: float) -> float:
    """GameMaker/Mine-imator ease("easeinoutquint")."""
    t = min(1.0, max(0.0, t))
    return 16.0 * t ** 5 if t < 0.5 else 1.0 - ((-2.0 * t + 2.0) ** 5) / 2.0


def _rot_x(a: float) -> Mat3:
    c, s = math.cos(a), math.sin(a)
    return ((1, 0, 0), (0, c, -s), (0, s, c))


def _rot_y(a: float) -> Mat3:
    c, s = math.cos(a), math.sin(a)
    return ((c, 0, s), (0, 1, 0), (-s, 0, c))


def _rot_z(a: float) -> Mat3:
    c, s = math.cos(a), math.sin(a)
    return ((c, -s, 0), (s, c, 0), (0, 0, 1))


def _mul(a: Mat3, b: Mat3) -> Mat3:
    return tuple(  # type: ignore[return-value]
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3)
    )


def engine_rotation(rx_deg: float, ry_deg: float, rz_deg: float) -> Mat3:
    """Same composition as blender_importer._engine_rotation_matrix (Z @ X @ -Y)."""
    rx, ry, rz = (math.radians(v) for v in (rx_deg, ry_deg, rz_deg))
    return _mul(_mul(_rot_z(rz), _rot_x(rx)), _rot_y(-ry))


def apply_rotation(m: Mat3, v: Vec3) -> Vec3:
    return (
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    )


@dataclass
class BendSpec:
    """A parsed .mimodel "bend" block, in Blender part-local units."""

    part: str = "lower"
    axes: list[str] = field(default_factory=lambda: ["x", "y", "z"])
    offset: float = 0.0      # blocks
    size: float = 0.0        # blocks; 0 = hard hinge
    size_units: float = 0.0  # Mine-imator units (what the original code calls bendsize)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_definition(cls, definition: dict[str, Any], units_per_block: float = 16.0) -> "BendSpec":
        axes = definition.get("axis", ["x", "y", "z"])
        if isinstance(axes, str):
            axes = [axes]
        # model_shape_generate_block: `bend_size = null` means a smooth 4 unit
        # bend.  An explicit 0 is a hard hinge.
        raw_size = definition.get("size")
        try:
            size = DEFAULT_BEND_SIZE if raw_size is None else float(raw_size)
        except (TypeError, ValueError):
            size = DEFAULT_BEND_SIZE
        try:
            offset = float(definition.get("offset", 0.0))
        except (TypeError, ValueError):
            offset = 0.0
        return cls(
            part=str(definition.get("part", "lower")).lower(),
            axes=[str(a).lower() for a in axes],
            offset=offset / units_per_block,
            size=max(0.0, size) / units_per_block,
            size_units=max(0.0, size),
            raw=definition,
        )

    # -- per-axis settings --------------------------------------------------
    def _per_axis(self, name: str, axis: str, default: Any) -> Any:
        value = self.raw.get(name, default)
        if isinstance(value, (list, tuple)):
            try:
                index = self.axes.index(axis)
            except ValueError:
                return default
            return value[index] if index < len(value) else default
        return value

    def engine_angles(self, state: dict[str, Any]) -> tuple[float, float, float]:
        """Clamp/invert the timeline bend angles; returns engine ROT_X/Y/Z degrees.

        .mimodel JSON axes are Y-up, so JSON z drives engine Y and JSON y
        drives engine Z.
        """
        out = []
        for key, axis in (("BEND_ANGLE_X", "x"), ("BEND_ANGLE_Y", "z"), ("BEND_ANGLE_Z", "y")):
            if axis not in self.axes:
                out.append(0.0)
                continue
            try:
                angle = float(state.get(key, 0.0))
                lo = float(self._per_axis("direction_min", axis, -180.0))
                hi = float(self._per_axis("direction_max", axis, 180.0))
            except (TypeError, ValueError):
                out.append(0.0)
                continue
            angle = min(hi, max(lo, angle))
            if bool(self._per_axis("invert", axis, False)):
                angle = -angle
            out.append(angle)
        return (out[0], out[1], out[2])

    # -- region -------------------------------------------------------------
    def axis_and_pivot(self) -> tuple[int, float, bool]:
        """(local axis index, pivot coordinate, bend the high side?)."""
        if self.part in {"upper", "lower"}:
            return 2, self.offset, self.part == "upper"
        if self.part in {"right", "left"}:
            return 0, self.offset, self.part == "right"
        return 1, -self.offset, self.part == "back"

    def weight(self, coord: float, eased: bool = False) -> float:
        """0..1 share of the bend rotation a point at `coord` receives.

        model_shape_get_bend multiplies the X and Y bend angles by an
        easeinoutquint curve of this weight and the Z angle linearly, so pass
        ``eased=True`` when the bend uses X or Y.
        """
        _, pivot, positive = self.axis_and_pivot()
        if self.size <= EPS:
            inside = coord >= pivot if positive else coord <= pivot
            return 1.0 if inside else 0.0
        lo, hi = pivot - self.size * 0.5, pivot + self.size * 0.5
        t = (coord - lo) / (hi - lo)
        t = min(1.0, max(0.0, t))
        t = t if positive else 1.0 - t
        return ease_in_out_quint(t) if eased else t

    @staticmethod
    def uses_easing(angles: tuple[float, float, float]) -> bool:
        return abs(angles[0]) > 1e-8 or abs(angles[1]) > 1e-8

    def plane_variant(self) -> "BendSpec":
        """Planes are flat on engine Y, so model_shape_generate_plane slices them
        along Z even for front/back bends (reusing the offset as a Z offset)."""
        if self.part not in ("front", "back"):
            return self
        from dataclasses import replace
        return replace(self, part="upper" if self.part == "front" else "lower")

    @property
    def segments(self) -> int:
        """Slices across the bend region: the original uses max(size, 2)."""
        if self.size <= EPS:
            return 1
        return int(max(2, min(MAX_BEND_SEGMENTS, round(self.size_units))))

    def cut_positions(self) -> list[float]:
        """Coordinates (along the bend axis) to slice meshes at."""
        _, pivot, _ = self.axis_and_pivot()
        if self.size <= EPS:
            return [pivot]
        lo = pivot - self.size * 0.5
        n = self.segments
        return [lo + self.size * k / n for k in range(n + 1)]

    def bone_rotation(self, angles: tuple[float, float, float]) -> Mat3:
        """Full (weight 1) rotation carried by the single bend bone."""
        return engine_rotation(angles[0], angles[1], angles[2])


def weighted_rotation(angles: tuple[float, float, float], weight: float) -> Mat3:
    return engine_rotation(angles[0] * weight, angles[1] * weight, angles[2] * weight)


# --------------------------------------------------------------------------
# 3D planes (model_shape_generate_plane_3d)
# --------------------------------------------------------------------------

@dataclass
class Plane3D:
    vertices: list[Vec3]
    faces: list[tuple[int, int, int, int]]
    loop_uvs: list[tuple[float, float]]   # 4 per face, Blender V (bottom = 0)
    voxels: int = 0
    sample_size: tuple[int, int] = (0, 0)


def plane3d_geometry(
    shape: dict[str, Any],
    texture_size: Sequence[float],
    image_size: tuple[int, int],
    opaque: Callable[[int, int], bool],
    to_blender: Callable[[Sequence[float]], Vec3],
) -> Plane3D:
    """Voxelize a "3d": true plane exactly like Mine-imator.

    One cube per fully opaque texture pixel, 1 unit thick on the plane's thin
    (JSON Z) axis.  Only exposed side faces are emitted, and every face of a
    voxel samples that pixel's own texel, so nothing is stretched.

    ``opaque(px, py)`` answers for absolute image pixels with (0, 0) at the TOP
    left.  Geometry is unscaled; apply the shape scale on the object.
    """
    f = _vec3(shape.get("from"), -8.0)
    t = _vec3(shape.get("to"), 8.0)
    uv = _vec3(shape.get("uv"))[:2]
    side = max(float(texture_size[0]), float(texture_size[1]))
    tw, th = float(image_size[0]), float(image_size[1])
    width, height = t[0] - f[0], t[1] - f[1]
    if width < 0:
        f[0], t[0], width = t[0], f[0], -width
    if height < 0:
        f[1], t[1], height = t[1], f[1], -height
    sx = max(1, int(math.ceil(width / side * tw - 1e-9)))
    sy = max(1, int(math.ceil(height / side * th - 1e-9)))
    sample_x0 = int(math.ceil(uv[0] / side * tw - 1e-9))
    sample_y0 = int(math.ceil(uv[1] / side * th - 1e-9))
    texscale = (math.ceil(width - 1e-9) / sx, math.ceil(height - 1e-9) / sy)
    tex_pixel = (texscale[0] / side, texscale[1] / side)         # 1 / texsizescale
    pix_uv = ((1 - 1 / 256) * tex_pixel[0], (1 - 1 / 256) * tex_pixel[1])
    uv0 = (uv[0] / side, uv[1] / side)

    frac_h = height - math.floor(height)
    xs = [f[0] + i * texscale[0] for i in range(sx + 1)]
    zs: list[float] = []
    z = f[1]
    for j in range(sy + 1):
        zs.append(z)
        z += texscale[1] * (frac_h if (j == sy - 1 and frac_h > 1e-9) else 1.0)
    d0, d1 = f[2], f[2] + 1.0                                      # thickness (JSON Z)
    mirror = bool(shape.get("texture_mirror", False))
    invert = bool(shape.get("invert", False))
    hide_front, hide_back = bool(shape.get("hide_front", False)), bool(shape.get("hide_back", False))

    def solid(ax: int, ay: int) -> bool:
        if ax < 0 or ay < 0 or ax >= sx or ay >= sy:
            return False
        return bool(opaque(sample_x0 + ax, sample_y0 + ay))

    zero = to_blender((0.0, 0.0, 0.0))

    def direction(n: Sequence[float]) -> Vec3:
        v = to_blender(n)
        return (v[0] - zero[0], v[1] - zero[1], v[2] - zero[2])

    vertices: list[Vec3] = []
    faces: list[tuple[int, int, int, int]] = []
    loop_uvs: list[tuple[float, float]] = []
    voxels = 0

    def add_face(corners: list[Sequence[float]], outward: Sequence[float], uvs: list[tuple[float, float]]) -> None:
        pts = [to_blender(c) for c in corners]
        e1 = (pts[1][0] - pts[0][0], pts[1][1] - pts[0][1], pts[1][2] - pts[0][2])
        e2 = (pts[2][0] - pts[0][0], pts[2][1] - pts[0][1], pts[2][2] - pts[0][2])
        n = (e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0])
        o = direction(outward)
        order = [0, 1, 2, 3]
        if (n[0] * o[0] + n[1] * o[1] + n[2] * o[2] < 0) != invert:
            order = [0, 3, 2, 1]
        base = len(vertices)
        vertices.extend(pts[i] for i in order)
        loop_uvs.extend(uvs[i] for i in order)
        faces.append((base, base + 1, base + 2, base + 3))

    for outer in range(sx):
        for inner in range(sy):
            ax = outer
            ay = sy - 1 - inner
            if mirror:
                ax = sx - 1 - ax
            if not solid(ax, ay):
                continue
            voxels += 1
            wface = not solid(ax - 1, ay)
            eface = not solid(ax + 1, ay)
            aface = not solid(ax, ay - 1)
            bface = not solid(ax, ay + 1)
            if mirror:
                wface, eface = eface, wface
            u0 = uv0[0] + ax * tex_pixel[0]
            v0 = uv0[1] + ay * tex_pixel[1]
            u1, vb = u0 + pix_uv[0], v0 + pix_uv[1]
            quad = [(u0, 1 - vb), (u1, 1 - vb), (u1, 1 - v0), (u0, 1 - v0)]    # BL, BR, TR, TL
            X0, X1, Z0, Z1 = xs[outer], xs[outer + 1], zs[inner], zs[inner + 1]
            if eface:
                add_face([(X1, Z0, d0), (X1, Z0, d1), (X1, Z1, d1), (X1, Z1, d0)], (1, 0, 0), quad)
            if wface:
                add_face([(X0, Z0, d1), (X0, Z0, d0), (X0, Z1, d0), (X0, Z1, d1)], (-1, 0, 0), quad)
            if not hide_front:   # "South": +Y in the engine, JSON +Z
                add_face([(X0, Z0, d1), (X1, Z0, d1), (X1, Z1, d1), (X0, Z1, d1)], (0, 0, 1), quad)
            if not hide_back:    # "North"
                add_face([(X1, Z0, d0), (X0, Z0, d0), (X0, Z1, d0), (X1, Z1, d0)], (0, 0, -1), quad)
            if aface:
                add_face([(X0, Z1, d1), (X1, Z1, d1), (X1, Z1, d0), (X0, Z1, d0)], (0, 1, 0), quad)
            if bface:
                add_face([(X0, Z0, d0), (X1, Z0, d0), (X1, Z0, d1), (X0, Z0, d1)], (0, -1, 0), quad)
    return Plane3D(vertices, faces, loop_uvs, voxels, (sx, sy))
