"""Blender-independent ports of Mine-imator's primitive shape generators.

Ported 1:1 from ``vbuffer_create_surface / cylinder / cone / sphere`` (GML).
Everything is plain Python so it can be unit-tested outside Blender.

Coordinate handling
-------------------
The GML generators emit triangles in Mine-imator's engine space (Z-up, front
face = counter-clockwise by the cross-product rule, which the tests verify
against the normals the GML itself declares).  ``build_shape`` converts the
result to Blender space:

* positions: engine (x, y, z) / 16  ->  Blender (x, -y, z)
* the Y flip is a reflection, so triangle winding is reversed to keep
  outward-facing normals
* V is flipped (GameMaker V=0 is the top row, Blender V=0 is the bottom)
* vertices are welded by position; UVs and normals stay per face-corner

Assumptions that were NOT in the supplied source (flagged in the report):
* ``vbuffer_add_triangle``: flat normal = normalize(cross(p2-p1, p3-p1)); with
  ``invert`` the normal is negated and the winding reversed.
* ``negate(invert)`` is +1 / -1.
* radius passed by Mine-imator is half a block (8 units), so a shape at scale
  1 is exactly one block across.
* ``cube`` has no supplied generator; it is built as +/-radius with each face
  mapping the whole tex1..tex2 rectangle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence
import math

PI = math.pi
MI_UNITS_PER_BLOCK = 16.0
MI_SHAPE_RADIUS = 8.0          # engine units; shape spans one block at scale 1
MAX_DETAIL = 256
SHAPE_KINDS = ("cube", "cone", "cylinder", "sphere", "surface")


# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ShapeParams:
    kind: str
    detail: int = 32
    closed: bool = True
    invert: bool = False
    mapped: bool = False
    hoffset: float = 0.0
    voffset: float = 0.0
    hrepeat: float = 1.0
    vrepeat: float = 1.0
    hmirror: bool = False
    vmirror: bool = False

    @classmethod
    def from_template(cls, kind: str, shape: dict[str, Any] | None) -> "ShapeParams":
        s = shape or {}

        def num(key: str, default: float) -> float:
            try:
                return float(s.get(key, default))
            except (TypeError, ValueError):
                return default

        return cls(
            kind=str(kind).lower(),
            detail=int(max(3, min(MAX_DETAIL, num("detail", 32)))),
            closed=bool(s.get("closed", True)),
            invert=bool(s.get("invert", False)),
            mapped=bool(s.get("tex_mapped", False)),
            hoffset=num("tex_hoffset", 0.0),
            voffset=num("tex_voffset", 0.0),
            hrepeat=num("tex_hrepeat", 1.0),
            vrepeat=num("tex_vrepeat", 1.0),
            hmirror=bool(s.get("tex_hmirror", False)),
            vmirror=bool(s.get("tex_vmirror", False)),
        )

    def tex_corners(self) -> tuple[list[float], list[float], float, float]:
        """tex1, tex2, texhorflip, texverflip as the GML functions receive them.

        Offset/repeat define the UV rectangle; mirroring swaps the matching
        axis (unmapped) or flips the sign passed as thflip/tvflip (mapped).
        """
        tex1 = [self.hoffset, self.voffset]
        tex2 = [self.hoffset + self.hrepeat, self.voffset + self.vrepeat]
        if self.hmirror:
            tex1[0], tex2[0] = tex2[0], tex1[0]
        if self.vmirror:
            tex1[1], tex2[1] = tex2[1], tex1[1]
        return tex1, tex2, (-1.0 if self.hmirror else 1.0), (-1.0 if self.vmirror else 1.0)


@dataclass
class ShapeMesh:
    """Welded mesh in Blender space, in blocks (1.0 = 16 MI units)."""

    vertices: list[tuple[float, float, float]]
    faces: list[tuple[int, int, int]]
    loop_uvs: list[tuple[float, float]]        # 3 per face, Blender V (bottom=0)
    loop_normals: list[tuple[float, float, float]]  # 3 per face, Blender space


# --------------------------------------------------------------------------
# vbuffer emulation
# --------------------------------------------------------------------------

Corner = tuple[float, float, float, float, float, float, float, float]  # x y z nx ny nz u v


class _Soup:
    """Collects triangles the way vbuffer_add_triangle / vertex_add do."""

    def __init__(self) -> None:
        self.triangles: list[tuple[Corner, Corner, Corner]] = []
        self._pending: list[Corner] = []

    def vertex_add(self, x, y, z, nx, ny, nz, u, v) -> None:
        self._pending.append((x, y, z, nx, ny, nz, u, v))
        if len(self._pending) == 3:
            a, b, c = self._pending
            self.triangles.append((a, b, c))
            self._pending = []

    def add_triangle(self, x1, y1, z1, x2, y2, z2, x3, y3, z3,
                     u1, v1, u2, v2, u3, v3, invert) -> None:
        ax, ay, az = x2 - x1, y2 - y1, z2 - z1
        bx, by, bz = x3 - x1, y3 - y1, z3 - z1
        nx, ny, nz = ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx
        length = math.sqrt(nx * nx + ny * ny + nz * nz)
        if length > 1e-12:
            nx, ny, nz = nx / length, ny / length, nz / length
        else:
            nx, ny, nz = 0.0, 0.0, 1.0
        if invert:
            nx, ny, nz = -nx, -ny, -nz
            self.vertex_add(x1, y1, z1, nx, ny, nz, u1, v1)
            self.vertex_add(x3, y3, z3, nx, ny, nz, u3, v3)
            self.vertex_add(x2, y2, z2, nx, ny, nz, u2, v2)
        else:
            self.vertex_add(x1, y1, z1, nx, ny, nz, u1, v1)
            self.vertex_add(x2, y2, z2, nx, ny, nz, u2, v2)
            self.vertex_add(x3, y3, z3, nx, ny, nz, u3, v3)


# --------------------------------------------------------------------------
# Generators (line-for-line ports of the GML)
# --------------------------------------------------------------------------

def _surface(s: _Soup, rad, tex1, tex2, invert) -> None:
    s.add_triangle(-rad, 0, rad, rad, 0, rad, rad, 0, -rad,
                   tex1[0], tex1[1], tex2[0], tex1[1], tex2[0], tex2[1], invert)
    s.add_triangle(-rad, 0, -rad, -rad, 0, rad, rad, 0, -rad,
                   tex1[0], tex2[1], tex1[0], tex1[1], tex2[0], tex2[1], invert)


def _cylinder(s: _Soup, rad, tex1, tex2, thflip, tvflip, detail, closed, invert, mapped) -> None:
    tex1 = [tex1[0] + 0.25, tex1[1]]
    tex2 = [tex2[0] + 0.25, tex2[1]]
    i = 0.0
    for _ in range(detail):
        ip = i
        i += 1.0 / detail
        texsize = [tex2[0] - tex1[0], tex2[1] - tex1[1]]
        texmid = [tex1[0] + texsize[0] / 2, tex1[1] + texsize[1] / 2]

        n1x, n1y = math.cos(ip * PI * 2), -math.sin(ip * PI * 2)
        n2x, n2y = math.cos(i * PI * 2), -math.sin(i * PI * 2)
        x1, y1, x2, y2 = n1x * rad, n1y * rad, n2x * rad, n2y * rad

        if invert:
            n1x, n1y, n2x, n2y = -n1x, -n1y, -n2x, -n2y

        if mapped:
            texsize = [(1 / 3) * thflip, tvflip]
            texmid[1] = texsize[1] / 2

        if closed:
            if mapped:
                texmid[0] = 5 / 6
            s.add_triangle(0, 0, -rad, x1, y1, -rad, x2, y2, -rad,
                           texmid[0], texmid[1],
                           texmid[0] + math.cos(ip * PI * 2) * (texsize[0] / 2),
                           texmid[1] + math.sin(ip * PI * 2) * (texsize[1] / 2),
                           texmid[0] + math.cos(i * PI * 2) * (texsize[0] / 2),
                           texmid[1] + math.sin(i * PI * 2) * (texsize[1] / 2), invert)
            if mapped:
                texmid[0] = 1 / 2
            s.add_triangle(0, 0, rad, x2, y2, rad, x1, y1, rad,
                           texmid[0], texmid[1],
                           texmid[0] + math.cos(i * PI * 2) * (texsize[0] / 2),
                           texmid[1] - math.sin(i * PI * 2) * (texsize[1] / 2),
                           texmid[0] + math.cos(ip * PI * 2) * (texsize[0] / 2),
                           texmid[1] - math.sin(ip * PI * 2) * (texsize[1] / 2), invert)

        if mapped:
            tex1 = [0.0, 0.0]
            tex2 = [abs(texsize[0]), abs(texsize[1])]
            if thflip < 0:
                tex1[0], tex2[0] = tex2[0], tex1[0]
            if tvflip < 0:
                tex1[1], tex2[1] = tex2[1], tex1[1]

        u_ip = tex1[0] + texsize[0] * ip
        u_i = tex1[0] + texsize[0] * i
        v_top, v_bot = tex1[1], tex1[1] + texsize[1]
        if invert:
            s.vertex_add(x1, y1, rad, n1x, n1y, 0, u_ip, v_top)
            s.vertex_add(x1, y1, -rad, n1x, n1y, 0, u_ip, v_bot)
            s.vertex_add(x2, y2, rad, n2x, n2y, 0, u_i, v_top)
            s.vertex_add(x2, y2, -rad, n2x, n2y, 0, u_i, v_bot)
            s.vertex_add(x2, y2, rad, n2x, n2y, 0, u_i, v_top)
            s.vertex_add(x1, y1, -rad, n1x, n1y, 0, u_ip, v_bot)
        else:
            s.vertex_add(x1, y1, -rad, n1x, n1y, 0, u_ip, v_bot)
            s.vertex_add(x1, y1, rad, n1x, n1y, 0, u_ip, v_top)
            s.vertex_add(x2, y2, rad, n2x, n2y, 0, u_i, v_top)
            s.vertex_add(x2, y2, rad, n2x, n2y, 0, u_i, v_top)
            s.vertex_add(x2, y2, -rad, n2x, n2y, 0, u_i, v_bot)
            s.vertex_add(x1, y1, -rad, n1x, n1y, 0, u_ip, v_bot)


def _cone(s: _Soup, rad, tex1, tex2, thflip, tvflip, detail, closed, invert, mapped) -> None:
    tex1 = [tex1[0] + 0.25, tex1[1]]
    tex2 = [tex2[0] + 0.25, tex2[1]]
    i = 0.0
    for _ in range(detail):
        ip = i
        i += 1.0 / detail
        texsize = [tex2[0] - tex1[0], tex2[1] - tex1[1]]
        texmid = [tex1[0] + texsize[0] / 2, tex1[1] + texsize[1] / 2]

        n1x, n1y = math.cos(ip * PI * 2), -math.sin(ip * PI * 2)
        n2x, n2y = math.cos(i * PI * 2), -math.sin(i * PI * 2)
        x1, y1, x2, y2 = n1x * rad, n1y * rad, n2x * rad, n2y * rad

        if invert:
            n1x, n1y, n2x, n2y = -n1x, -n1y, -n2x, -n2y

        if mapped:
            texsize = [0.5 * thflip, tvflip]
            texmid[1] = texsize[1] / 2

        if closed:
            if mapped:
                texmid[0] = 3 / 4
            s.add_triangle(0, 0, -rad, x1, y1, -rad, x2, y2, -rad,
                           texmid[0], texmid[1],
                           texmid[0] + math.cos(ip * PI * 2) * (texsize[0] / 2),
                           texmid[1] + math.sin(ip * PI * 2) * (texsize[1] / 2),
                           texmid[0] + math.cos(i * PI * 2) * (texsize[0] / 2),
                           texmid[1] + math.sin(i * PI * 2) * (texsize[1] / 2), invert)

        if mapped:
            tex1 = [0.0, 0.0]
            tex2 = [abs(texsize[0]), abs(texsize[1])]
            if thflip < 0:
                tex1[0], tex2[0] = tex2[0], tex1[0]
            if tvflip < 0:
                tex1[1], tex2[1] = tex2[1], tex1[1]

        u_ip = tex1[0] + texsize[0] * ip
        u_i = tex1[0] + texsize[0] * i
        v_top, v_bot = tex1[1], tex1[1] + texsize[1]
        if invert:
            s.vertex_add(x2, y2, -rad, n2x, n2y, 0, u_i, v_bot)
            s.vertex_add(0, 0, rad, 0, 0, 1, u_i, v_top)
            s.vertex_add(x1, y1, -rad, n1x, n1y, 0, u_ip, v_bot)
        else:
            s.vertex_add(0, 0, rad, 0, 0, 1, u_i, v_top)
            s.vertex_add(x2, y2, -rad, n2x, n2y, 0, u_i, v_bot)
            s.vertex_add(x1, y1, -rad, n1x, n1y, 0, u_ip, v_bot)


def _sphere(s: _Soup, rad, tex1, tex2, detail, invert) -> None:
    tex1 = [tex1[0] + 0.25, tex1[1]]
    tex2 = [tex2[0] + 0.25, tex2[1]]
    n = -1.0 if invert else 1.0
    rings = detail - 2
    i = 0.0
    for _ in range(detail):
        ip = i
        i += 1.0 / detail
        j = 0.0
        for _ in range(rings):
            jp = j
            j += 1.0 / rings
            texsize = [tex2[0] - tex1[0], tex2[1] - tex1[1]]
            texmid = [tex1[0] + texsize[0] / 2, tex1[1] + texsize[1] / 2]

            def nrm(a: float, b: float) -> tuple[float, float, float]:
                return (math.sin(a * PI * 2) * math.sin(b * PI),
                        -math.cos(a * PI * 2) * math.sin(b * PI),
                        -math.cos(b * PI))

            n1, n2, n3, n4 = nrm(ip, jp), nrm(ip, j), nrm(i, jp), nrm(i, j)
            p1, p2, p3, p4 = ((c * rad for c in nv) for nv in (n1, n2, n3, n4))
            p1, p2, p3, p4 = tuple(p1), tuple(p2), tuple(p3), tuple(p4)

            def uv(nv, a):
                return (tex2[0] - a * texsize[0], texmid[1] - nv[2] * (texsize[1] / 2))

            def add(p, nv, a):
                u, v = uv(nv, a)
                s.vertex_add(p[0], p[1], p[2], nv[0] * n, nv[1] * n, nv[2] * n, u, v)

            if jp > 0:
                if invert:
                    add(p3, n3, i); add(p1, n1, ip); add(p4, n4, i)
                else:
                    add(p1, n1, ip); add(p3, n3, i); add(p4, n4, i)
            if j < 1:
                if invert:
                    add(p4, n4, i); add(p1, n1, ip); add(p2, n2, ip)
                else:
                    add(p1, n1, ip); add(p4, n4, i); add(p2, n2, ip)


def _cube(s: _Soup, rad, tex1, tex2, invert) -> None:
    """No GML was supplied for cubes: +/-rad box, whole tex rectangle per face."""
    r = rad
    # (corner order is counter-clockwise seen from outside, by the cross rule)
    faces = [
        ((r, -r, -r), (r, r, -r), (r, r, r), (r, -r, r)),        # +X
        ((-r, r, -r), (-r, -r, -r), (-r, -r, r), (-r, r, r)),    # -X
        ((r, r, -r), (-r, r, -r), (-r, r, r), (r, r, r)),        # +Y
        ((-r, -r, -r), (r, -r, -r), (r, -r, r), (-r, -r, r)),    # -Y
        ((-r, -r, r), (r, -r, r), (r, r, r), (-r, r, r)),        # +Z
        ((-r, r, -r), (r, r, -r), (r, -r, -r), (-r, -r, -r)),    # -Z
    ]
    uvs = ((tex1[0], tex2[1]), (tex2[0], tex2[1]), (tex2[0], tex1[1]), (tex1[0], tex1[1]))
    for a, b, c, d in faces:
        s.add_triangle(*a, *b, *c, *uvs[0], *uvs[1], *uvs[2], invert)
        s.add_triangle(*a, *c, *d, *uvs[0], *uvs[2], *uvs[3], invert)


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def generate_engine_triangles(params: ShapeParams, radius: float = MI_SHAPE_RADIUS) -> list[tuple[Corner, Corner, Corner]]:
    """Raw engine-space triangles exactly as the GML would emit them."""
    soup = _Soup()
    tex1, tex2, thflip, tvflip = params.tex_corners()
    kind = params.kind
    if kind == "surface":
        _surface(soup, radius, tex1, tex2, params.invert)
    elif kind == "cylinder":
        _cylinder(soup, radius, tex1, tex2, thflip, tvflip, params.detail, params.closed, params.invert, params.mapped)
    elif kind == "cone":
        _cone(soup, radius, tex1, tex2, thflip, tvflip, params.detail, params.closed, params.invert, params.mapped)
    elif kind == "sphere":
        _sphere(soup, radius, tex1, tex2, params.detail, params.invert)
    elif kind == "cube":
        _cube(soup, radius, tex1, tex2, params.invert)
    else:
        raise ValueError(f"Unsupported shape kind: {kind!r}")
    return soup.triangles


def build_shape(params: ShapeParams, radius: float = MI_SHAPE_RADIUS) -> ShapeMesh:
    """Generate a shape and convert it to a welded Blender-space mesh."""
    scale = 1.0 / MI_UNITS_PER_BLOCK
    index: dict[tuple[int, int, int], int] = {}
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    loop_uvs: list[tuple[float, float]] = []
    loop_normals: list[tuple[float, float, float]] = []

    def vertex_index(x: float, y: float, z: float) -> int:
        bx, by, bz = x * scale, -y * scale, z * scale
        key = (round(bx * 1e6), round(by * 1e6), round(bz * 1e6))
        found = index.get(key)
        if found is None:
            found = len(vertices)
            index[key] = found
            vertices.append((bx, by, bz))
        return found

    for a, b, c in generate_engine_triangles(params, radius):
        ids = [vertex_index(corner[0], corner[1], corner[2]) for corner in (a, b, c)]
        if len({*ids}) < 3:
            continue  # collapsed by welding (pole slivers)
        # Reflection (y -> -y) flips handedness: reverse the winding.
        order = (0, 2, 1)
        faces.append((ids[order[0]], ids[order[1]], ids[order[2]]))
        for k in order:
            corner = (a, b, c)[k]
            loop_uvs.append((corner[6], 1.0 - corner[7]))
            loop_normals.append((corner[3], -corner[4], corner[5]))
    return ShapeMesh(vertices, faces, loop_uvs, loop_normals)
