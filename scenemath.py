"""Blender-independent scene math for the Mine-imator importer.

Matrices are row-major 4x4 tuples (the layout ``mathutils.Matrix(rows)``
accepts), so the Blender glue only wraps them.  Covers:

* shape placement honouring the saved rotation point
* ``inherit.position / rotation / scale`` = false (parent contribution is
  stripped from the parent's world matrix)
* colour / alpha / visibility inheritance along the timeline tree
* which timelines belong to a model (bodyparts) and which are *attached*
  (folders, shapes, nested models, ...)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence
import math

try:  # package import inside Blender, plain import in tests
    from . import core, modelgeom
except ImportError:  # pragma: no cover
    import core  # type: ignore
    import modelgeom  # type: ignore

Mat = tuple[tuple[float, float, float, float], tuple[float, float, float, float],
            tuple[float, float, float, float], tuple[float, float, float, float]]

SHAPE_TYPES = ("cube", "cone", "cylinder", "sphere", "surface")
# Mine-imator stores (0, -8, 0) (Y-up) as the default pivot of primitives:
# bottom-centre of a +/-8 unit shape, so a shape sits on its position.
DEFAULT_SHAPE_ROT_POINT = (0.0, -8.0, 0.0)


# --------------------------------------------------------------------------
# Small matrix toolkit
# --------------------------------------------------------------------------

def identity() -> Mat:
    return ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))


def mul(a: Mat, b: Mat) -> Mat:
    return tuple(  # type: ignore[return-value]
        tuple(sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)) for i in range(4)
    )


def translation(v: Sequence[float]) -> Mat:
    return ((1.0, 0.0, 0.0, float(v[0])), (0.0, 1.0, 0.0, float(v[1])),
            (0.0, 0.0, 1.0, float(v[2])), (0.0, 0.0, 0.0, 1.0))


def diagonal(s: Sequence[float]) -> Mat:
    return ((float(s[0]), 0.0, 0.0, 0.0), (0.0, float(s[1]), 0.0, 0.0),
            (0.0, 0.0, float(s[2]), 0.0), (0.0, 0.0, 0.0, 1.0))


def from_rot3(r: modelgeom.Mat3) -> Mat:
    return ((r[0][0], r[0][1], r[0][2], 0.0), (r[1][0], r[1][1], r[1][2], 0.0),
            (r[2][0], r[2][1], r[2][2], 0.0), (0.0, 0.0, 0.0, 1.0))


def apply(m: Mat, p: Sequence[float]) -> tuple[float, float, float]:
    return tuple(m[i][0] * p[0] + m[i][1] * p[1] + m[i][2] * p[2] + m[i][3] for i in range(3))  # type: ignore[return-value]


def inverse(m: Mat) -> Mat:
    """General 4x4 inverse (Gauss-Jordan with partial pivoting)."""
    a = [list(row) + [1.0 if i == j else 0.0 for j in range(4)] for i, row in enumerate(m)]
    for col in range(4):
        pivot = max(range(col, 4), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-14:
            raise ValueError("matrix is singular")
        a[col], a[pivot] = a[pivot], a[col]
        scale = a[col][col]
        a[col] = [x / scale for x in a[col]]
        for r in range(4):
            if r != col:
                factor = a[r][col]
                if factor:
                    a[r] = [x - factor * y for x, y in zip(a[r], a[col])]
    return tuple(tuple(row[4:]) for row in a)  # type: ignore[return-value]


def is_identity(m: Mat, tol: float = 1e-9) -> bool:
    return all(abs(m[i][j] - (1.0 if i == j else 0.0)) <= tol for i in range(4) for j in range(4))


# --------------------------------------------------------------------------
# Placement
# --------------------------------------------------------------------------

def engine_rotation_matrix(state: dict[str, Any]) -> Mat:
    """Same Z @ X @ -Y composition as blender_importer._engine_rotation_matrix."""
    return from_rot3(modelgeom.engine_rotation(
        float(state.get("ROT_X", 0.0)), float(state.get("ROT_Y", 0.0)), float(state.get("ROT_Z", 0.0))))


def plain_local_matrix(state: dict[str, Any]) -> Mat:
    """T(pos) @ R @ S for folders / helpers (no pivot shift)."""
    return mul(mul(translation(core.mi_position(state)), engine_rotation_matrix(state)),
               diagonal(core.mi_scale(state)))


def pivot_of(timeline: dict[str, Any], kind: str) -> tuple[float, float, float]:
    """The saved rotation point in Mine-imator (.mimodel-style, Y-up) units."""
    raw = timeline.get("rot_point")
    if isinstance(raw, (list, tuple)) and len(raw) >= 3:
        try:
            return (float(raw[0]), float(raw[1]), float(raw[2]))
        except (TypeError, ValueError):
            pass
    return DEFAULT_SHAPE_ROT_POINT if kind in SHAPE_TYPES else (0.0, 0.0, 0.0)


def shape_local_matrix(state: dict[str, Any], rot_point_yup: Sequence[float]) -> Mat:
    """T(pos) @ R @ S @ T(-pivot).

    The pivot becomes the object's local origin, exactly as the importer's
    ``_pixel_item`` treats item pivots (rot_point goes through ``mi_vector``).
    """
    pivot = core.mi_vector(rot_point_yup)
    return mul(plain_local_matrix(state), translation((-pivot[0], -pivot[1], -pivot[2])))


# --------------------------------------------------------------------------
# inherit.position / rotation / scale = false
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Inherit:
    position: bool = True
    rotation: bool = True
    scale: bool = True
    color: bool = True
    alpha: bool = True
    visibility: bool = True

    @classmethod
    def of(cls, timeline: dict[str, Any]) -> "Inherit":
        raw = timeline.get("inherit")
        raw = raw if isinstance(raw, dict) else {}

        def flag(key: str) -> bool:
            return bool(raw.get(key, True))

        return cls(flag("position"), flag("rotation"), flag("scale"), flag("color"), flag("alpha"), flag("visibility"))

    @property
    def transforms_stripped(self) -> bool:
        return not (self.position and self.rotation and self.scale)


def strip_inherited(world: Mat, inherit: Inherit) -> Mat:
    """Parent world matrix with the non-inherited components removed.

    Columns of the linear part are split into scale (length) and rotation
    (direction).  Dropped components are replaced by identity.
    """
    cols = [(world[0][j], world[1][j], world[2][j]) for j in range(3)]
    lengths = [math.sqrt(sum(c * c for c in col)) for col in cols]
    axes = []
    for j, (col, length) in enumerate(zip(cols, lengths)):
        if length > 1e-12:
            axes.append(tuple(c / length for c in col))
        else:
            axes.append(tuple(1.0 if i == j else 0.0 for i in range(3)))
    out = [[0.0] * 4 for _ in range(4)]
    for j in range(3):
        direction = axes[j] if inherit.rotation else tuple(1.0 if i == j else 0.0 for i in range(3))
        factor = lengths[j] if inherit.scale else 1.0
        for i in range(3):
            out[i][j] = direction[i] * factor
    for i in range(3):
        out[i][3] = world[i][3] if inherit.position else 0.0
    out[3][3] = 1.0
    return tuple(tuple(row) for row in out)  # type: ignore[return-value]


def parent_inverse_for(parent_world: Mat, inherit: Inherit) -> Mat:
    """Value for ``obj.matrix_parent_inverse`` so that
    ``parent_world @ parent_inverse == strip_inherited(parent_world)``.
    """
    return mul(inverse(parent_world), strip_inherited(parent_world, inherit))


# --------------------------------------------------------------------------
# Colour / alpha / visibility inheritance
# --------------------------------------------------------------------------

Color = tuple[float, float, float]
WHITE: Color = (1.0, 1.0, 1.0)


def parse_hex(value: Any, default: Color = WHITE) -> Color:
    if not isinstance(value, str):
        return default
    raw = value.strip().lstrip("#")
    if len(raw) >= 6:
        try:
            return tuple(int(raw[i:i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]
        except ValueError:
            pass
    return default


@dataclass(frozen=True)
class Style:
    rgb_mul: Color = WHITE
    mix_color: Color = WHITE
    mix_percent: float = 0.0
    alpha: float = 1.0
    visible: bool = True

    @property
    def is_plain(self) -> bool:
        return self.rgb_mul == WHITE and self.mix_percent <= 1e-6 and self.alpha >= 1.0

    def key(self) -> tuple:
        r = lambda c: tuple(round(x, 4) for x in c)
        return (r(self.rgb_mul), r(self.mix_color), round(self.mix_percent, 4), round(self.alpha, 4))


def own_style(state: dict[str, Any]) -> Style:
    try:
        percent = max(0.0, min(1.0, float(state.get("MIX_PERCENT", 0.0))))
    except (TypeError, ValueError):
        percent = 0.0
    try:
        alpha = max(0.0, float(state.get("ALPHA", 1.0)))
    except (TypeError, ValueError):
        alpha = 1.0
    return Style(
        rgb_mul=parse_hex(state.get("RGB_MUL")),
        mix_color=parse_hex(state.get("MIX_COLOR")),
        mix_percent=percent,
        alpha=min(alpha, 1.0),
        visible=bool(state.get("VISIBLE", True)),
    )


def combine_style(parent: Style, own: Style, inherit: Inherit) -> Style:
    rgb, mix_color, mix_percent = own.rgb_mul, own.mix_color, own.mix_percent
    if inherit.color:
        rgb = tuple(a * b for a, b in zip(parent.rgb_mul, own.rgb_mul))  # type: ignore[assignment]
        p1, p2 = parent.mix_percent, own.mix_percent
        if p1 > 1e-6 and p2 > 1e-6:
            mix_percent = p1 + p2 - p1 * p2
            w1, w2 = p1 * (1 - p2), p2
            mix_color = tuple((a * w1 + b * w2) / (w1 + w2) for a, b in zip(parent.mix_color, own.mix_color))  # type: ignore[assignment]
        elif p1 > 1e-6:
            mix_color, mix_percent = parent.mix_color, p1
    alpha = own.alpha * parent.alpha if inherit.alpha else own.alpha
    visible = own.visible and (parent.visible if inherit.visibility else True)
    return Style(rgb, mix_color, mix_percent, alpha, visible)


def resolve_style(project: Any, timeline: dict[str, Any], cache: dict[str, Style] | None = None) -> Style:
    """Effective style of a timeline after walking its ancestors."""
    cache = cache if cache is not None else {}
    tid = str(timeline.get("id"))
    if tid in cache:
        return cache[tid]
    own = own_style(core.frame0_state(timeline))
    parent = project.timeline(timeline.get("parent"))
    result = combine_style(resolve_style(project, parent, cache), own, Inherit.of(timeline)) if parent else own
    cache[tid] = result
    return result


# --------------------------------------------------------------------------
# Hierarchy
# --------------------------------------------------------------------------

def model_bodyparts(project: Any, model_id: str) -> list[dict[str, Any]]:
    """Bodyparts that belong to *this* model, parents before children.

    Only bodypart -> bodypart links are followed.  Walking every descendant
    (the old behaviour) leaked a nested model's bodyparts into its parent
    model whenever both used the same part names.
    """
    result: list[dict[str, Any]] = []
    pending = [c for c in project.children(model_id) if str(c.get("type", "")).lower() == "bodypart"]
    while pending:
        item = pending.pop(0)
        result.append(item)
        pending.extend(c for c in project.children(str(item.get("id"))) if str(c.get("type", "")).lower() == "bodypart")
    return result


def attached_children(project: Any, timeline_id: str) -> list[dict[str, Any]]:
    """Direct children that are not bodyparts or audio: folders, shapes,
    nested models, items, ... They hang off the parent's object."""
    return [c for c in project.children(timeline_id)
            if str(c.get("type", "")).lower() not in {"bodypart", "audio"}]
