# Generacion de geometria en el espacio interno de Mine-imator.
#
# Portado de los scripts de Mine-imator: model_file_load*, model_shape_generate_*,
# vbuffer_create_*, render_generate_item y vbuffer_add_pixels. Los poligonos se
# devuelven con el mismo sentido de giro que usa Mine-imator (normal exterior =
# producto vectorial dextrogiro de (v1 - v0) x (v2 - v0)); mi_builder se encarga
# de la conversion a Blender.

import json
import math
import os

from . import gm_math as gm
from .mi_data import point3d, point2d, parse_color, as_bool, WHITE, BLACK, ease

X, Y, Z = 0, 1, 2


class MatKey(tuple):
    """(tex, color, alpha, emissive, mix_color, mix_pct, add_color, repeat, cull)"""
    __slots__ = ()

    def __new__(cls, tex=None, color=WHITE, alpha=1.0, emissive=0.0, mix=BLACK, mix_pct=0.0, add=BLACK, repeat=False,
                cull=False):
        r = lambda c: tuple(round(float(v), 4) for v in c)
        return tuple.__new__(cls, (tex, r(color), round(float(alpha), 4), round(float(emissive), 4),
                                   r(mix), round(float(mix_pct), 4), r(add), bool(repeat), bool(cull)))

    def with_cull(self, cull):
        cull = bool(cull)
        if self[8] == cull:
            return self
        return tuple.__new__(MatKey, self[:8] + (cull,))  # valores ya redondeados

    tex = property(lambda s: s[0])
    color = property(lambda s: s[1])
    alpha = property(lambda s: s[2])
    emissive = property(lambda s: s[3])
    mix = property(lambda s: s[4])
    mix_pct = property(lambda s: s[5])
    add = property(lambda s: s[6])
    repeat = property(lambda s: s[7])
    cull = property(lambda s: s[8])


class Poly:
    __slots__ = ('verts', 'uvs', 'mat', 'smooth', 'weights')

    def __init__(self, verts, uvs, mat=None, smooth=False, weights=None):
        self.verts = verts
        self.uvs = uvs
        self.mat = mat
        self.smooth = smooth
        self.weights = weights  # peso del hueso de doblado por vertice (esqueleto)

    def reversed(self):
        return Poly(self.verts[::-1], self.uvs[::-1], self.mat, self.smooth,
                    self.weights[::-1] if self.weights else None)


def quad(a, b, c, d, ta, tb, tc, td):
    return Poly([a, b, c, d], [ta, tb, tc, td])


def tri(a, b, c, ta, tb, tc, invert=False):
    """vbuffer_add_triangle: invert intercambia los dos primeros vertices."""
    if invert:
        return Poly([b, a, c], [tb, ta, tc])
    return Poly([a, b, c], [ta, tb, tc])


# ---------------------------------------------------------------------------
# Modelos (.mimodel)
# ---------------------------------------------------------------------------

PART_AXIS = {'right': X, 'left': X, 'front': Y, 'back': Y, 'upper': Z, 'lower': Z}
AXIS_NAME = {'x': X, 'z': Y, 'y': Z}


class BendInfo:
    pass


class ModelFile:
    def __init__(self, data, source, base):
        """source: 'file' (modelo del usuario) o 'mc' (incluido en Mine-imator).
        base: carpeta del archivo (file) o None."""
        self.source = source
        self.base = base
        self.name = data.get('name', '')
        self.texture_name = data.get('texture', '')
        ts = point2d(data.get('texture_size'), (64, 64))
        s = max(ts)
        self.texture_size = (s, s)
        self.scale = point3d(data.get('scale'), (1, 1, 1))
        self.floor_box_uvs = as_bool(data.get('floor_box_uvs'), False)
        self.player_skin = as_bool(data.get('player_skin'), False)
        # Valores de "padre" para las partes raiz
        self.color_blend = WHITE
        self.color_alpha = 1.0
        self.color_emissive = 0.0
        self.color_mix = BLACK
        self.color_mix_percent = 0.0
        self.bend = None
        self.is_part = False
        self.parts = []
        self.part_map = {}
        for p in data.get('parts') or []:
            part = ModelPart(p, self, self)
            if part.valid:
                self.parts.append(part)

    @classmethod
    def load_file(cls, path):
        with open(path, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
        return cls(data, 'file', os.path.dirname(path))

    @classmethod
    def load_data(cls, data):
        return cls(data, 'mc', None)


def _color_mul(a, b):
    return (a[0] * b[0], a[1] * b[1], a[2] * b[2])


def _color_add(a, b):
    return (min(1.0, a[0] + b[0]), min(1.0, a[1] + b[1]), min(1.0, a[2] + b[2]))


def _list_or_single(val, axes, default):
    """Lee un valor de bend que puede ser unico o lista por eje."""
    out = list(default)
    if isinstance(val, (int, float)) and not isinstance(val, bool) and len(axes) == 1:
        out[axes[0]] = float(val)
    elif isinstance(val, bool) and len(axes) == 1:
        out[axes[0]] = float(val)
    elif isinstance(val, list):
        for i, v in enumerate(val):
            if i < len(axes):
                out[axes[i]] = float(v) if not isinstance(v, str) else v
    return out


class ModelPart:
    def __init__(self, m, parent, root):
        self.valid = False
        if m.get('visible') is False:
            return
        if not isinstance(m.get('name'), str):
            return
        self.name = m['name']
        self.parent = parent
        self.root = root
        self.is_part = True
        self.description = m.get('description', '')

        if isinstance(m.get('texture'), str):
            self.texture_name = m['texture']
            ts = point2d(m.get('texture_size'), parent.texture_size)
            s = max(ts)
            self.texture_size = (s, s)
        else:
            self.texture_name = parent.texture_name
            self.texture_size = parent.texture_size

        inherit = as_bool(m.get('color_inherit'), True)
        self.color_blend = parse_color(m.get('color_blend'), WHITE)
        self.color_alpha = float(m.get('color_alpha', 1) if m.get('color_alpha') is not None else 1)
        self.color_emissive = float(m.get('color_emissive', m.get('color_brightness', 0)) or 0)
        self.color_mix = parse_color(m.get('color_mix'), BLACK)
        self.color_mix_percent = float(m.get('color_mix_percent', 0) or 0)
        if inherit:
            self.color_blend = _color_mul(self.color_blend, parent.color_blend)
            self.color_alpha *= parent.color_alpha
            self.color_emissive = min(1.0, max(0.0, self.color_emissive + parent.color_emissive))
            self.color_mix = _color_add(self.color_mix, parent.color_mix)
            self.color_mix_percent = min(1.0, max(0.0, self.color_mix_percent + parent.color_mix_percent))

        pscale = parent.scale
        pos = point3d(m.get('position'), (0, 0, 0))
        self.position = [pos[0] * pscale[0], pos[1] * pscale[1], pos[2] * pscale[2]]
        self.rotation = point3d(m.get('rotation'), (0, 0, 0))
        sc = point3d(m.get('scale'), (1, 1, 1))
        self.scale = (sc[0] * pscale[0], sc[1] * pscale[1], sc[2] * pscale[2])
        self.floor_box_uvs = root.floor_box_uvs

        # Bloqueado a la mitad doblada del padre
        self.lock_bend = True
        if parent.is_part and parent.bend is not None:
            if m.get('lock_bend') is not None:
                self.lock_bend = as_bool(m.get('lock_bend'), True)
            if self.lock_bend:
                ax = PART_AXIS[parent.bend.part]
                self.position[ax] -= parent.bend.offset
        self.position = tuple(self.position)

        self.bend = None
        bm = m.get('bend')
        if isinstance(bm, dict) and bm.get('part') in PART_AXIS and isinstance(bm.get('offset'), (int, float)):
            b = BendInfo()
            b.part = bm['part']
            ax = PART_AXIS[b.part]
            b.offset = float(bm['offset']) * self.scale[ax]
            b.size = bm.get('size')
            if isinstance(b.size, (int, float)):
                b.size = float(b.size) * self.scale[ax]
            else:
                b.size = None
            b.end_offset = float(bm.get('end_offset', 0) or 0)
            axes = []
            av = bm.get('axis')
            if isinstance(av, str):
                av = [av]
            for a in av or []:
                if a in AXIS_NAME:
                    axes.append(AXIS_NAME[a])
            b.axis = [i in axes for i in range(3)]
            b.direction_min = _list_or_single(bm.get('direction_min'), axes, [-180.0] * 3)
            b.direction_max = _list_or_single(bm.get('direction_max'), axes, [180.0] * 3)
            inv = bm.get('invert')
            b.invert = [False] * 3
            if isinstance(inv, (bool, int, float)) and len(axes) == 1:
                b.invert = [bool(inv)] * 3
            elif isinstance(inv, list):
                for i, v in enumerate(inv):
                    if i < len(axes):
                        b.invert[axes[i]] = bool(v)
            # Direccion antigua
            dr = bm.get('direction')
            if isinstance(dr, str):
                dr = [dr]
            if isinstance(dr, list):
                for i, d in enumerate(dr):
                    if i >= len(axes):
                        break
                    a = axes[i]
                    if d == 'both':
                        b.direction_min[a], b.direction_max[a] = -180.0, 180.0
                    elif d == 'forward':
                        b.direction_min[a], b.direction_max[a] = 0.0, 180.0
                        b.invert[a] = not b.invert[a]
                    else:
                        b.direction_min[a], b.direction_max[a] = 0.0, 180.0
            b.default_angle = _list_or_single(bm.get('angle'), axes, [0.0] * 3)
            self.bend = b

        self.shapes = []
        for s in m.get('shapes') or []:
            sh = ModelShape(s, self)
            if sh.valid:
                self.shapes.append(sh)

        root.part_map[self.name] = self
        self.parts = []
        for p in m.get('parts') or []:
            part = ModelPart(p, self, root)
            if part.valid:
                self.parts.append(part)
        self.valid = True

    def _limit_bend(self, bend):
        b = self.bend
        out = [0.0, 0.0, 0.0]
        for i in range(3):
            v = bend[i]
            if v == 0:
                continue
            v = min(max(v, b.direction_min[i]), b.direction_max[i])
            if b.invert[i]:
                v = -v
            if not b.axis[i]:
                v = 0.0
            out[i] = v
        return out

    def bend_matrix(self, bend):
        """model_part_get_bend_matrix para la parte (hijos bloqueados)."""
        if self.bend is None:
            return list(gm.IDENTITY)
        bv = self._limit_bend(bend)
        pos = [0.0, 0.0, 0.0]
        pos[PART_AXIS[self.bend.part]] = self.bend.offset
        return gm.build(pos[0], pos[1], pos[2], bv[0], bv[1], bv[2], 1, 1, 1)


class ModelShape:
    def __init__(self, m, part):
        self.valid = False
        if m.get('visible') is False:
            return
        if not isinstance(m.get('type'), str) or m['type'] not in ('block', 'plane'):
            return
        if not isinstance(m.get('from'), list) or not isinstance(m.get('to'), list):
            return
        self.part = part
        self.type = m['type']
        self.description = m.get('description', '') or ''
        self.use_model_color = as_bool(m.get('use_model_color'), False)

        if isinstance(m.get('texture'), str):
            self.texture_name = m['texture']
            ts = point2d(m.get('texture_size'), part.texture_size)
            s = max(ts)
            self.texture_size = (s, s)
        else:
            self.texture_name = None  # hereda de la parte
            self.texture_size = part.texture_size

        inherit = as_bool(m.get('color_inherit'), True)
        self.color_blend = parse_color(m.get('color_blend'), WHITE)
        self.color_alpha = float(m.get('color_alpha', 1) if m.get('color_alpha') is not None else 1)
        self.color_emissive = float(m.get('color_emissive', m.get('color_brightness', 0)) or 0)
        self.color_mix = parse_color(m.get('color_mix'), BLACK)
        self.color_mix_percent = float(m.get('color_mix_percent', 0) or 0)
        if inherit:
            self.color_blend = _color_mul(self.color_blend, part.color_blend)
            self.color_alpha *= part.color_alpha
            self.color_emissive = min(1.0, max(0.0, self.color_emissive + part.color_emissive))
            self.color_mix = _color_add(self.color_mix, part.color_mix)
            self.color_mix_percent = min(1.0, max(0.0, self.color_mix_percent + part.color_mix_percent))

        self.texture_mirror = as_bool(m.get('texture_mirror'), False)
        self.invert = as_bool(m.get('invert'), False)
        self.hide_front = as_bool(m.get('hide_front'), False)
        self.hide_back = as_bool(m.get('hide_back'), as_bool(m.get('hide_backface'), False))
        self.face_camera = as_bool(m.get('face_camera'), False)
        self.floor_box_uvs = part.floor_box_uvs

        fr = list(point3d(m.get('from')))
        to = list(point3d(m.get('to')))
        if self.type == 'plane':
            to[Y] = fr[Y]
        infl = float(m.get('inflate', 0) or 0)
        inflate = [infl, infl, infl]
        self.is3d = False
        if self.type == 'plane':
            self.is3d = as_bool(m.get('3d'), False)
            if self.is3d:
                to[Y] += 1
            else:
                inflate[Y] = 0.0
        self.from_noscale = tuple(fr)
        self.to_noscale = tuple(to)
        self.inflate = tuple(inflate)

        pos = point3d(m.get('position'), (0, 0, 0))
        self.position = (pos[0] * part.scale[0], pos[1] * part.scale[1], pos[2] * part.scale[2])
        self.rotation = point3d(m.get('rotation'), (0, 0, 0))
        sc = point3d(m.get('scale'), (1, 1, 1))
        self.scale = (sc[0] * part.scale[0], sc[1] * part.scale[1], sc[2] * part.scale[2])
        self.frm = tuple((fr[i] - inflate[i]) * self.scale[i] for i in range(3))
        self.to = tuple((to[i] + inflate[i]) * self.scale[i] for i in range(3))
        self.uv = point2d(m.get('uv'), (0, 0))
        self.bend_shape = as_bool(m.get('bend'), True)
        self.valid = True

    # -- bend ---------------------------------------------------------------

    def bend_matrix(self, bendvec):
        """model_part_get_bend_matrix llamado desde la forma."""
        part = self.part
        bv = part._limit_bend(bendvec)
        ax = PART_AXIS[part.bend.part]
        pos = [0.0, 0.0, 0.0]
        pos[ax] = part.bend.offset - self.position[ax]
        mat = gm.build(pos[0], pos[1], pos[2], bv[0], bv[1], bv[2], 1, 1, 1)
        pre = gm.build(-pos[0], -pos[1], -pos[2], self.rotation[0], self.rotation[1], self.rotation[2], 1, 1, 1)
        return gm.multiply(pre, mat)

    def bend_params(self, bend_style):
        """(eje, x1, inicio, tamaño, invertido, nítido, cortes) de la zona de doblado."""
        part = self.part
        bp = part.bend.part
        if self.type == 'block':
            segaxis = PART_AXIS[bp]
        else:
            segaxis = X if bp in ('left', 'right') else Z
        bsize = part.bend.size
        sharp = bend_style == 'blocky' and bsize is None and sum(part.bend.axis) == 1
        bendsize = bsize if bsize is not None else (4.0 if bend_style == 'realistic' else 1.0)
        detail = 2.0 if sharp else max(bendsize, 2.0)
        if bsize is not None and bsize >= 1 and self.scale[segaxis] > 0.5:
            detail /= self.scale[segaxis]
        x1 = self.frm[segaxis]
        size = self.to[segaxis] - x1
        bendstart = (part.bend.offset - (self.position[segaxis] + x1)) - bendsize / 2
        invangle = bp in ('lower', 'back', 'left')
        n = max(int(math.ceil(detail)), 4)
        cuts = []
        for k in range(n + 1):
            t = bendstart + bendsize * k / n
            if 1e-5 < t < size - 1e-5:
                cuts.append(x1 + t)
        return segaxis, x1, bendstart, bendsize, invangle, sharp, cuts

    def rig_transform(self, bend_style):
        """Para el esqueleto: geometria sin doblar + peso de doblado por vertice."""
        rotm = gm.rotation(*self.rotation)
        fn = (lambda p: gm.transform_point(rotm, p))
        if self.part.bend is None or not self.bend_shape:
            return None, [], fn, (lambda p: 0.0)
        segaxis, x1, bendstart, bendsize, invangle, sharp, cuts = self.bend_params(bend_style)

        # Mine-imator suaviza el ángulo en X e Y (easeinoutquint) y es lineal en Z
        eased = (self.part.bend.axis[X] or self.part.bend.axis[Y]) and not sharp

        def wfn(p):
            t = p[segaxis] - x1
            w = min(max((t - bendstart) / bendsize, 0.0), 1.0)
            if invangle:
                w = 1.0 - w
            return ease('easeinoutquint', w) if eased else w
        return segaxis, cuts, fn, wfn

    def transform(self, bend, bend_style):
        """Devuelve (eje, cortes, funcion) que transforma vertices locales."""
        part = self.part
        isbent = part.bend is not None and self.bend_shape and any(abs(b) > 1e-6 for b in bend)
        if not isbent:
            rotm = gm.rotation(*self.rotation)
            return None, [], (lambda p: gm.transform_point(rotm, p))

        bp = part.bend.part
        if self.type == 'block':
            segaxis = PART_AXIS[bp]
        else:
            segaxis = X if bp in ('left', 'right') else Z
        bsize = part.bend.size
        sharp = bend_style == 'blocky' and bsize is None and sum(part.bend.axis) == 1
        bendsize = bsize if bsize is not None else (4.0 if bend_style == 'realistic' else 1.0)
        detail = 2.0 if sharp else max(bendsize, 2.0)
        if bsize is not None and bsize >= 1 and self.scale[segaxis] > 0.5:
            detail /= self.scale[segaxis]
        x1 = self.frm[segaxis]
        size = self.to[segaxis] - x1
        bendstart = (part.bend.offset - (self.position[segaxis] + x1)) - bendsize / 2
        bendend = bendstart + bendsize
        invangle = bp in ('lower', 'back', 'left')

        n = max(int(math.ceil(detail)), 4)
        cuts = []
        for k in range(n + 1):
            t = bendstart + bendsize * k / n
            if 1e-5 < t < size - 1e-5:
                cuts.append(x1 + t)

        def fn(p):
            t = p[segaxis] - x1
            if t < bendstart:
                w = 0.0
            elif t >= bendend:
                w = 1.0
            else:
                w = (t - bendstart) / bendsize
            if invangle:
                w = 1.0 - w
            if sharp:
                w = round(w) if abs(w - 0.5) > 0.001 else w
                bv = (bend[0] * w, bend[1] * w, bend[2] * w)
            else:
                e = ease('easeinoutquint', w)
                bv = (bend[0] * e, bend[1] * e, bend[2] * w)
            return gm.transform_point(self.bend_matrix(bv), p)

        return segaxis, cuts, fn


def split_poly(poly, axis, c):
    """Divide un poligono convexo por el plano coord[axis] = c."""
    vs, uvs = poly.verts, poly.uvs
    d = [v[axis] - c for v in vs]
    if all(x >= -1e-7 for x in d) or all(x <= 1e-7 for x in d):
        return [poly]
    a_v, a_t, b_v, b_t = [], [], [], []
    n = len(vs)
    for i in range(n):
        j = (i + 1) % n
        vi, vj, ti, tj, di, dj = vs[i], vs[j], uvs[i], uvs[j], d[i], d[j]
        if di <= 0:
            a_v.append(vi)
            a_t.append(ti)
        if di >= 0:
            b_v.append(vi)
            b_t.append(ti)
        if (di < 0 < dj) or (dj < 0 < di):
            f = di / (di - dj)
            pv = tuple(vi[k] + (vj[k] - vi[k]) * f for k in range(3))
            pt = (ti[0] + (tj[0] - ti[0]) * f, ti[1] + (tj[1] - ti[1]) * f)
            a_v.append(pv)
            a_t.append(pt)
            b_v.append(pv)
            b_t.append(pt)
    out = []
    for vv, tt in ((a_v, a_t), (b_v, b_t)):
        if len(vv) >= 3:
            out.append(Poly(vv, tt, poly.mat, poly.smooth))
    return out


def apply_transform(polys, axis, cuts, fn, wfn=None):
    if axis is not None:
        for c in cuts:
            nxt = []
            for p in polys:
                nxt.extend(split_poly(p, axis, c))
            polys = nxt
    out = []
    for p in polys:
        w = [wfn(v) for v in p.verts] if wfn else None
        out.append(Poly([fn(v) for v in p.verts], p.uvs, p.mat, p.smooth, w))
    return out


def gen_block(shape):
    """model_shape_generate_block (sin doblar, eje Z)."""
    x1, y1, z1 = shape.frm
    x2, y2, z2 = shape.to
    tsx, tsy = shape.texture_size
    texsize = [shape.to_noscale[i] - shape.from_noscale[i] for i in range(3)]
    uv = list(shape.uv)
    if shape.floor_box_uvs:
        texsize = [math.floor(t + 0.000001) if (t % 1) else math.floor(t) for t in texsize]
        uv = [math.floor(uv[0]), math.floor(uv[1])]
    sx = texsize[X] / tsx
    sy = texsize[Y] / tsy
    sz = texsize[Z] / tsy
    u0, v0 = uv[0] / tsx, uv[1] / tsy

    def rect(u, v, w, h):
        return [(u, v), (u + w, v), (u + w, v + h), (u, v + h)]

    east = rect(u0 + sx, v0, sy, sz)
    west = rect(u0 - sy, v0, sy, sz)
    south = rect(u0, v0, sx, sz)
    north = rect(u0 + sx + sy, v0, sx, sz)
    up = rect(u0, v0 - sy, sx, sy)
    # Abajo esta invertido verticalmente
    d4 = (u0 + sx, v0 - sy)
    down = [(d4[0], d4[1] + sy), (d4[0] + sx, d4[1] + sy), (d4[0] + sx, d4[1]), d4]

    if shape.texture_mirror:
        east, west = west, east
        sw = lambda r: [r[1], r[0], r[3], r[2]]
        east, west, south, north, up, down = sw(east), sw(west), sw(south), sw(north), sw(up), sw(down)

    p1, p2, p3, p4 = (x1, y2, z1), (x2, y2, z1), (x2, y1, z1), (x1, y1, z1)
    n1, n2, n3, n4 = (x1, y2, z2), (x2, y2, z2), (x2, y1, z2), (x1, y1, z2)

    polys = [
        quad(p1, p2, p3, p4, *down),
        quad(n2, n3, p3, p2, *east),
        quad(n4, n1, p1, p4, *west),
        quad(n1, n2, p2, p1, *south),
        quad(n3, n4, p4, p3, *north),
        quad(n4, n3, n2, n1, *up),
    ]
    if shape.invert:
        polys = [p.reversed() for p in polys]
    return polys


def _v_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _v_add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _v_scale(a, k):
    return (a[0] * k, a[1] * k, a[2] * k)


def _v_dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _v_cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _v_lerp(a, b, t):
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(len(a)))


def _bilerp(q, x, y):
    """Bilinear interpolation over a quad given as (p0, p1, p2, p3) with p0->p1 along x, p0->p3 along y."""
    return _v_lerp(_v_lerp(q[0], q[1], x), _v_lerp(q[3], q[2], x), y)


def voxelize_block(polys, alpha, tex_size, depth=1.0, threshold=1.0 / 255.0):
    """Turn the visible texels of an (inflated) block's faces into a one-texel-deep 3D shell.

    This is the "3D outer layers" feature (hat, jacket, sleeves, pants of player skins).
    `polys` are the quads from gen_block, `alpha` is TextureLib.alpha()'s (w, h, fn) with y = 0 at the
    top, `tex_size` is the model's (width, height) in texels and `depth` is the shell thickness in
    Mine-imator units (1 unit = 1 texel). Each visible surface texel keeps its exact front UVs; boundary
    edges extend inward to the uninflated body, so transparent gaps get real depth without creating
    hidden faces between adjacent texels. Winding follows the source quads' own convention.
    """
    w, h, fn = alpha
    tsx, tsy = tex_size
    if w < 1 or h < 1 or tsx <= 0 or tsy <= 0:
        return polys
    corners = [v for p in polys for v in p.verts]
    if not corners:
        return polys
    center = tuple(sum(v[i] for v in corners) / len(corners) for i in range(3))

    def visible(u, v):
        x = min(w - 1, max(0, int(math.floor(u * w))))
        y = min(h - 1, max(0, int(math.floor(v * h))))
        return fn(x, y) > threshold

    out = []
    for poly in polys:
        if len(poly.verts) != 4:
            out.append(poly)
            continue
        P, T = poly.verts, poly.uvs
        face_center = tuple(sum(v[i] for v in P) / 4.0 for i in range(3))
        n_src = _v_cross(_v_sub(P[1], P[0]), _v_sub(P[2], P[0]))
        length = math.sqrt(_v_dot(n_src, n_src))
        if length < 1e-9:
            continue
        # Convention of the source quad: does its own cross product point away from the block centre?
        want = 1.0 if _v_dot(n_src, _v_sub(face_center, center)) >= 0.0 else -1.0
        outward = _v_scale(n_src, want / length)
        inward = _v_scale(outward, -depth)
        columns = max(1, int(round(max(abs((T[1][0] - T[0][0]) * tsx), abs((T[1][1] - T[0][1]) * tsy)))))
        rows = max(1, int(round(max(abs((T[3][0] - T[0][0]) * tsx), abs((T[3][1] - T[0][1]) * tsy)))))
        occupied = set()
        for c in range(columns):
            for r in range(rows):
                u, v = _bilerp(T, (c + 0.5) / columns, (r + 0.5) / rows)
                if visible(u, v):
                    occupied.add((c, r))

        def emit(pts, uvs, side_dir=None):
            n = _v_cross(_v_sub(pts[1], pts[0]), _v_sub(pts[2], pts[0]))
            target = outward if side_dir is None else side_dir
            if _v_dot(n, target) * want < 0.0:
                pts, uvs = pts[::-1], uvs[::-1]
            out.append(Poly(list(pts), list(uvs)))

        for c, r in sorted(occupied, key=lambda k: (k[1], k[0])):
            x0, x1 = c / columns, (c + 1) / columns
            y0, y1 = r / rows, (r + 1) / rows
            front = [_bilerp(P, x0, y0), _bilerp(P, x1, y0), _bilerp(P, x1, y1), _bilerp(P, x0, y1)]
            front_uv = [_bilerp(T, x0, y0), _bilerp(T, x1, y0), _bilerp(T, x1, y1), _bilerp(T, x0, y1)]
            back = [_v_add(f, inward) for f in front]
            mid = _bilerp(T, (c + 0.5) / columns, (r + 0.5) / rows)
            solid = [mid] * 4
            cell_center = _v_scale(_v_add(front[0], front[2]), 0.5)
            out.append(Poly(front, front_uv))  # same winding as the source quad
            sides = (
                ((c - 1, r), [front[0], front[3], back[3], back[0]], _v_sub(_v_scale(_v_add(front[0], front[3]), 0.5), cell_center)),
                ((c + 1, r), [front[1], back[1], back[2], front[2]], _v_sub(_v_scale(_v_add(front[1], front[2]), 0.5), cell_center)),
                ((c, r - 1), [front[0], back[0], back[1], front[1]], _v_sub(_v_scale(_v_add(front[0], front[1]), 0.5), cell_center)),
                ((c, r + 1), [front[3], front[2], back[2], back[3]], _v_sub(_v_scale(_v_add(front[3], front[2]), 0.5), cell_center)),
            )
            for neighbour, pts, direction in sides:
                if neighbour not in occupied:
                    emit(pts, solid, direction)
    return out


def gen_plane(shape):
    """model_shape_generate_plane (sin doblar)."""
    x1, y1, z1 = shape.frm
    x2, z2 = shape.to[X], shape.to[Z]
    tsx, tsy = shape.texture_size
    texsize = [shape.to_noscale[i] - shape.from_noscale[i] for i in range(3)]
    uv = list(shape.uv)
    if shape.floor_box_uvs:
        texsize = [math.floor(t + 0.000001) if (t % 1) else math.floor(t) for t in texsize]
        uv = [math.floor(uv[0]), math.floor(uv[1])]
    sx, sz = texsize[X] / tsx, texsize[Z] / tsy
    t1 = (uv[0] / tsx, uv[1] / tsy)
    t2 = (t1[0] + sx, t1[1])
    t3 = (t1[0] + sx, t1[1] + sz)
    t4 = (t1[0], t1[1] + sz)
    if shape.texture_mirror:
        t1, t2 = t2, t1
        t3, t4 = t4, t3
    top, bottom = t1[1], t3[1]
    a = (t1[0], top)
    b = (t2[0], top)
    c = (t2[0], bottom)
    d = (t1[0], bottom)
    p1, p2 = (x1, y1, z1), (x2, y1, z1)
    n1, n2 = (x1, y1, z2), (x2, y1, z2)
    polys = []
    if not shape.hide_front:
        polys.append(quad(n1, n2, p2, p1, a, b, c, d))
    if not shape.hide_back:
        polys.append(quad(n2, n1, p1, p2, b, a, d, c))
    if shape.invert:
        polys = [p.reversed() for p in polys]
    return polys


def gen_plane_3d(shape, alpha):
    """model_shape_generate_plane_3d (sin doblar). alpha: (w, h, fn(x, y))."""
    tw, th, getalpha = alpha
    x1, y1, z1 = shape.frm
    y2 = shape.to[Y]
    tsx, tsy = shape.texture_size
    texsize = [shape.to_noscale[i] - shape.from_noscale[i] for i in range(3)]
    texsizeuv = (texsize[X] / tsx, texsize[Z] / tsy)
    texuv = (shape.uv[0] / tsx, shape.uv[1] / tsy)
    samplepos = (int(math.ceil(texuv[0] * tw - 1e-9)), int(math.ceil(texuv[1] * th - 1e-9)))
    S0 = int(math.ceil(texsizeuv[0] * tw - 1e-9))
    S1 = int(math.ceil(texsizeuv[1] * th - 1e-9))
    if S0 <= 0 or S1 <= 0:
        return []
    texscale = (math.ceil(texsize[X] - 1e-9) / S0, math.ceil(texsize[Z] - 1e-9) / S1)
    texsizescale = (tsx / texscale[0], tsy / texscale[1])
    ptex = (1.0 / texsizescale[0], 1.0 / texsizescale[1])

    def solid(ax, ay):
        return getalpha(samplepos[0] + ax, samplepos[1] + ay) >= 0.999

    infl_outer = (shape.inflate[X] / S0) * 2
    infl_inner = (shape.inflate[X] / S1) * 2
    y1p = []
    y2p = []
    segouter = 0.0
    fracx = texsize[X] % 1
    fracz = texsize[Z] % 1
    for outer in range(S0 + 1):
        segin = 0.0
        insize = 1.0
        r1, r2 = [], []
        for inner in range(S1 + 1):
            r1.append((x1 + segouter, y1, z1 + segin))
            r2.append((x1 + segouter, y2, z1 + segin))
            insize = 1.0
            if inner == S1 - 1 and fracz > 0:
                insize = fracz
            segin += (insize + infl_inner) * shape.scale[Z] * texscale[1]
        y1p.append(r1)
        y2p.append(r2)
        segouter += (insize + infl_outer) * shape.scale[X] * texscale[0]

    polys = []
    mirror = shape.texture_mirror
    for outer in range(S0):
        for inner in range(S1):
            ax = outer
            ay = S1 - 1 - inner
            if mirror:
                ax = S0 - 1 - ax
            if not solid(ax, ay):
                continue
            wf = ax == 0 or not solid(ax - 1, ay)
            ef = ax == S0 - 1 or not solid(ax + 1, ay)
            af = ay == 0 or not solid(ax, ay - 1)
            bf = ay == S1 - 1 or not solid(ax, ay + 1)
            if mirror:
                wf, ef = ef, wf
            pt = (texuv[0] + ax / texsizescale[0], texuv[1] + ay / texsizescale[1])
            t1 = pt
            t2 = (pt[0] + ptex[0], pt[1])
            t3 = (pt[0] + ptex[0], pt[1] + ptex[1])
            t4 = (pt[0], pt[1] + ptex[1])
            p1, p2 = y1p[outer][inner], y1p[outer + 1][inner]
            p3, p4 = y2p[outer + 1][inner], y2p[outer][inner]
            q1, q2 = y1p[outer][inner + 1], y1p[outer + 1][inner + 1]
            q3, q4 = y2p[outer + 1][inner + 1], y2p[outer][inner + 1]
            if ef:
                polys.append(quad(q3, q2, p2, p3, t1, t2, t3, t4))
            if wf:
                polys.append(quad(q1, q4, p4, p1, t1, t2, t3, t4))
            if not shape.hide_front:
                polys.append(quad(q4, q3, p3, p4, t2, t1, t4, t3))
            if not shape.hide_back:
                polys.append(quad(q2, q1, p1, p2, t2, t1, t4, t3))
            if af:
                polys.append(quad(q1, q2, q3, q4, t1, t2, t3, t4))
            if bf:
                polys.append(quad(p4, p3, p2, p1, t1, t2, t3, t4))
    if shape.invert:
        polys = [p.reversed() for p in polys]
    return polys


# ---------------------------------------------------------------------------
# Formas primitivas (vbuffer_create_*), radio 8
# ---------------------------------------------------------------------------

def _shape_tex(temp):
    thflip = -1 if temp.shape_tex_hmirror else 1
    tvflip = -1 if temp.shape_tex_vmirror else 1
    if temp.shape_invert:
        thflip *= -1
    t1 = [temp.shape_tex_hoffset, temp.shape_tex_voffset]
    t2 = [temp.shape_tex_hoffset + temp.shape_tex_hrepeat, temp.shape_tex_voffset + temp.shape_tex_vrepeat]
    if thflip < 0:
        t1[0] = 1.0 - t1[0]
        t2[0] = 1.0 - t2[0]
    if tvflip < 0:
        t1[1] = 1.0 - t1[1]
        t2[1] = 1.0 - t2[1]
    return t1, t2, thflip, tvflip


def gen_primitive(temp):
    t = temp.type
    t1, t2, th, tv = _shape_tex(temp)
    rad = 8.0
    inv = temp.shape_invert
    if t == 'surface':
        return _surface(rad, t1, t2, inv)
    if t == 'cube':
        return _cube(rad, t1, t2, th, tv, inv, temp.shape_tex_mapped)
    if t == 'cone':
        return _cone(rad, t1, t2, th, tv, max(3, temp.shape_detail), temp.shape_closed, inv, temp.shape_tex_mapped, temp.shape_smooth)
    if t == 'cylinder':
        return _cylinder(rad, t1, t2, th, tv, max(3, temp.shape_detail), temp.shape_closed, inv, temp.shape_tex_mapped, temp.shape_smooth)
    if t == 'sphere':
        return _sphere(rad, t1, t2, max(3, temp.shape_detail), inv, temp.shape_smooth)
    return []


def _t(*a):
    """vbuffer_add_triangle con 15/16 argumentos."""
    x1, y1, z1, x2, y2, z2, x3, y3, z3, u1, v1, u2, v2, u3, v3 = a[:15]
    inv = a[15] if len(a) > 15 else False
    return tri((x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (u1, v1), (u2, v2), (u3, v3), inv)


def _surface(r, t1, t2, inv):
    return [_t(-r, 0, r, r, 0, r, r, 0, -r, t1[0], t1[1], t2[0], t1[1], t2[0], t2[1], inv),
            _t(-r, 0, -r, -r, 0, r, r, 0, -r, t1[0], t2[1], t1[0], t1[1], t2[0], t2[1], inv)]


def _cube(r, t1, t2, th, tv, inv, mapped):
    ts = (1 / 3, 1 / 2)
    out = []

    def mp(a):
        b = [a[0] + ts[0], a[1] + ts[1]]
        a = list(a)
        if th < 0:
            a[0], b[0] = b[0], a[0]
        if tv < 0:
            a[1], b[1] = b[1], a[1]
        return a, b

    if mapped:
        t1, t2 = mp([ts[0] if inv else ts[0] * 2, 0])
    out.append(_t(r, r, -r, r, r, r, r, -r, r, t1[0], t2[1], t1[0], t1[1], t2[0], t1[1], inv))
    out.append(_t(r, r, -r, r, -r, r, r, -r, -r, t1[0], t2[1], t2[0], t1[1], t2[0], t2[1], inv))
    if mapped:
        t1, t2 = mp([ts[0] * 2 if inv else ts[0], 0])
    out.append(_t(-r, r, r, -r, r, -r, -r, -r, r, t2[0], t1[1], t2[0], t2[1], t1[0], t1[1], inv))
    out.append(_t(-r, -r, r, -r, r, -r, -r, -r, -r, t1[0], t1[1], t2[0], t2[1], t1[0], t2[1], inv))
    if mapped:
        t1, t2 = mp([0, 0])
    out.append(_t(-r, r, r, r, r, r, r, r, -r, t1[0], t1[1], t2[0], t1[1], t2[0], t2[1], inv))
    out.append(_t(-r, r, -r, -r, r, r, r, r, -r, t1[0], t2[1], t1[0], t1[1], t2[0], t2[1], inv))
    if mapped:
        t1, t2 = mp([0, ts[1]])
    out.append(_t(r, -r, r, -r, -r, r, r, -r, -r, t1[0], t1[1], t2[0], t1[1], t1[0], t2[1], inv))
    out.append(_t(-r, -r, r, -r, -r, -r, r, -r, -r, t2[0], t1[1], t2[0], t2[1], t1[0], t2[1], inv))
    if mapped:
        t1, t2 = mp([ts[0], ts[1]])
    out.append(_t(-r, -r, r, r, -r, r, -r, r, r, t1[0], t1[1], t2[0], t1[1], t1[0], t2[1], inv))
    out.append(_t(r, -r, r, r, r, r, -r, r, r, t2[0], t1[1], t2[0], t2[1], t1[0], t2[1], inv))
    if mapped:
        t1, t2 = mp([ts[0] * 2, ts[1]])
    out.append(_t(r, -r, -r, -r, -r, -r, -r, r, -r, t2[0], t2[1], t1[0], t2[1], t1[0], t1[1], inv))
    out.append(_t(r, r, -r, r, -r, -r, -r, r, -r, t2[0], t1[1], t2[0], t2[1], t1[0], t1[1], inv))
    return out


def _sphere(r, t1, t2, detail, inv, smooth):
    t1 = [t1[0] + 0.25, t1[1]]
    t2 = [t2[0] + 0.25, t2[1]]
    out = []
    tsz = (t2[0] - t1[0], t2[1] - t1[1])
    tmid_y = t1[1] + tsz[1] / 2

    def pt(a, b):
        nx = math.sin(a * math.pi * 2) * math.sin(b * math.pi)
        ny = -math.cos(a * math.pi * 2) * math.sin(b * math.pi)
        nz = -math.cos(b * math.pi)
        return (nx * r, ny * r, nz * r), (t2[0] - a * tsz[0], tmid_y - nz * (tsz[1] / 2))

    # Rejilla de puntos calculada una sola vez (cada punto lo comparten cuatro celdas)
    grid = [[pt(ii / detail, jj / (detail - 2)) for jj in range(detail - 1)] for ii in range(detail + 1)]
    for ii in range(detail):
        for jj in range(detail - 2):
            jp, j = jj / (detail - 2), (jj + 1) / (detail - 2)
            v1, u1 = grid[ii][jj]
            v2, u2 = grid[ii][jj + 1]
            v3, u3 = grid[ii + 1][jj]
            v4, u4 = grid[ii + 1][jj + 1]
            if jp > 0:
                if inv:
                    out.append(Poly([v3, v1, v4], [u3, u1, u4], smooth=smooth))
                else:
                    out.append(Poly([v1, v3, v4], [u1, u3, u4], smooth=smooth))
            if j < 1 - 1e-9:
                if inv:
                    out.append(Poly([v4, v1, v2], [u4, u1, u2], smooth=smooth))
                else:
                    out.append(Poly([v1, v4, v2], [u1, u4, u2], smooth=smooth))
    return out


def _cylinder(r, t1, t2, th, tv, detail, closed, inv, mapped, smooth):
    t1 = [t1[0] + 0.25, t1[1]]
    t2 = [t2[0] + 0.25, t2[1]]
    out = []
    for ii in range(detail):
        ip, i = ii / detail, (ii + 1) / detail
        tsz = [t2[0] - t1[0], t2[1] - t1[1]]
        tmid = [t1[0] + tsz[0] / 2, t1[1] + tsz[1] / 2]
        c1, s1 = math.cos(ip * math.pi * 2), math.sin(ip * math.pi * 2)
        c2, s2 = math.cos(i * math.pi * 2), math.sin(i * math.pi * 2)
        x1, y1 = c1 * r, -s1 * r
        x2, y2 = c2 * r, -s2 * r
        if mapped:
            tsz = [(1 / 3) * th, tv]
            tmid[1] = tsz[1] / 2
        if closed:
            if mapped:
                tmid[0] = 5 / 6
            out.append(_t(0, 0, -r, x1, y1, -r, x2, y2, -r,
                          tmid[0], tmid[1], tmid[0] + c1 * (tsz[0] / 2), tmid[1] + s1 * (tsz[1] / 2),
                          tmid[0] + c2 * (tsz[0] / 2), tmid[1] + s2 * (tsz[1] / 2), inv))
            if mapped:
                tmid[0] = 1 / 2
            out.append(_t(0, 0, r, x2, y2, r, x1, y1, r,
                          tmid[0], tmid[1], tmid[0] + c2 * (tsz[0] / 2), tmid[1] - s2 * (tsz[1] / 2),
                          tmid[0] + c1 * (tsz[0] / 2), tmid[1] - s1 * (tsz[1] / 2), inv))
        a = list(t1)
        if mapped:
            a = [0, 0]
            b = [abs(tsz[0]), abs(tsz[1])]
            if th < 0:
                a[0], b[0] = b[0], a[0]
            if tv < 0:
                a[1], b[1] = b[1], a[1]
        ua, ub = a[0] + tsz[0] * ip, a[0] + tsz[0] * i
        vt, vb = a[1], a[1] + tsz[1]
        if inv:
            out.append(Poly([(x1, y1, r), (x1, y1, -r), (x2, y2, r)], [(ua, vt), (ua, vb), (ub, vt)], smooth=smooth))
            out.append(Poly([(x2, y2, -r), (x2, y2, r), (x1, y1, -r)], [(ub, vb), (ub, vt), (ua, vb)], smooth=smooth))
        else:
            out.append(Poly([(x1, y1, -r), (x1, y1, r), (x2, y2, r)], [(ua, vb), (ua, vt), (ub, vt)], smooth=smooth))
            out.append(Poly([(x2, y2, r), (x2, y2, -r), (x1, y1, -r)], [(ub, vt), (ub, vb), (ua, vb)], smooth=smooth))
        if mapped:
            t1 = a
    return out


def _cone(r, t1, t2, th, tv, detail, closed, inv, mapped, smooth):
    t1 = [t1[0] + 0.25, t1[1]]
    t2 = [t2[0] + 0.25, t2[1]]
    out = []
    for ii in range(detail):
        ip, i = ii / detail, (ii + 1) / detail
        tsz = [t2[0] - t1[0], t2[1] - t1[1]]
        tmid = [t1[0] + tsz[0] / 2, t1[1] + tsz[1] / 2]
        c1, s1 = math.cos(ip * math.pi * 2), math.sin(ip * math.pi * 2)
        c2, s2 = math.cos(i * math.pi * 2), math.sin(i * math.pi * 2)
        x1, y1 = c1 * r, -s1 * r
        x2, y2 = c2 * r, -s2 * r
        if mapped:
            tsz = [0.5 * th, tv]
            tmid[1] = tsz[1] / 2
        if closed:
            if mapped:
                tmid[0] = 3 / 4
            out.append(_t(0, 0, -r, x1, y1, -r, x2, y2, -r,
                          tmid[0], tmid[1], tmid[0] + c1 * (tsz[0] / 2), tmid[1] + s1 * (tsz[1] / 2),
                          tmid[0] + c2 * (tsz[0] / 2), tmid[1] + s2 * (tsz[1] / 2), inv))
        a = list(t1)
        if mapped:
            a = [0, 0]
            b = [abs(tsz[0]), abs(tsz[1])]
            if th < 0:
                a[0], b[0] = b[0], a[0]
            if tv < 0:
                a[1], b[1] = b[1], a[1]
        ua, ub = a[0] + tsz[0] * ip, a[0] + tsz[0] * i
        vt, vb = a[1], a[1] + tsz[1]
        if inv:
            out.append(Poly([(x2, y2, -r), (0, 0, r), (x1, y1, -r)], [(ub, vb), (ub, vt), (ua, vb)], smooth=smooth))
        else:
            out.append(Poly([(0, 0, r), (x2, y2, -r), (x1, y1, -r)], [(ub, vt), (ub, vb), (ua, vb)], smooth=smooth))
        if mapped:
            t1 = a
    return out


# ---------------------------------------------------------------------------
# Items (render_generate_item + vbuffer_add_pixels)
# ---------------------------------------------------------------------------

def gen_item(img_w, img_h, getalpha, is3d, sheet_size=None, slot=0):
    if sheet_size:
        cols, rows = max(1, int(sheet_size[0])), max(1, int(sheet_size[1]))
        slotpos = (slot % cols, slot // cols)
        slotsize = (max(1.0, img_w / cols), max(1.0, img_h / rows))
        slottexsize = (1.0 / cols, 1.0 / rows)
        slottex = (slotpos[0] * slottexsize[0], slotpos[1] * slottexsize[1])
        texpos = (slotpos[0] * slotsize[0], slotpos[1] * slotsize[1])
    else:
        slotsize = (float(img_w), float(img_h))
        slottexsize = (1.0, 1.0)
        slottex = (0.0, 0.0)
        texpos = (0.0, 0.0)

    size = [16.0, 1.0 if is3d else 0.0, 16.0]
    if slotsize[0] > slotsize[1]:
        size[2] *= slotsize[1] / slotsize[0]
    elif slotsize[1] > slotsize[0]:
        size[0] *= slotsize[0] / slotsize[1]
    scale = (size[0] / slotsize[0], 1.0, size[2] / slotsize[1])

    s = slottex
    e = (slottex[0] + slottexsize[0], slottex[1] + slottexsize[1])
    polys = [
        quad((0, size[1], size[2]), (size[0], size[1], size[2]), (size[0], size[1], 0), (0, size[1], 0),
             (s[0], s[1]), (e[0], s[1]), (e[0], e[1]), (s[0], e[1])),
        quad((size[0], 0, size[2]), (0, 0, size[2]), (0, 0, 0), (size[0], 0, 0),
             (e[0], s[1]), (s[0], s[1]), (s[0], e[1]), (e[0], e[1])),
    ]
    if not is3d:
        return polys

    sw, sh = int(math.ceil(slotsize[0])), int(math.ceil(slotsize[1]))
    bx, by = int(math.floor(texpos[0])), int(math.floor(texpos[1]))

    def solid(xx, yy):
        return getalpha(bx + xx, by + yy) >= 0.999

    tpx = (1.0 / img_w, 1.0 / img_h)

    def sc(p):
        return (p[0] * scale[0], p[1] * scale[1], p[2] * scale[2])

    px = 0.0
    for xx in range(sw):
        pz = slotsize[1]
        for yy in range(sh):
            if not solid(xx, yy):
                pz -= 1
                continue
            wf = xx == 0 or not solid(xx - 1, yy)
            ef = xx == sw - 1 or not solid(xx + 1, yy)
            af = yy == 0 or not solid(xx, yy - 1)
            bf = yy == sh - 1 or not solid(xx, yy + 1)
            u, v = (bx + xx) * tpx[0], (by + yy) * tpx[1]
            t1, t2, t3, t4 = (u, v), (u + tpx[0], v), (u + tpx[0], v + tpx[1]), (u, v + tpx[1])
            if ef:
                polys.append(quad(sc((px + 1, 1, pz - 1)), sc((px + 1, 1, pz)), sc((px + 1, 0, pz)), sc((px + 1, 0, pz - 1)), t1, t2, t3, t4))
            if wf:
                polys.append(quad(sc((px, 0, pz - 1)), sc((px, 0, pz)), sc((px, 1, pz)), sc((px, 1, pz - 1)), t1, t2, t3, t4))
            if af:
                polys.append(quad(sc((px, 1, pz)), sc((px, 0, pz)), sc((px + 1, 0, pz)), sc((px + 1, 1, pz)), t1, t2, t3, t4))
            if bf:
                polys.append(quad(sc((px, 0, pz - 1)), sc((px, 1, pz - 1)), sc((px + 1, 1, pz - 1)), sc((px + 1, 0, pz - 1)), t1, t2, t3, t4))
            pz -= 1
        px += 1
    return polys


# ---------------------------------------------------------------------------
# Bloques de Minecraft (blockstates + modelos JSON)
# ---------------------------------------------------------------------------

def _match_when(when, state):
    if 'OR' in when:
        return any(_match_when(w, state) for w in when['OR'])
    if 'AND' in when:
        return all(_match_when(w, state) for w in when['AND'])
    for k, v in when.items():
        opts = str(v).lower().split('|')
        if str(state.get(k, '')).lower() not in opts:
            return False
    return True


def block_models(assets, blockstate_path, state):
    """Devuelve una lista de {model, x, y, uvlock} para el estado dado."""
    bs = assets.read_json(blockstate_path) if blockstate_path else None
    if not isinstance(bs, dict):
        return []
    out = []

    def pick(v):
        if isinstance(v, list):
            return v[0] if v else None
        return v

    if isinstance(bs.get('variants'), dict):
        best = None
        for key, v in bs['variants'].items():
            ok = True
            if key:
                for pair in key.split(','):
                    if '=' not in pair:
                        continue
                    k, val = pair.split('=', 1)
                    if k in state and str(state[k]).lower() != val.lower():
                        ok = False
                        break
            if ok:
                best = pick(v)
                break
        if best is None and bs['variants']:
            best = pick(next(iter(bs['variants'].values())))
        if best:
            out.append(best)
    if isinstance(bs.get('multipart'), list):
        for case in bs['multipart']:
            when = case.get('when')
            if when is None or _match_when(when, state):
                a = pick(case.get('apply'))
                if a:
                    out.append(a)
    return out


def _load_mc_model(assets, name):
    """Resuelve la cadena de padres. Devuelve (texturas, elementos)."""
    textures = {}
    elements = None
    seen = 0
    while name and seen < 20:
        seen += 1
        n = name.replace('minecraft:', '')
        data = assets.read_json('assets/minecraft/models/%s.json' % n)
        if not isinstance(data, dict):
            break
        for k, v in (data.get('textures') or {}).items():
            textures.setdefault(k, v)
        if elements is None and isinstance(data.get('elements'), list):
            elements = data['elements']
        name = data.get('parent')
    return textures, elements or []


def _resolve_tex(textures, ref):
    seen = 0
    while isinstance(ref, str) and ref.startswith('#') and seen < 10:
        ref = textures.get(ref[1:])
        seen += 1
    if isinstance(ref, str):
        return ref.replace('minecraft:', '')
    return None


def _rot_mc(p, axis, ang, origin, rescale):
    a = math.radians(ang)
    c, s = math.cos(a), math.sin(a)
    x, y, z = p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]
    if axis == 'x':
        y, z = y * c - z * s, y * s + z * c
        if rescale:
            f = 1 / max(abs(c), 1e-6) if abs(ang) in (22.5, 45) else 1
            y, z = y * f, z * f
    elif axis == 'y':
        x, z = x * c + z * s, -x * s + z * c
        if rescale:
            f = 1 / max(abs(c), 1e-6) if abs(ang) in (22.5, 45) else 1
            x, z = x * f, z * f
    else:
        x, y = x * c - y * s, x * s + y * c
        if rescale:
            f = 1 / max(abs(c), 1e-6) if abs(ang) in (22.5, 45) else 1
            x, y = x * f, y * f
    return (x + origin[0], y + origin[1], z + origin[2])


def _rot90(p, axis, deg):
    """Rotacion de variante de blockstate (sentido horario visto desde +eje)."""
    return _rot_mc(p, axis, -deg, (8, 8, 8), False)


FACE_DEFAULT_UV = {
    'down': lambda f, t: (f[0], 16 - t[2], t[0], 16 - f[2]),
    'up': lambda f, t: (f[0], f[2], t[0], t[2]),
    'north': lambda f, t: (16 - t[0], 16 - t[1], 16 - f[0], 16 - f[1]),
    'south': lambda f, t: (f[0], 16 - t[1], t[0], 16 - f[1]),
    'west': lambda f, t: (f[2], 16 - t[1], t[2], 16 - f[1]),
    'east': lambda f, t: (16 - t[2], 16 - t[1], 16 - f[2], 16 - f[1]),
}


def _face_corners(face, f, t):
    """Esquinas (arriba-izq, arriba-der, abajo-der, abajo-izq) vistas desde fuera."""
    x1, y1, z1 = f
    x2, y2, z2 = t
    return {
        'north': [(x2, y2, z1), (x1, y2, z1), (x1, y1, z1), (x2, y1, z1)],
        'south': [(x1, y2, z2), (x2, y2, z2), (x2, y1, z2), (x1, y1, z2)],
        'east': [(x2, y2, z2), (x2, y2, z1), (x2, y1, z1), (x2, y1, z2)],
        'west': [(x1, y2, z1), (x1, y2, z2), (x1, y1, z2), (x1, y1, z1)],
        'up': [(x1, y2, z1), (x2, y2, z1), (x2, y2, z2), (x1, y2, z2)],
        'down': [(x1, y1, z2), (x2, y1, z2), (x2, y1, z1), (x1, y1, z1)],
    }[face]


def gen_block_mc(assets, blockstate_path, state, tex_cb, tint_cb):
    """Genera los poligonos de un bloque en coordenadas de Mine-imator.

    tex_cb(texname) -> (TexRef, frames) ; tint_cb(texname, tintindex) -> color o None
    Devuelve lista de (Poly, texname, tint).
    """
    out = []
    for ap in block_models(assets, blockstate_path, state):
        mname = ap.get('model')
        if not mname:
            continue
        textures, elements = _load_mc_model(assets, mname)
        vx, vy = int(ap.get('x', 0) or 0), int(ap.get('y', 0) or 0)
        for el in elements:
            f = el.get('from', [0, 0, 0])
            t = el.get('to', [16, 16, 16])
            rot = el.get('rotation')
            for fname, fd in (el.get('faces') or {}).items():
                if fname not in FACE_DEFAULT_UV or not isinstance(fd, dict):
                    continue
                texname = _resolve_tex(textures, fd.get('texture'))
                if not texname:
                    continue
                uv = fd.get('uv') or FACE_DEFAULT_UV[fname](f, t)
                u1, v1, u2, v2 = [float(c) / 16.0 for c in uv]
                corner_uv = [(u1, v1), (u2, v1), (u2, v2), (u1, v2)]
                r = int(fd.get('rotation', 0) or 0) // 90 % 4
                corner_uv = corner_uv[-r:] + corner_uv[:-r] if r else corner_uv
                pts = _face_corners(fname, f, t)
                if isinstance(rot, dict):
                    pts = [_rot_mc(p, rot.get('axis', 'y'), float(rot.get('angle', 0)),
                                   rot.get('origin', [8, 8, 8]), as_bool(rot.get('rescale'), False)) for p in pts]
                if vx:
                    pts = [_rot90(p, 'x', vx) for p in pts]
                if vy:
                    pts = [_rot90(p, 'y', vy) for p in pts]
                frames = tex_cb(texname)
                if frames and frames > 1:
                    corner_uv = [(u, v / frames) for (u, v) in corner_uv]
                # Contra-reloj (vista exterior) en Minecraft -> MI: intercambiar Y/Z
                ccw = [pts[0], pts[3], pts[2], pts[1]]
                cuv = [corner_uv[0], corner_uv[3], corner_uv[2], corner_uv[1]]
                mi_pts = [(p[0], p[2], p[1]) for p in ccw]
                poly = Poly(mi_pts[::-1], cuv[::-1])
                tint = tint_cb(texname, fd.get('tintindex', -1)) if fd.get('tintindex', -1) is not None and fd.get('tintindex', -1) >= 0 else None
                out.append((poly, texname, tint))
    return out
