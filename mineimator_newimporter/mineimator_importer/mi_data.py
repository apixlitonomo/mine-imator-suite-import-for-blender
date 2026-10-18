# Lectura de proyectos .miproject y evaluacion de timelines (valores,
# keyframes, herencia y matrices) tal y como lo hace Mine-imator.
#
# Todas las coordenadas de este modulo estan en el espacio interno de
# Mine-imator (Z arriba, sistema levogiro). La conversion a Blender se hace
# en mi_builder.

import json
import math
import os

from . import gm_math as gm

# ---------------------------------------------------------------------------
# Versiones
# ---------------------------------------------------------------------------

# Numeros de formato de proyecto (enums.gml -> e_project)
FORMAT_110_PRE_1 = 24
FORMAT_120_PRE_1 = 28
FORMAT_200_PRE_1 = 33
FORMAT_200_PRE_5 = 34

VERSIONS = (
    ('AUTO', "Detectar automáticamente", "Usa el número de formato guardado en el proyecto"),
    ('MI_110', "Mine-imator 1.1.x", "Proyectos JSON de 1.1.0 a 1.1.5 (formato 24-27)"),
    ('MI_120', "Mine-imator 1.2.x", "Proyectos de 1.2.0 a 1.2.9 (formato 28-32)"),
    ('MI_200', "Mine-imator 2.0.x", "Proyectos de 2.0.0 y 2.0.x (formato 33-34)"),
    ('CB', "Mine-imator Community Build", "Builds de la comunidad basadas en 2.0"),
    ('VEXEL', "Mine-imator Vexel Build", "Vexel Build 1.x (formato 35 o superior)"),
)

# Formato representativo para cada version elegida a mano
VERSION_FORMAT = {
    'MI_110': 26,
    'MI_120': 32,
    'MI_200': FORMAT_200_PRE_5,
    'CB': FORMAT_200_PRE_5,
    'VEXEL': 39,
}


def describe_format(fmt, created_in=""):
    if created_in:
        return created_in
    if fmt < FORMAT_110_PRE_1:
        return "Mine-imator anterior a 1.1 (formato binario)"
    if fmt < FORMAT_120_PRE_1:
        return "Mine-imator 1.1.x"
    if fmt < FORMAT_200_PRE_1:
        return "Mine-imator 1.2.x"
    if fmt <= FORMAT_200_PRE_5:
        return "Mine-imator 2.0.x"
    return "Mine-imator Vexel/Community Build (formato %d)" % fmt


# ---------------------------------------------------------------------------
# Valores
# ---------------------------------------------------------------------------

WHITE = (1.0, 1.0, 1.0)
BLACK = (0.0, 0.0, 0.0)

DEFAULTS = {
    'POS_X': 0.0, 'POS_Y': 0.0, 'POS_Z': 0.0,
    'ROT_X': 0.0, 'ROT_Y': 0.0, 'ROT_Z': 0.0,
    'SCA_X': 1.0, 'SCA_Y': 1.0, 'SCA_Z': 1.0,
    'BEND_ANGLE_X': 0.0, 'BEND_ANGLE_Y': 0.0, 'BEND_ANGLE_Z': 0.0,
    'ALPHA': 1.0,
    'RGB_ADD': BLACK, 'RGB_SUB': BLACK, 'RGB_MUL': WHITE,
    'HSB_ADD': BLACK, 'HSB_SUB': BLACK, 'HSB_MUL': WHITE,
    'MIX_COLOR': BLACK, 'MIX_PERCENT': 0.0,
    'GLOW_COLOR': WHITE,
    'EMISSIVE': 0.0, 'METALLIC': 0.0, 'ROUGHNESS': 1.0,
    'SUBSURFACE': 0.0, 'SUBSURFACE_COLOR': WHITE,
    'VISIBLE': 1.0,
    'TEXTURE_OBJ': None, 'TEXTURE_MATERIAL_OBJ': None, 'TEXTURE_NORMAL_OBJ': None,
    'ITEM_SLOT': 0, 'CUSTOM_ITEM_SLOT': 0.0,
    'TRANSITION': 'linear',
    'TEXT': '',
    'CAM_FOV': 45.0,
    'LIGHT_COLOR': WHITE, 'LIGHT_STRENGTH': 1.0, 'LIGHT_RANGE': 250.0,
    'LIGHT_SPOT_RADIUS': 50.0, 'LIGHT_SIZE': 2.0,
}

COLOR_VALUES = {
    'RGB_ADD', 'RGB_SUB', 'RGB_MUL', 'HSB_ADD', 'HSB_SUB', 'HSB_MUL', 'MIX_COLOR',
    'GLOW_COLOR', 'SUBSURFACE_COLOR', 'LIGHT_COLOR', 'TEXT_OUTLINE_COLOR',
}

# Valores que no se interpolan (tl_value_interpolate)
STEP_VALUES = {
    'VISIBLE', 'TEXTURE_OBJ', 'TEXTURE_MATERIAL_OBJ', 'TEXTURE_NORMAL_OBJ',
    'CUSTOM_ITEM_SLOT', 'TRANSITION', 'TEXT', 'TEXT_FONT', 'IK_TARGET',
    'IK_TARGET_ANGLE', 'PATH_OBJ', 'SOUND_OBJ', 'EASE_IN_X', 'EASE_IN_Y',
    'EASE_OUT_X', 'EASE_OUT_Y', 'SPAWN', 'FREEZE', 'CLEAR', 'ATTRACTOR',
}

# Nombres antiguos (project_load_values_update_name)
LEGACY_VALUE_NAMES = {
    'BRIGHTNESS': 'EMISSIVE',
    'BG_SKY_CLOUDS_HEIGHT': 'BG_SKY_CLOUDS_Z',
}


def parse_color(val, default=WHITE):
    if isinstance(val, str):
        s = val.strip().lstrip('#')
        if len(s) >= 6:
            try:
                return (int(s[0:2], 16) / 255.0, int(s[2:4], 16) / 255.0, int(s[4:6], 16) / 255.0)
            except ValueError:
                return default
        return default
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        c = int(val)  # Color de GameMaker (BGR)
        return ((c & 255) / 255.0, ((c >> 8) & 255) / 255.0, ((c >> 16) & 255) / 255.0)
    return default


def save_id(val):
    """Convierte un id guardado ('null', 'default', 'XYZ...') a None/str."""
    if val is None or val == 'null' or val == '':
        return None
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return None if val <= 0 else str(val)
    return str(val)


def point3d(val, default=(0.0, 0.0, 0.0)):
    """value_get_point3D: intercambia Y y Z al leer."""
    if isinstance(val, (list, tuple)) and len(val) >= 3:
        return (float(val[0]), float(val[2]), float(val[1]))
    return tuple(default)


def point2d(val, default=(0.0, 0.0)):
    if isinstance(val, (list, tuple)) and len(val) >= 2:
        return (float(val[0]), float(val[1]))
    return tuple(default)


def as_bool(val, default=False):
    if val is None:
        return default
    if isinstance(val, str):
        return val.lower() in ('true', '1')
    return bool(val)


def load_values(src, dst, load_format):
    if not isinstance(src, dict):
        return
    for key, val in src.items():
        if load_format < FORMAT_200_PRE_5:
            key = LEGACY_VALUE_NAMES.get(key, key)
        if key in COLOR_VALUES:
            dst[key] = parse_color(val, dst.get(key, WHITE))
        elif key in ('TEXTURE_OBJ', 'TEXTURE_MATERIAL_OBJ', 'TEXTURE_NORMAL_OBJ', 'IK_TARGET', 'IK_TARGET_ANGLE', 'PATH_OBJ', 'SOUND_OBJ', 'TEXT_FONT', 'ATTRACTOR'):
            dst[key] = save_id(val)
        elif val == 'null':
            dst[key] = None
        else:
            dst[key] = val


# ---------------------------------------------------------------------------
# Easing (ease.gml)
# ---------------------------------------------------------------------------

def _bounce_out(x):
    if x < 1 / 2.75:
        return 7.5625 * x * x
    if x < 2 / 2.75:
        x -= 1.5 / 2.75
        return 7.5625 * x * x + 0.75
    if x < 2.5 / 2.75:
        x -= 2.25 / 2.75
        return 7.5625 * x * x + 0.9375
    x -= 2.625 / 2.75
    return 7.5625 * x * x + 0.984375


def ease(func, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    x2 = x * 2
    xm1 = x - 1
    f = (func or 'linear').lower()
    if f == 'linear':
        return x
    if f == 'instant':
        return 0.0
    if f == 'easeinquad':
        return x * x
    if f == 'easeoutquad':
        return -x * (x - 2)
    if f == 'easeinoutquad':
        return 0.5 * x2 * x2 if x2 < 1 else -0.5 * ((x2 - 1) * (x2 - 3) - 1)
    if f == 'easeincubic':
        return x ** 3
    if f == 'easeoutcubic':
        return xm1 ** 3 + 1
    if f == 'easeinoutcubic':
        return 0.5 * x2 ** 3 if x2 < 1 else 0.5 * ((x2 - 2) ** 3 + 2)
    if f == 'easeinquart':
        return x ** 4
    if f == 'easeoutquart':
        return -(xm1 ** 4 - 1)
    if f == 'easeinoutquart':
        return 0.5 * x2 ** 4 if x2 < 1 else -0.5 * ((x2 - 2) ** 4 - 2)
    if f == 'easeinquint':
        return x ** 5
    if f == 'easeoutquint':
        return xm1 ** 5 + 1
    if f == 'easeinoutquint':
        return 0.5 * x2 ** 5 if x2 < 1 else 0.5 * ((x2 - 2) ** 5 + 2)
    if f == 'easeinsine':
        return -math.cos(x * math.pi / 2) + 1
    if f == 'easeoutsine':
        return math.sin(x * math.pi / 2)
    if f == 'easeinoutsine':
        return -0.5 * (math.cos(math.pi * x) - 1)
    if f == 'easeinexpo':
        return 2 ** (10 * (x - 1))
    if f == 'easeoutexpo':
        return -(2 ** (-10 * x)) + 1
    if f == 'easeinoutexpo':
        return 0.5 * 2 ** (10 * (x2 - 1)) if x2 < 1 else 0.5 * (-(2 ** (-10 * (x2 - 1))) + 2)
    if f == 'easeincirc':
        return -(math.sqrt(1 - x * x) - 1)
    if f == 'easeoutcirc':
        return math.sqrt(1 - xm1 * xm1)
    if f == 'easeinoutcirc':
        if x2 < 1:
            return -0.5 * (math.sqrt(1 - x2 * x2) - 1)
        return 0.5 * (math.sqrt(max(0, 1 - (x2 - 2) ** 2)) + 1)
    if f == 'easeinelastic':
        p = 0.3
        s = p / (2 * math.pi) * math.asin(1)
        return -(2 ** (10 * (x - 1)) * math.sin(((x - 1) - s) * (2 * math.pi) / p))
    if f == 'easeoutelastic':
        p = 0.3
        s = p / (2 * math.pi) * math.asin(1)
        return 2 ** (-10 * x) * math.sin((x - s) * (2 * math.pi) / p) + 1
    if f == 'easeinoutelastic':
        p = 0.45
        s = p / (2 * math.pi) * math.asin(1)
        if x2 < 1:
            return -0.5 * (2 ** (10 * (x2 - 1)) * math.sin(((x2 - 1) - s) * (2 * math.pi) / p))
        return 2 ** (-10 * (x2 - 1)) * math.sin(((x2 - 1) - s) * (2 * math.pi) / p) * 0.5 + 1
    if f == 'easeinback':
        s = 1.70158
        return x * x * ((s + 1) * x - s)
    if f == 'easeoutback':
        s = 1.70158
        return xm1 * xm1 * ((s + 1) * xm1 + s) + 1
    if f == 'easeinoutback':
        s = 1.70158 * 1.525
        if x2 < 1:
            return 0.5 * (x2 * x2 * ((s + 1) * x2 - s))
        return 0.5 * ((x2 - 2) ** 2 * ((s + 1) * (x2 - 2) + s) + 2)
    if f == 'easeinbounce':
        return 1 - _bounce_out(1 - x)
    if f == 'easeoutbounce':
        return _bounce_out(x)
    if f == 'easeinoutbounce':
        if x < 0.5:
            return (1 - _bounce_out(1 - x * 2)) * 0.5
        return _bounce_out(x * 2 - 1) * 0.5 + 0.5
    return x


def interpolate(key, p, a, b):
    if key in STEP_VALUES or a is None or b is None:
        return a
    if key in COLOR_VALUES:
        p = min(max(p, 0.0), 1.0)
        return tuple(a[i] + (b[i] - a[i]) * p for i in range(3))
    if key == 'ITEM_SLOT':
        return int(math.floor(a + p * (b - a)))
    if isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or isinstance(b, str):
        return a
    try:
        return a + p * (b - a)
    except TypeError:
        return a


# ---------------------------------------------------------------------------
# Objetos del proyecto
# ---------------------------------------------------------------------------

TYPE_ALIASES = {
    'character': 'char', 'special_block': 'spblock', 'specialblock': 'spblock',
    'particle_spawner': 'particles', 'spot_light': 'spotlight', 'point_light': 'pointlight',
    'path_point': 'pathpoint',
}

SHAPE_TYPES = ('cube', 'cone', 'cylinder', 'sphere', 'surface')
MODEL_TYPES = ('char', 'spblock', 'model', 'bodypart')


def norm_type(t):
    t = (t or '').lower()
    return TYPE_ALIASES.get(t, t)


class Resource:
    def __init__(self, data):
        self.id = save_id(data.get('id'))
        self.type = (data.get('type') or '').lower()
        self.filename = data.get('filename') or ''
        self.item_sheet_size = point2d(data.get('item_sheet_size'), (16, 16))
        self.player_skin = as_bool(data.get('player_skin'), False)
        self.data = data


class Template:
    def __init__(self, data, load_format):
        self.id = save_id(data.get('id'))
        self.type = norm_type(data.get('type'))
        self.name = data.get('name') or ''
        self.data = data
        d = data

        # Modelos
        if self.type in ('char', 'spblock', 'bodypart'):
            key = 'skin' if load_format == FORMAT_110_PRE_1 and 'skin' in d else 'model_tex'
            self.model_tex = save_id(d.get(key))
            self.model_use_blend_color = as_bool(d.get('model_use_blend_color'), False)
            self.model_blend_color = parse_color(d.get('model_blend_color'), WHITE)
            m = d.get('model') or {}
            self.model_name = m.get('name', '') if isinstance(m, dict) else ''
            self.model_state = parse_state(m.get('state') if isinstance(m, dict) else None)
            self.model_part_name = m.get('part_name', '') if isinstance(m, dict) else ''
            self.model_res = None
        elif self.type == 'model':
            self.model_res = save_id(d.get('model'))
            self.model_tex = save_id(d.get('model_tex'))
            self.model_name = ''
            self.model_state = {}
            self.model_use_blend_color = False
            self.model_blend_color = WHITE
            self.model_part_name = ''
        else:
            self.model_res = None
            self.model_tex = None
            self.model_name = ''
            self.model_state = {}
            self.model_part_name = ''

        # Items
        it = d.get('item') if isinstance(d.get('item'), dict) else {}
        self.item_tex = save_id(it.get('tex', 'default'))
        self.item_name = it.get('name') if isinstance(it.get('name'), str) else None
        if self.item_name and load_format < FORMAT_120_PRE_1:
            self.item_name = self.item_name.replace('items/', 'item/')
        self.item_slot = int(it.get('slot', 0) or 0)
        self.item_3d = as_bool(it.get('3d'), True)

        # Bloques
        bl = d.get('block') if isinstance(d.get('block'), dict) else {}
        self.block_name = bl.get('name', '') if isinstance(bl.get('name'), str) else ''
        self.block_state = parse_state(bl.get('state'))
        self.block_tex = save_id(bl.get('tex', 'default'))
        self.block_repeat_enable = as_bool(bl.get('repeat_enable'), False)
        self.block_repeat = point3d(bl.get('repeat'), (1, 1, 1))
        self.scenery = save_id(d.get('scenery'))

        # Formas
        sh = d.get('shape') if isinstance(d.get('shape'), dict) else {}
        self.shape_tex = save_id(sh.get('tex'))
        self.shape_tex_mapped = as_bool(sh.get('tex_mapped'), False)
        if self.type not in ('cube', 'cylinder', 'cone'):
            self.shape_tex_mapped = False
        self.shape_tex_hoffset = float(sh.get('tex_hoffset', 0) or 0)
        self.shape_tex_voffset = float(sh.get('tex_voffset', 0) or 0)
        self.shape_tex_hrepeat = float(sh.get('tex_hrepeat', 1) if sh.get('tex_hrepeat') is not None else 1)
        self.shape_tex_vrepeat = float(sh.get('tex_vrepeat', 1) if sh.get('tex_vrepeat') is not None else 1)
        self.shape_tex_hmirror = as_bool(sh.get('tex_hmirror'), False)
        self.shape_tex_vmirror = as_bool(sh.get('tex_vmirror'), False)
        self.shape_closed = as_bool(sh.get('closed'), True)
        self.shape_invert = as_bool(sh.get('invert'), False)
        self.shape_detail = int(sh.get('detail', 32) or 32)
        self.shape_smooth = as_bool(sh.get('smooth'), True)

        # Texto
        tx = d.get('text') if isinstance(d.get('text'), dict) else {}
        self.text_3d = as_bool(tx.get('3d'), False)

    def rot_point(self):
        """temp_update_rot_point"""
        rp = [0.0, 0.0, 0.0]
        rep = self.block_repeat if self.block_repeat_enable else (1, 1, 1)
        if self.type == 'block':
            rp[0] = rep[0] * 16 / 2
            rp[1] = rep[1] * 16 / 2
        elif self.type == 'item':
            rp[0] = 16 / 2
            rp[1] = 0.5 * (1.0 if self.item_3d else 0.0)
        elif self.type == 'text':
            rp[1] = 0.5 * (1.0 if self.text_3d else 0.0)
        if self.type in SHAPE_TYPES:
            rp[2] = -8.0
        return tuple(rp)


def parse_state(st):
    """Estados de modelo/bloque: dict, lista [k, v, k, v] o 'a=b,c=d'."""
    out = {}
    if isinstance(st, dict):
        for k, v in st.items():
            out[str(k)] = str(v).lower() if isinstance(v, bool) else str(v)
    elif isinstance(st, list):
        for i in range(0, len(st) - 1, 2):
            out[str(st[i])] = str(st[i + 1])
    elif isinstance(st, str) and st:
        for pair in st.split(','):
            if '=' in pair:
                k, v = pair.split('=', 1)
                out[k.strip()] = v.strip()
    return out


class Keyframe:
    __slots__ = ('position', 'values')

    def __init__(self, position, values):
        self.position = position
        self.values = values


class Timeline:
    def __init__(self, data, load_format):
        d = data
        self.id = save_id(d.get('id'))
        self.type = norm_type(d.get('type'))
        self.name = d.get('name') or ''
        self.temp = save_id(d.get('temp'))
        self.hide = as_bool(d.get('hide'), False)
        self.depth = d.get('depth', 0) or 0
        self.model_part_name = d.get('model_part_name', '') or ''
        self.part_of = save_id(d.get('part_of'))
        self.parent = save_id(d.get('parent'))
        self.parent_tree_index = d.get('parent_tree_index', 0) or 0
        self.lock_bend = as_bool(d.get('lock_bend'), True)
        self.text = d.get('text', '') if isinstance(d.get('text'), str) else ''

        inh = d.get('inherit') if isinstance(d.get('inherit'), dict) else {}
        self.inherit_position = as_bool(inh.get('position'), True)
        self.inherit_rotation = as_bool(inh.get('rotation'), True)
        self.inherit_rot_point = as_bool(inh.get('rot_point'), False)
        self.inherit_scale = as_bool(inh.get('scale'), True)
        self.inherit_alpha = as_bool(inh.get('alpha'), True)
        self.inherit_color = as_bool(inh.get('color'), True)
        self.inherit_visibility = as_bool(inh.get('visibility'), True)
        self.inherit_bend = as_bool(inh.get('bend'), False)
        self.inherit_texture = as_bool(inh.get('texture'), True)

        self.scale_resize = as_bool(d.get('scale_resize'), False)
        self.rot_point_custom = as_bool(d.get('rot_point_custom'), False)
        self.rot_point = point3d(d.get('rot_point'), (0, 0, 0))
        self.backfaces = as_bool(d.get('backfaces'), False)
        self.hq_hiding = as_bool(d.get('hq_hiding'), False)
        self.lq_hiding = as_bool(d.get('lq_hiding'), False)

        # Valores por defecto: globales + los guardados en el timeline
        self.value_default = dict(DEFAULTS)
        load_values(d.get('default_values'), self.value_default, load_format)

        self.keyframes = []
        kfs = d.get('keyframes')
        if isinstance(kfs, dict):
            for pos, vals in kfs.items():
                try:
                    frame = int(float(pos))
                except ValueError:
                    continue
                v = dict(self.value_default)
                load_values(vals, v, load_format)
                self.keyframes.append(Keyframe(frame, v))
            self.keyframes.sort(key=lambda k: k.position)

        self.children = []

    def values_at(self, frame):
        kfs = self.keyframes
        if not kfs:
            return self.value_default
        cur = None
        nxt = None
        for kf in kfs:
            if kf.position <= frame:
                cur = kf
            else:
                nxt = kf
                break
        if cur is None:
            return nxt.values
        if nxt is None:
            return cur.values
        span = nxt.position - cur.position
        p = (frame - cur.position) / span if span > 0 else 1.0
        p = ease(cur.values.get('TRANSITION', 'linear'), p)
        out = {}
        a, b = cur.values, nxt.values
        for k in a:
            out[k] = interpolate(k, p, a[k], b.get(k, a[k]))
        return out

    def keyframe_positions(self):
        return [k.position for k in self.keyframes]


class Project:
    def __init__(self, path, version='AUTO'):
        self.path = path
        self.folder = os.path.dirname(os.path.abspath(path))
        ext = os.path.splitext(path)[1].lower()
        if ext in ('.mproj', '.mani'):
            raise ValueError("Los proyectos binarios anteriores a Mine-imator 1.1 (.mproj/.mani) no están soportados. "
                             "Ábrelo y guárdalo con Mine-imator 1.1 o superior para convertirlo a .miproject.")
        with open(path, 'r', encoding='utf-8-sig') as f:
            root = json.load(f)
        if ext == '.mimodel':
            root = _mimodel_as_project(path, root)
        elif ext == '.miobject' and not root.get('created_in'):
            root['created_in'] = 'Objeto de Mine-imator (.miobject)'
        if not isinstance(root.get('timelines'), list) and not isinstance(root.get('templates'), list):
            raise ValueError("El archivo no parece un proyecto, objeto o modelo de Mine-imator: " + os.path.basename(path))
        self.root = root
        self.file_format = int(root.get('format', FORMAT_200_PRE_5) or FORMAT_200_PRE_5)
        self.created_in = root.get('created_in', '') or ''
        self.version = version
        self.load_format = self.file_format if version == 'AUTO' else VERSION_FORMAT.get(version, self.file_format)
        self.version_label = describe_format(self.file_format, self.created_in)

        proj = root.get('project') or {}
        self.name = proj.get('name') or os.path.splitext(os.path.basename(path))[0]
        self.tempo = proj.get('tempo', 24) or 24
        self.render = root.get('render') or {}
        self.background = root.get('background') or {}

        self.resources = {}
        for r in root.get('resources') or []:
            res = Resource(r)
            if res.id:
                self.resources[res.id] = res

        self.templates = {}
        for t in root.get('templates') or []:
            tp = Template(t, self.load_format)
            if tp.id:
                self.templates[tp.id] = tp

        self.timelines = {}
        self.order = []
        for t in root.get('timelines') or []:
            tl = Timeline(t, self.load_format)
            if tl.id:
                self.timelines[tl.id] = tl
                self.order.append(tl)

        for tl in self.order:
            if tl.parent and tl.parent in self.timelines:
                self.timelines[tl.parent].children.append(tl)
            else:
                tl.parent = None
        for tl in self.order:
            tl.children.sort(key=lambda c: c.parent_tree_index)

        self.bend_style = (self.render.get('render_bend_style') or self.render.get('bend_style') or 'realistic')

    # -- utilidades --------------------------------------------------------

    def frame_range(self):
        frames = [p for tl in self.order for p in tl.keyframe_positions()]
        if not frames:
            return 0, 0
        return min(frames), max(frames)

    def resource(self, rid):
        if rid is None:
            return None
        return self.resources.get(rid)

    def roots(self):
        return [tl for tl in self.order if tl.parent is None]


def _mimodel_as_project(path, model):
    """Envuelve un .mimodel suelto en un proyecto con una timeline por parte
    (lo mismo que hace Mine-imator al añadir el modelo a la escena)."""
    if not isinstance(model.get('parts'), list):
        raise ValueError("El .mimodel no tiene partes: " + os.path.basename(path))
    name = model.get('name') or os.path.splitext(os.path.basename(path))[0]
    timelines = [{'id': 'ROOT', 'type': 'model', 'name': name, 'temp': 'TEMP', 'parent': 'null'}]
    counter = [0]

    def add(parts, parent):
        for p in parts:
            if not isinstance(p, dict) or p.get('visible') is False or not isinstance(p.get('name'), str):
                continue
            counter[0] += 1
            pid = 'PART%d' % counter[0]
            tl = {'id': pid, 'type': 'bodypart', 'temp': 'TEMP', 'model_part_name': p['name'],
                  'part_of': 'ROOT', 'parent': parent,
                  # false: la parte sigue a la mitad superior del padre, no a la doblada
                  'lock_bend': p.get('lock_bend', True)}
            bend = p.get('bend')
            if isinstance(bend, dict) and bend.get('angle') is not None:
                ax = bend.get('axis')
                axes = [ax] if isinstance(ax, str) else (ax or [])
                ang = bend['angle'] if isinstance(bend['angle'], list) else [bend['angle']] * len(axes)
                key = {'x': 'BEND_ANGLE_X', 'z': 'BEND_ANGLE_Y', 'y': 'BEND_ANGLE_Z'}
                tl['default_values'] = {key[a]: ang[i] for i, a in enumerate(axes) if a in key and i < len(ang)}
            timelines.append(tl)
            add(p.get('parts') or [], pid)

    add(model['parts'], 'ROOT')
    return {
        'format': FORMAT_200_PRE_5,
        'created_in': 'Modelo de Mine-imator (.mimodel)',
        'project': {'name': name},
        'resources': [{'id': 'RES', 'type': 'model', 'filename': os.path.basename(path)}],
        'templates': [{'id': 'TEMP', 'type': 'model', 'model': 'RES', 'model_tex': 'null'}],
        'timelines': timelines,
    }


# ---------------------------------------------------------------------------
# Evaluacion de un fotograma (tl_update_values + tl_update_matrix)
# ---------------------------------------------------------------------------

INHERIT_KEYS = ('ALPHA', 'RGB_ADD', 'RGB_SUB', 'RGB_MUL', 'MIX_COLOR', 'MIX_PERCENT',
                'EMISSIVE', 'VISIBLE', 'BEND_ANGLE_X', 'BEND_ANGLE_Y', 'BEND_ANGLE_Z',
                'TEXTURE_OBJ')


class TLState:
    __slots__ = ('values', 'inherit', 'matrix', 'matrix_render', 'rot_point_render')


def _vec_bend(v):
    return (v.get('BEND_ANGLE_X', 0.0) or 0.0, v.get('BEND_ANGLE_Y', 0.0) or 0.0, v.get('BEND_ANGLE_Z', 0.0) or 0.0)


def _color_add(a, b):
    return tuple(min(1.0, a[i] + b[i]) for i in range(3))


def _color_mul(a, b):
    return tuple(a[i] * b[i] for i in range(3))


REST_VALUES = {'POS_X': 0.0, 'POS_Y': 0.0, 'POS_Z': 0.0, 'ROT_X': 0.0, 'ROT_Y': 0.0, 'ROT_Z': 0.0,
               'SCA_X': 1.0, 'SCA_Y': 1.0, 'SCA_Z': 1.0,
               'BEND_ANGLE_X': 0.0, 'BEND_ANGLE_Y': 0.0, 'BEND_ANGLE_Z': 0.0}


def supports_ik(part):
    """tl_supports_ik: parte con bend inferior solo en X y end_offset > 0."""
    b = part.bend if part is not None else None
    return (b is not None and b.part == 'lower' and b.end_offset > 0 and
            b.axis[0] and not b.axis[1] and not b.axis[2])


def do_ik(mat, part, target, pole, tl_pos, blend, angle_offset):
    """do_ik de Mine-imator (tl_update_ik.gml): FABRIK sobre dos huesos + polo.

    mat: matriz de la parte antes del IK. Devuelve (matriz de la primera articulación,
    ángulo de bend en grados)."""
    b = part.bend
    jp = [gm.position(mat)]
    bm = gm.multiply(part.bend_matrix((0.0, 0.0, 0.0)), mat)
    jp.append(gm.position(bm))
    jp.append(gm.position(gm.multiply(gm.create((0.0, 0.0, -b.end_offset), (0, 0, 0), (1, 1, 1)), bm)))

    pp = None
    if pole is not None:
        # separa el polo para que las matemáticas de ángulo no se degeneren
        pp = list(pole)
        for i in range(3):
            if pp[i] == tl_pos[i]:
                pp[i] += 0.0001
        for i in range(3):
            if pp[i] == target[i]:
                pp[i] += 0.0001
        pp = tuple(pp)

    end = tuple(jp[2][i] + (target[i] - jp[2][i]) * blend for i in range(3))
    jl = [gm.v_distance(jp[0], jp[1]), gm.v_distance(jp[1], jp[2])]
    invert = b.invert[0]
    if not invert:
        jp[1] = gm.v_add(jp[0], gm.v_sub(jp[0], jp[1]))
        jp[2] = gm.v_add(jp[0], gm.v_sub(jp[0], jp[2]))

    nobend = True
    if gm.v_distance(jp[0], end) > jl[0] + jl[1]:
        d = gm.v_direction(jp[0], end)
        for i in (1, 2):
            jp[i] = gm.v_add(jp[i - 1], gm.v_mul(d, jl[i - 1]))
    else:
        d1 = gm.v_direction(jp[0], end)
        d2 = gm.v_direction(jp[0], jp[1])
        if abs(d1[0]) == abs(d2[0]) and abs(d1[1]) == abs(d2[1]) and abs(d1[2]) == abs(d2[2]):
            end = gm.v_add(end, gm.v_mul(gm.v_normalize(gm.v_cross(d1, (-1.0, 0.0, 0.0))), 0.0001))
        dis = gm.v_distance(jp[0], end)
        t = 0
        while t < 30 and dis > 0.01:
            jp[2] = end
            jp[1] = gm.v_add(jp[2], gm.v_mul(gm.v_direction(jp[2], jp[1]), jl[1]))
            for i in (1, 2):
                jp[i] = gm.v_add(jp[i - 1], gm.v_mul(gm.v_direction(jp[i - 1], jp[i]), jl[i - 1]))
            dis = gm.v_distance(jp[0], end)
            t += 1
        nobend = False

    inv = gm.inverse(mat)
    p0, p1, p2 = (gm.transform_point(inv, p) for p in jp)
    if pp is not None:
        pp = gm.transform_point(inv, pp)
    ab = gm.v_direction(p0, p1)
    bc = gm.v_direction(p1, p2)
    ac = gm.v_direction(p0, p2)
    haspole = pp is not None

    if haspole and not nobend:
        poleproj = gm.v_project_plane(pp, p0, ac)
        p1proj = gm.v_project_plane(p1, p0, ac)
        ang = gm.v_angle_signed(gm.v_sub(p1proj, p0), gm.v_sub(poleproj, p0), ac)
        p1 = gm.v_add(p0, gm.v_rotate_axis_angle(gm.v_sub(p1, p0), ac, math.radians(ang + angle_offset)))
        ab = gm.v_direction(p0, p1)
        bc = gm.v_direction(p1, p2)

    if not haspole and nobend:
        n = gm.v_normal(ab, -90.0) if (ab[0] == 0 and ab[1] == 0) else gm.v_normal(ab, 0.0)
    else:
        p2proj = gm.v_project_plane(pp, p0, ab) if (haspole and nobend) else gm.v_project_plane(p2, p0, ab)
        n = gm.v_direction(p2proj, p0)
        if haspole and nobend:
            n = gm.v_rotate_axis_angle(n, ac, math.radians(angle_offset))
        if invert:
            n = gm.v_mul(n, -1.0)
        if haspole and nobend:
            n = gm.v_mul(n, -1.0)

    joint = gm.create_rotate_to(n, gm.v_mul(ab, -1.0))
    bend = math.degrees(math.acos(max(-1.0, min(1.0, gm.v_dot(ab, bc)))))
    return joint, bend


def evaluate(project, frame, part_lookup, rest_parts=False, ik=True, no_scale=()):
    """Devuelve {tl_id: TLState} para el fotograma dado.

    part_lookup(tl) -> ModelPart o None (para body parts).
    rest_parts: las partes de modelo (bodypart) quedan en su pose de reposo.
    ik: resuelve el IK como Mine-imator (tl_update_ik); no se aplica en pose de reposo.
    no_scale: ids de timelines cuya escala (SCA) se ignora (miembros con IK en el esqueleto:
        el IK de Blender no puede estirar la cadena).
    """
    values = {tl.id: tl.values_at(frame) for tl in project.order}
    for tid in no_scale:
        if tid in values:
            v = dict(values[tid])
            v.update({'SCA_X': 1.0, 'SCA_Y': 1.0, 'SCA_Z': 1.0})
            values[tid] = v
    if rest_parts:
        for tl in project.order:
            if tl.type == 'bodypart':
                v = dict(values[tl.id])
                v.update(REST_VALUES)
                values[tl.id] = v
    states = {}
    pre_ik = {}      # matriz de cada parte antes de aplicar el IK
    ik_data = {}     # id -> (matriz de la articulación, ángulo, mezcla)

    def vinherit(tl):
        v = values[tl.id]
        vi = {k: v.get(k, DEFAULTS.get(k)) for k in INHERIT_KEYS}
        ikd = ik_data.get(tl.id)
        if ikd is not None:
            for k in ('BEND_ANGLE_X', 'BEND_ANGLE_Y', 'BEND_ANGLE_Z'):
                vi[k] = (vi[k] or 0.0) * (1.0 - ikd[2])
        inhalpha = inhcolor = inhvis = inhbend = inhtex = True
        cur = tl
        while cur.parent:
            par = project.timelines[cur.parent]
            pv = values[par.id]
            if not cur.inherit_alpha:
                inhalpha = False
            if not cur.inherit_color:
                inhcolor = False
            if not cur.inherit_visibility:
                inhvis = False
            if not cur.inherit_bend:
                inhbend = False
            if not cur.inherit_texture or values[cur.id].get('TEXTURE_OBJ') is not None:
                inhtex = False
            if inhalpha:
                vi['ALPHA'] = (vi['ALPHA'] or 0) * (pv.get('ALPHA', 1.0) or 0)
            if inhcolor:
                if pv.get('RGB_ADD', BLACK) != BLACK:
                    vi['RGB_ADD'] = _color_add(vi['RGB_ADD'], pv['RGB_ADD'])
                if pv.get('RGB_SUB', BLACK) != BLACK:
                    vi['RGB_SUB'] = _color_add(vi['RGB_SUB'], pv['RGB_SUB'])
                if pv.get('RGB_MUL', WHITE) != WHITE:
                    vi['RGB_MUL'] = _color_mul(vi['RGB_MUL'], pv['RGB_MUL'])
                if pv.get('MIX_COLOR', BLACK) != BLACK:
                    vi['MIX_COLOR'] = _color_add(vi['MIX_COLOR'], pv['MIX_COLOR'])
                vi['MIX_PERCENT'] = min(1.0, max(0.0, (vi['MIX_PERCENT'] or 0) + (pv.get('MIX_PERCENT', 0) or 0)))
            if inhvis:
                vi['VISIBLE'] = (vi['VISIBLE'] or 0) * (pv.get('VISIBLE', 1) or 0)
            if inhbend:
                for k in ('BEND_ANGLE_X', 'BEND_ANGLE_Y', 'BEND_ANGLE_Z'):
                    vi[k] = (vi[k] or 0) + (pv.get(k, 0) or 0)
            if inhtex:
                vi['TEXTURE_OBJ'] = pv.get('TEXTURE_OBJ')
            cur = par
        if ikd is not None:
            vi['BEND_ANGLE_X'] = (vi['BEND_ANGLE_X'] or 0.0) + ikd[1] * ikd[2]
        return vi

    def rot_point_render(tl):
        temp = project.templates.get(tl.temp) if tl.temp else None
        if temp is not None and not tl.part_of and not tl.rot_point_custom:
            return temp.rot_point()
        return tl.rot_point

    def compute(tl):
        st = states.get(tl.id)
        if st is not None:
            return st
        v = values[tl.id]
        st = TLState()
        st.values = v
        st.inherit = vinherit(tl)

        if tl.parent:
            par = project.timelines[tl.parent]
            pst = compute(par)
            mparent = list(pst.matrix_render if tl.inherit_rot_point else pst.matrix)
            if par.type == 'bodypart' and tl.lock_bend:
                ppart = part_lookup(par)
                if ppart is not None and ppart.bend is not None:
                    mparent = gm.multiply(ppart.bend_matrix(_vec_bend(pst.inherit)), mparent)
        else:
            mparent = list(gm.IDENTITY)

        if tl.type == 'bodypart':
            part = part_lookup(tl)
            if part is not None:
                if tl.part_of:
                    mparent = gm.multiply(gm.create(part.position, part.rotation, (1, 1, 1)), mparent)
                else:
                    mparent = gm.multiply(gm.create((0, 0, 0), part.rotation, (1, 1, 1)), mparent)

        pos = (v.get('POS_X', 0.0) or 0.0, v.get('POS_Y', 0.0) or 0.0, v.get('POS_Z', 0.0) or 0.0)
        rot = (v.get('ROT_X', 0.0) or 0.0, v.get('ROT_Y', 0.0) or 0.0, v.get('ROT_Z', 0.0) or 0.0)
        sca = (_num(v.get('SCA_X'), 1.0), _num(v.get('SCA_Y'), 1.0), _num(v.get('SCA_Z'), 1.0))

        m = gm.multiply(gm.create(pos, rot, sca), mparent)

        if not tl.inherit_rotation:
            m = gm.remove_rotation(m)
            m = gm.multiply(gm.create((0, 0, 0), rot, (1, 1, 1)), m)

        # IK: orientación de la primera articulación
        pre_ik[tl.id] = list(m)
        ikd = ik_data.get(tl.id)
        if ikd is not None:
            m = gm.multiply(ikd[0], m)

        if tl.scale_resize or not tl.inherit_scale:
            sca2 = list(sca)
            cur = tl
            while True:
                if not cur.inherit_scale or not cur.parent:
                    break
                par = project.timelines[cur.parent]
                pv = values[par.id]
                sca2[0] *= _num(pv.get('SCA_X'), 1.0)
                sca2[1] *= _num(pv.get('SCA_Y'), 1.0)
                sca2[2] *= _num(pv.get('SCA_Z'), 1.0)
                cur = par
            mparent = gm.remove_scale(mparent)
            parmat = list(mparent)
            if not tl.inherit_rotation:
                parmat = gm.remove_rotation(parmat)
            mns = gm.multiply(gm.create(pos, rot, (1, 1, 1)), parmat)
            if ikd is not None:
                mns = gm.multiply(ikd[0], mns)
            m = list(m)
            for i in range(11):
                m[i] = mns[i]
            if tl.inherit_scale:
                m = gm.multiply(gm.create((0, 0, 0), (0, 0, 0), sca2), m)
            else:
                m = gm.multiply(gm.create((0, 0, 0), (0, 0, 0), sca), m)

        if not tl.inherit_position:
            m = list(m)
            m[12], m[13], m[14] = pos

        rp = rot_point_render(tl)
        st.rot_point_render = rp
        st.matrix = m
        st.matrix_render = gm.multiply(gm.create((-rp[0], -rp[1], -rp[2]), (0, 0, 0), (1, 1, 1)), m)
        states[tl.id] = st
        return st

    def run():
        states.clear()
        pre_ik.clear()
        for tl in project.order:
            compute(tl)

    run()

    # IK (tl_update_ik): se repite hasta que las posiciones dejan de cambiar (máx. 2 veces)
    if ik and not rest_parts:
        cands = []
        for tl in project.order:
            if tl.type != 'bodypart' or not values[tl.id].get('IK_TARGET'):
                continue
            part = part_lookup(tl)
            if supports_ik(part):
                cands.append((tl, part))
        for _ in range(2 if cands else 0):
            new = {}
            for tl, part in cands:
                v = values[tl.id]
                tgt = states.get(str(v.get('IK_TARGET')))
                if tgt is None:
                    continue
                pole_id = save_id(v.get('IK_TARGET_ANGLE'))
                pst = states.get(pole_id) if pole_id else None
                blend = _num(v.get('IK_BLEND'), 1.0)
                new[tl.id] = do_ik(pre_ik[tl.id], part, gm.position(tgt.matrix),
                                   gm.position(pst.matrix) if pst is not None else None,
                                   gm.position(states[tl.id].matrix), blend,
                                   _num(v.get('IK_ANGLE_OFFSET'), 0.0)) + (blend,)
            same = (set(new) == set(ik_data) and all(
                max(abs(a - b) for a, b in zip(new[k][0], ik_data[k][0])) < 1e-6 and abs(new[k][1] - ik_data[k][1]) < 1e-6
                for k in new))
            if same:
                break
            ik_data.clear()
            ik_data.update(new)
            run()
    return states

def _num(v, default):
    if v is None or isinstance(v, str):
        return default
    return float(v)
