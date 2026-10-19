# Builds the Blender scene from a Mine-imator project (geometry, rigs, IK, bends, optional animation).

import math
import os
import tempfile

import bpy
import numpy as np
from mathutils import Matrix, Vector

from . import gm_math as gm
from . import mi_data
from . import mi_assets
from . import mi_geometry as geo
from .mi_geometry import MatKey

IMAGE_RES_TYPES = ('skin', 'downloadedskin', 'downloaded_skin', 'texture', 'itemsheet', 'item_sheet',
                   'blocksheet', 'block_sheet', 'legacyblocksheet', 'particlesheet')


class Options:
    def __init__(self, **kw):
        self.version = kw.get('version', 'AUTO')
        self.mi_dir = kw.get('mi_dir', '')
        self.frame = int(kw.get('frame', 0))
        self.animation = bool(kw.get('animation', False))
        self.anim_step = max(1, int(kw.get('anim_step', 1)))
        self.unit_scale = float(kw.get('unit_scale', 1.0 / 16.0))
        self.import_hidden = bool(kw.get('import_hidden', True))
        self.pack_textures = bool(kw.get('pack_textures', True))
        self.merge_vertices = bool(kw.get('merge_vertices', True))
        self.depth_masks = bool(kw.get('depth_masks', True))
        self.rig = bool(kw.get('rig', True))
        self.ik = bool(kw.get('ik', True))
        self.ik_keep_pose = bool(kw.get('ik_keep_pose', True))
        self.bend_weights = bool(kw.get('bend_weights', True))
        self.standard_view = bool(kw.get('standard_view', True))
        # --- Suite integration -------------------------------------------------------------
        self.asset_zip = kw.get('asset_zip', '') or ''        # explicit Minecraft asset ZIP (optional)
        self.outer_layers_3d = bool(kw.get('outer_layers_3d', False))  # voxel hat/jacket/sleeves/pants
        self.categories = kw.get('categories')                 # None = everything, else a set of Suite categories
        self.category_collections = kw.get('category_collections') or {}  # category -> bpy Collection


# ---------------------------------------------------------------------------
# Conversion de coordenadas
# ---------------------------------------------------------------------------

def gm_to_matrix(m):
    return Matrix(((m[0], m[4], m[8], m[12]),
                   (m[1], m[5], m[9], m[13]),
                   (m[2], m[6], m[10], m[14]),
                   (0.0, 0.0, 0.0, 1.0)))


class Converter:
    """Mine-imator (Z arriba, levogiro) -> Blender (Z arriba, dextrogiro)."""

    def __init__(self, unit):
        self.unit = unit
        self.C = Matrix.Diagonal((unit, -unit, unit, 1.0))
        self.Ci = Matrix.Diagonal((1.0 / unit, -1.0 / unit, 1.0 / unit, 1.0))

    def matrix(self, m):
        return self.C @ gm_to_matrix(m) @ self.Ci

    def point(self, p):
        u = self.unit
        return (p[0] * u, -p[1] * u, p[2] * u)


def decompose(W):
    """Devuelve (Wd, residuo) con Wd = Loc*Rot*Scale y W = Wd @ residuo."""
    try:
        loc, rot, sca = W.decompose()
        for i in range(3):
            if abs(sca[i]) < 1e-8:
                sca[i] = 1e-8
        Wd = Matrix.LocRotScale(loc, rot, sca)
        res = Wd.inverted_safe() @ W
        ident = Matrix.Identity(4)
        if all(abs(res[i][j] - ident[i][j]) < 1e-5 for i in range(4) for j in range(4)):
            res = None
        return Wd, res
    except (ValueError, ZeroDivisionError):
        return Matrix.Translation(W.to_translation()), None


# ---------------------------------------------------------------------------
# Texturas y materiales
# ---------------------------------------------------------------------------

class TextureLib:
    def __init__(self, assets, cache_dir, log):
        self.assets = assets
        self.cache_dir = cache_dir
        self.log = log
        self.images = {}
        self.alphas = {}
        self.missing = set()

    def image(self, ref):
        if ref is None:
            return None
        if ref in self.images:
            return self.images[ref]
        img = None
        kind, path = ref
        try:
            if kind == 'mc':
                fpath = self.assets.extract(path, self.cache_dir) if self.assets and self.assets.ok else None
            else:
                fpath = path
            if fpath and os.path.isfile(fpath):
                img = bpy.data.images.load(fpath, check_existing=True)
                img.alpha_mode = 'STRAIGHT'
        except RuntimeError as e:
            self.log("Texture could not be loaded %s: %s" % (path, e))
            img = None
        if img is None and ref not in self.missing:
            self.missing.add(ref)
            self.log("Texture not found: %s" % path)
        self.images[ref] = img
        return img

    def size(self, ref):
        img = self.image(ref)
        if img is None:
            return None
        return img.size[0], img.size[1]

    def alpha(self, ref):
        """(ancho, alto, fn(x, y) -> alfa) con y=0 arriba."""
        if ref in self.alphas:
            return self.alphas[ref]
        img = self.image(ref)
        result = None
        if img is not None and img.size[0] > 0 and img.size[1] > 0:
            w, h = img.size[0], img.size[1]
            try:
                import numpy as np
                arr = np.empty(w * h * 4, dtype=np.float32)
                img.pixels.foreach_get(arr)
                a = arr[3::4].reshape(h, w)
            except ImportError:
                px = list(img.pixels)
                a = [[px[(r * w + c) * 4 + 3] for c in range(w)] for r in range(h)]

            def fn(x, y, a=a, w=w, h=h):
                if x < 0 or y < 0 or x >= w or y >= h:
                    return 0.0
                return float(a[h - 1 - y][x])
            result = (w, h, fn)
            self.alphas[('array', ref)] = a
        self.alphas[ref] = result
        return result

    def partial_alpha(self, ref):
        """True si la textura tiene píxeles semitransparentes (ni 0 ni 1)."""
        if not self.alpha(ref):
            return False
        key = ('partial', ref)
        if key not in self.alphas:
            arr = self.alphas[('array', ref)]
            try:
                import numpy as np
                self.alphas[key] = bool(np.any((arr > 0.004) & (arr < 0.996)))
            except ImportError:
                self.alphas[key] = any(0.004 < v < 0.996 for row in arr for v in row)
        return self.alphas[key]

    def frames(self, ref):
        s = self.size(ref)
        if not s:
            return 1
        w, h = s
        if w > 0 and h > w and h % w == 0:
            return h // w
        return 1


def srgb_to_linear(c):
    """Los colores de Mine-imator estan en sRGB; los nodos de Blender son lineales."""
    def ch(v):
        v = min(max(float(v), 0.0), 1.0)
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
    return (ch(c[0]), ch(c[1]), ch(c[2]))


def _find_socket(sockets, *names):
    for n in names:
        for s in sockets:
            if s.identifier == n:
                return s
    for n in names:
        if n in sockets:
            return sockets[n]
    return None


class MaterialLib:
    def __init__(self, textures, prefix):
        self.textures = textures
        self.prefix = prefix
        self.cache = {}

    def get(self, key):
        if key in self.cache:
            return self.cache[key]
        img = self.textures.image(key.tex) if key.tex else None
        base = os.path.splitext(img.name)[0] if img else 'color'
        mat = bpy.data.materials.new('%s_%s' % (self.prefix, base))
        self._setup(mat, key, img)
        self.cache[key] = mat
        return mat

    def _setup(self, mat, key, img):
        try:
            mat.use_nodes = True
        except AttributeError:
            pass
        # Transparencia: "Blended" (sin grano) solo para objetos con opacidad
        # parcial (lentes, rubores, sombras). Las texturas recortadas usan
        # "Dithered", que ordena bien la profundidad en EEVEE.
        semi = key.alpha < 0.999
        if hasattr(mat, 'surface_render_method'):
            mat.surface_render_method = 'BLENDED' if semi else 'DITHERED'
        if hasattr(mat, 'blend_method'):
            try:
                mat.blend_method = 'BLEND' if semi else 'HASHED'
            except TypeError:
                pass
        if hasattr(mat, 'shadow_method'):
            try:
                mat.shadow_method = 'HASHED'
            except TypeError:
                pass
        if hasattr(mat, 'show_transparent_back'):
            mat.show_transparent_back = not key.cull
        mat.use_backface_culling = key.cull
        color = srgb_to_linear(key.color)
        add = srgb_to_linear(key.add)
        mixc = srgb_to_linear(key.mix)
        mat.diffuse_color = (color[0], color[1], color[2], key.alpha)

        nt = mat.node_tree
        nodes, links = nt.nodes, nt.links
        bsdf = None
        for n in nodes:
            if n.type == 'BSDF_PRINCIPLED':
                bsdf = n
        out = None
        for n in nodes:
            if n.type == 'OUTPUT_MATERIAL':
                out = n
        if bsdf is None:
            bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        if out is None:
            out = nodes.new('ShaderNodeOutputMaterial')
            links.new(bsdf.outputs[0], out.inputs[0])
        bsdf.location = (300, 0)
        out.location = (600, 0)

        s_base = _find_socket(bsdf.inputs, 'Base Color')
        s_alpha = _find_socket(bsdf.inputs, 'Alpha')
        s_rough = _find_socket(bsdf.inputs, 'Roughness')
        s_spec = _find_socket(bsdf.inputs, 'Specular IOR Level', 'Specular')
        s_ecol = _find_socket(bsdf.inputs, 'Emission Color', 'Emission')
        s_estr = _find_socket(bsdf.inputs, 'Emission Strength')
        if s_rough:
            s_rough.default_value = 1.0
        if s_spec:
            s_spec.default_value = 0.0

        color_out = None
        alpha_out = None
        x = -900
        if img is not None:
            tex = nodes.new('ShaderNodeTexImage')
            tex.image = img
            tex.interpolation = 'Closest'
            tex.extension = 'REPEAT' if key.repeat else 'EXTEND'
            tex.location = (x, 0)
            color_out = tex.outputs['Color']
            alpha_out = tex.outputs['Alpha']
            x += 250

        def mix(blend, a, b_val, fac=1.0):
            nonlocal x
            n = nodes.new('ShaderNodeMix')
            n.data_type = 'RGBA'
            n.blend_type = blend
            n.location = (x, 0)
            x += 200
            f = _find_socket(n.inputs, 'Factor_Float')
            ia = _find_socket(n.inputs, 'A_Color')
            ib = _find_socket(n.inputs, 'B_Color')
            f.default_value = fac
            if isinstance(a, tuple):
                ia.default_value = (a[0], a[1], a[2], 1.0)
            else:
                links.new(a, ia)
            ib.default_value = (b_val[0], b_val[1], b_val[2], 1.0)
            return _find_socket(n.outputs, 'Result_Color')

        cur = color_out
        if cur is None:
            cur = tuple(color)
        elif key.color != (1.0, 1.0, 1.0):
            cur = mix('MULTIPLY', cur, color)
        if any(c > 0 for c in key.add):
            cur = mix('ADD', cur, add)
        if key.mix_pct > 0:
            cur = mix('MIX', cur, mixc, key.mix_pct)
        if cur is not color_out and color_out is not None:
            # Marks a texture that is tinted by Mine-imator's color. Post-import tools that rebuild the node
            # tree (MCprep material preparation) would drop these nodes, so the Suite leaves such materials alone.
            mat['mi_tinted'] = True

        if isinstance(cur, tuple):
            s_base.default_value = (cur[0], cur[1], cur[2], 1.0)
        else:
            links.new(cur, s_base)

        if s_alpha:
            a_sock = alpha_out
            if a_sock is not None and key.alpha < 1.0:
                m = nodes.new('ShaderNodeMath')
                m.operation = 'MULTIPLY'
                m.location = (x, -250)
                links.new(a_sock, m.inputs[0])
                m.inputs[1].default_value = key.alpha
                a_sock = m.outputs[0]
            if key.cull and (a_sock is not None or key.alpha < 1.0):
                # Ocultar caras traseras tambien en Cycles (alfa * (1 - Backfacing))
                geo_n = nodes.new('ShaderNodeNewGeometry')
                geo_n.location = (x - 200, -450)
                inv = nodes.new('ShaderNodeMath')
                inv.operation = 'SUBTRACT'
                inv.location = (x, -450)
                inv.inputs[0].default_value = 1.0
                links.new(geo_n.outputs['Backfacing'], inv.inputs[1])
                m = nodes.new('ShaderNodeMath')
                m.operation = 'MULTIPLY'
                m.location = (x + 200, -350)
                if a_sock is not None:
                    links.new(a_sock, m.inputs[0])
                else:
                    m.inputs[0].default_value = key.alpha
                links.new(inv.outputs[0], m.inputs[1])
                a_sock = m.outputs[0]
            if a_sock is not None:
                links.new(a_sock, s_alpha)
            else:
                s_alpha.default_value = key.alpha

        if key.emissive > 0 and s_ecol and s_estr:
            if isinstance(cur, tuple):
                s_ecol.default_value = (cur[0], cur[1], cur[2], 1.0)
            else:
                links.new(cur, s_ecol)
            s_estr.default_value = key.emissive


# ---------------------------------------------------------------------------
# Importador
# ---------------------------------------------------------------------------

class Importer:
    def __init__(self, filepath, options, report=None, extras=None):
        self.filepath = filepath
        self.extras = extras      # optional Suite hooks (cameras, lights, scenery, paths...)
        self.local_fix = {}       # tl.id -> matrix applied to the right of the object's world matrix
        self.unit_scale_ids = set()  # tl.id of objects whose world scale is dropped (cameras, lights)
        self.stats = {}
        self.opt = options
        self.messages = []
        self._report = report
        self.project = mi_data.Project(filepath, options.version)
        mi_dir = mi_assets.find_mineimator_dir(filepath, options.mi_dir, options.version)
        self.assets = mi_assets.MinecraftAssets(mi_dir, options.asset_zip)
        if not self.assets.ok:
            self.log("Warning: the Minecraft resources (Data/Minecraft) were not found. "
                     "Minecraft characters, blocks and items cannot be converted. "
                     "Set the Mine-imator folder in the options.")
        cache = os.path.join(tempfile.gettempdir(), 'mineimator_to_blender_cache')
        self.textures = TextureLib(self.assets, cache, self.log)
        self.materials = MaterialLib(self.textures, 'MI')
        self.conv = Converter(options.unit_scale)
        self.models = {}
        self.model_info = {}
        self.obj_alpha = {}
        self.objects = {}
        self.base_world = {}
        self.skipped = {}
        self.rigs = {}
        self.bone_of = {}
        self.rig_meshes = set()
        self.links = {}
        self.base_pf = {}
        self.frame_states = None
        self.skipped_category = {}

    def log(self, msg):
        self.messages.append(msg)
        print('[Mine-imator] ' + msg)

    # -- modelos -----------------------------------------------------------

    def template_model(self, temp):
        if temp is None:
            return None, None
        key = temp.id
        if key in self.models:
            return self.models[key], self.model_info.get(key)
        model, info = None, None
        try:
            if temp.type == 'model':
                res = self.project.resource(temp.model_res)
                if res is not None:
                    path = mi_assets.find_project_file(self.project.folder, res.filename)
                    if path and path.lower().endswith(('.mimodel', '.json')):
                        model = geo.ModelFile.load_file(path)
                        if not model.parts:
                            model = None
                    if model is None:
                        self.log("Model could not be loaded '%s'" % res.filename)
            elif temp.type in ('char', 'spblock', 'bodypart') and self.assets.ok and temp.model_name:
                info = self.assets.resolve_character(temp.model_name, temp.model_state, special=(temp.type == 'spblock'))
                if info is None and temp.type == 'bodypart':
                    info = self.assets.resolve_character(temp.model_name, temp.model_state, special=True)
                if info:
                    data = self.assets.read_json(info['path'])
                    if isinstance(data, dict):
                        model = geo.ModelFile.load_data(data)
                if model is None:
                    self.log("Minecraft model not found: %s" % temp.model_name)
        except (OSError, ValueError, KeyError, TypeError) as e:
            self.log("Error loading the model of template '%s': %s" % (temp.name or temp.id, e))
            model = None
        self.models[key] = model
        self.model_info[key] = info
        return model, info

    def part_lookup(self, tl):
        if tl.type != 'bodypart':
            return None
        temp = self.project.templates.get(tl.temp)
        model, _info = self.template_model(temp)
        if model is None:
            return None
        name = tl.model_part_name or (temp.model_part_name if temp else '')
        return model.part_map.get(name)

    # -- texturas ----------------------------------------------------------

    def tex_override(self, st):
        """Textura del timeline: (hay_valor, texref). 'none' = sin textura explícita."""
        rid = st.inherit.get('TEXTURE_OBJ')
        if rid is None:
            return False, None
        if str(rid).lower() == 'none':
            return True, None
        ref = self.res_texref(rid)
        return (ref is not None), ref

    def res_texref(self, rid):
        res = self.project.resource(rid)
        if res is None:
            return None
        if res.type not in IMAGE_RES_TYPES and not res.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tga', '.bmp')):
            return None
        path = mi_assets.find_project_file(self.project.folder, res.filename)
        if not path:
            self.log("Resource not found: %s" % res.filename)
            return None
        return ('file', path)

    def model_texref(self, model, name):
        if not name:
            return None
        if model.source == 'file':
            p = mi_assets.find_project_file(model.base, name)
            if p is None and not os.path.splitext(name)[1]:
                p = mi_assets.find_project_file(model.base, name + '.png')
            if p is None:
                p = mi_assets.find_project_file(self.project.folder, name)
            return ('file', p) if p else None
        p = self.assets.texture_path(name) if self.assets.ok else None
        return ('mc', p) if p else None

    # -- geometria por timeline ----------------------------------------------

    def color_params(self, st):
        vi = st.inherit
        mul = vi.get('RGB_MUL') or mi_data.WHITE
        add = vi.get('RGB_ADD') or mi_data.BLACK
        return mul, add, float(vi.get('ALPHA', 1.0) or 0.0), float(st.values.get('EMISSIVE', 0.0) or 0.0), \
            vi.get('MIX_COLOR') or mi_data.BLACK, float(vi.get('MIX_PERCENT', 0.0) or 0.0)

    def geometry(self, tl, st, rig=False):
        t = tl.type
        temp = self.project.templates.get(tl.temp)
        if t == 'bodypart':
            return self.geo_bodypart(tl, st, temp, rig)
        if t in mi_data.SHAPE_TYPES and temp is not None:
            return self.geo_shape(tl, st, temp)
        if t == 'item' and temp is not None:
            return self.geo_item(tl, st, temp)
        if t == 'block' and temp is not None:
            return self.geo_block(tl, st, temp)
        return None

    def geo_bodypart(self, tl, st, temp, rig=False):
        model, info = self.template_model(temp)
        part = self.part_lookup(tl)
        if part is None or not part.shapes:
            return None
        mul, add, alpha, emis, mixc, mixp = self.color_params(st)
        has_ov, override = self.tex_override(st)
        if not has_ov and temp is not None and temp.model_tex:
            override = self.res_texref(temp.model_tex)
        shape_tex = (info or {}).get('shape_texture') or {}
        shape_hide = set((info or {}).get('shape_hide') or [])
        shape_color = (info or {}).get('shape_color') or {}
        state_tex = (info or {}).get('texture')
        bend = mi_data._vec_bend(st.inherit)

        polys = []
        for shape in part.shapes:
            if shape.description and shape.description in shape_hide:
                continue
            if shape.color_alpha * alpha <= 0.001:
                continue
            # Textura
            texref = override
            if texref is None:
                name = shape.texture_name or part.texture_name
                if shape.description in shape_tex:
                    name = shape_tex[shape.description]
                elif state_tex and name == model.texture_name:
                    name = state_tex
                texref = self.model_texref(model, name)
            # Geometria
            if shape.type == 'block':
                sp = geo.gen_block(shape)
                if (self.opt.outer_layers_3d and model is not None and model.player_skin
                        and not shape.invert and max(shape.inflate) > 0.0):
                    # 3D outer layers: hat, jacket, sleeves and pants become one-texel-deep voxel shells
                    a = self.textures.alpha(texref) if texref else None
                    if a:
                        sp = geo.voxelize_block(sp, a, shape.texture_size, 1.0)
                        self.stats['3d_outer_layer'] = self.stats.get('3d_outer_layer', 0) + 1
                        self.stats['3d_outer_polys'] = self.stats.get('3d_outer_polys', 0) + len(sp)
                    else:
                        self.log("Warning: 3D outer layer '%s' needs a readable skin texture; kept the flat layer"
                                 % (shape.description or part.name))
            elif shape.is3d:
                a = self.textures.alpha(texref) if texref else None
                sp = geo.gen_plane_3d(shape, a) if a else geo.gen_plane(shape)
            else:
                sp = geo.gen_plane(shape)
            if rig:
                # Con esqueleto: malla sin doblar; el doblado lo hacen los huesos
                axis, cuts, fn, wfn = shape.rig_transform(self.project.bend_style)
                sp = geo.apply_transform(sp, axis, cuts, fn, wfn)
            else:
                axis, cuts, fn = shape.transform(bend, self.project.bend_style)
                sp = geo.apply_transform(sp, axis, cuts, fn)
            # Color
            color = shape.color_blend
            if shape.description in shape_color:
                color = geo._color_mul(color, mi_data.parse_color(shape_color[shape.description]))
            if shape.use_model_color and temp is not None and temp.model_use_blend_color:
                color = geo._color_mul(color, temp.model_blend_color)
            color = geo._color_mul(color, mul)
            mix_c, mix_p = (mixc, mixp) if mixp > 0 else (shape.color_mix, shape.color_mix_percent)
            key = MatKey(texref, color, shape.color_alpha * alpha, shape.color_emissive + emis, mix_c, mix_p, add, False)
            px, py, pz = shape.position
            for p in sp:
                p.verts = [(v[0] + px, v[1] + py, v[2] + pz) for v in p.verts]
                p.mat = key
            polys.extend(sp)
        return polys

    def geo_shape(self, tl, st, temp):
        polys = geo.gen_primitive(temp)
        mul, add, alpha, emis, mixc, mixp = self.color_params(st)
        has_ov, texref = self.tex_override(st)
        if not has_ov and temp.shape_tex:
            texref = self.res_texref(temp.shape_tex)
        key = MatKey(texref, mul, alpha, emis, mixc, mixp, add, True)
        for p in polys:
            p.mat = key
        return polys

    def geo_item(self, tl, st, temp):
        mul, add, alpha, emis, mixc, mixp = self.color_params(st)
        sheet = None
        texref = None
        slot = temp.item_slot
        if temp.item_name and self.assets.ok:
            slot = self.assets.item_slot(temp.item_name)
        if as_bool_num(st.values.get('CUSTOM_ITEM_SLOT')):
            slot = int(st.values.get('ITEM_SLOT', slot) or 0)
        # La textura del timeline (TEXTURE_OBJ) sustituye a la de la plantilla
        rid = st.inherit.get('TEXTURE_OBJ')
        res = self.project.resource(rid) if rid else None
        if res is None:
            res = self.project.resource(temp.item_tex)
        if res is not None:
            texref = self.res_texref(res.id)
            if res.type in ('itemsheet', 'item_sheet'):
                sheet = res.item_sheet_size
        if texref is None and self.assets.ok:
            name = temp.item_name or self.assets.item_texture_name(slot)
            p = self.assets.texture_path(name) if name else None
            if p:
                texref = ('mc', p)
            sheet = None
        if texref is None:
            return None
        a = self.textures.alpha(texref)
        if a is None:
            return None
        w, h, fn = a
        polys = geo.gen_item(w, h, fn, temp.item_3d, sheet, slot)
        key = MatKey(texref, mul, alpha, emis, mixc, mixp, add, False)
        for p in polys:
            p.mat = key
        return polys

    def geo_block(self, tl, st, temp):
        if not self.assets.ok or not temp.block_name:
            return None
        bs_path, state = self.assets.resolve_block(temp.block_name, temp.block_state)
        if not bs_path:
            self.log("Block not found: %s" % temp.block_name)
            return None
        mul, add, alpha, emis, mixc, mixp = self.color_params(st)
        bg = self.project.background
        grass = mi_data.parse_color(bg.get('grass_color'), (0.57, 0.74, 0.35))
        foliage = mi_data.parse_color(bg.get('foliage_color'), (0.47, 0.67, 0.19))
        water = mi_data.parse_color(bg.get('water_color'), (0.25, 0.46, 0.89))
        _has, override = self.tex_override(st)

        def tex_cb(name):
            p = self.assets.texture_path(name)
            return self.textures.frames(('mc', p)) if p else 1

        def tint_cb(name, idx):
            n = temp.block_name
            if 'water' in n:
                return water
            if 'leaves' in n or 'vine' in n:
                return foliage
            return grass

        faces = geo.gen_block_mc(self.assets, bs_path, state, tex_cb, tint_cb)
        polys = []
        rep = temp.block_repeat if temp.block_repeat_enable else (1, 1, 1)
        rx, ry, rz = (max(1, int(r)) for r in rep)
        for poly, texname, tint in faces:
            texref = override
            if texref is None:
                p = self.assets.texture_path(texname)
                texref = ('mc', p) if p else None
            color = geo._color_mul(mul, tint) if tint else mul
            key = MatKey(texref, color, alpha, emis, mixc, mixp, add, False)
            for ix in range(rx):
                for iy in range(ry):
                    for iz in range(rz):
                        verts = [(v[0] + 16 * ix, v[1] + 16 * iy, v[2] + 16 * iz) for v in poly.verts]
                        polys.append(geo.Poly(verts, poly.uvs, key))
        return polys

    # -- creacion de objetos ---------------------------------------------------

    def make_mesh(self, name, polys, residual, groups=None):
        polys = [p for p in polys if len(p.verts) >= 3]
        if not polys:
            return None
        # Todos los vértices de una vez (orden invertido: Mine-imator gira al revés)
        co = []
        uvl = []
        wl = []
        sizes = []
        for p in polys:
            co += p.verts[::-1]
            uvl += p.uvs[::-1]
            n = len(p.verts)
            wl += p.weights[::-1] if p.weights else [0.0] * n
            sizes.append(n)
        u = self.conv.unit
        arr = np.asarray(co, dtype=np.float64) * (u, -u, u)
        if residual is not None:
            R = np.array(residual, dtype=np.float64)
            arr = arr @ R[:3, :3].T + R[:3, 3]
        w = np.asarray(wl, dtype=np.float64)
        # Planos de dos caras: anverso y reverso coinciden exactamente. Con el culling de caras
        # traseras, Cycles puede golpear primero la cara de atrás (transparente) y perder también
        # la de delante. Se separa cada cara una distancia mínima a lo largo de su propia normal.
        starts = np.concatenate(([0], np.cumsum(sizes)))
        seen = {}
        for pi, n in enumerate(sizes):
            if n < 3:
                continue
            key = tuple(sorted(map(tuple, np.round(arr[starts[pi]:starts[pi] + n], 4).tolist())))
            seen.setdefault(key, []).append(pi)
        eps = 2e-4
        for pis in seen.values():
            if len(pis) < 2:
                continue
            for pi in pis:
                s, n = starts[pi], sizes[pi]
                P = arr[s:s + n]
                nrm = np.cross(P, np.roll(P, -1, axis=0)).sum(axis=0)
                ln = np.linalg.norm(nrm)
                if ln > 1e-12:
                    arr[s:s + n] = P + (nrm / ln) * eps
        if self.opt.merge_vertices:
            # Une vértices coincidentes (mismo sitio y mismo peso de doblado)
            keys = np.column_stack((np.round(arr, 5), np.round(w, 3)))
            _k, first, inv = np.unique(keys, axis=0, return_index=True, return_inverse=True)
            inv = inv.reshape(-1)
            verts, vweights = arr[first], w[first]
        else:
            inv = np.arange(len(arr))
            verts, vweights = arr, w
        inv = inv.tolist()

        faces = []
        keep = []
        mats = []
        smooth = []
        matlist = []
        matidx = {}
        pos = 0
        for p, n in zip(polys, sizes):
            face = inv[pos:pos + n]
            pos += n
            ok = len(set(face)) >= 3
            keep.append(ok)
            if not ok:
                continue
            faces.append(face)
            mk = p.mat if p.mat is not None else MatKey()
            i = matidx.get(mk)
            if i is None:
                i = matidx[mk] = len(matlist)
                matlist.append(mk)
            mats.append(i)
            smooth.append(bool(p.smooth))
        if not faces:
            return None
        me = bpy.data.meshes.new(name)
        me.from_pydata(verts.tolist(), [], faces)
        vweights = vweights.tolist()
        uv = np.asarray(uvl, dtype=np.float32).reshape(-1, 2)
        if not all(keep):
            uv = uv[np.repeat(np.asarray(keep), sizes)]
        uv[:, 1] = 1.0 - uv[:, 1]
        me.uv_layers.new(name='UVMap').data.foreach_set('uv', uv.ravel())
        for mk in matlist:
            me.materials.append(self.materials.get(mk))
        me.polygons.foreach_set('material_index', mats)
        me.polygons.foreach_set('use_smooth', smooth)
        # Sin validate(): borra los polígonos duplicados, y el anverso y el reverso de cada
        # plano de Mine-imator comparten vértices; se perdería una de las dos caras.
        me.update()
        self._pending_groups = None
        if groups:
            self._pending_groups = (groups, vweights)
        return me

    def assign_groups(self, obj):
        """Crea los grupos de vértices (hueso de la parte / hueso de doblado)."""
        if not getattr(self, '_pending_groups', None):
            return
        (bone, bend_bone), weights = self._pending_groups
        self._pending_groups = None
        n = len(obj.data.vertices)
        weights = weights[:n]
        g_main = obj.vertex_groups.new(name=bone)
        if bend_bone is None:
            g_main.add(list(range(n)), 1.0, 'REPLACE')
            return
        g_bend = obj.vertex_groups.new(name=bend_bone)
        buckets = {}
        for i, w in enumerate(weights):
            buckets.setdefault(round(w, 3), []).append(i)
        for w, idx in buckets.items():
            if w < 1.0:
                g_main.add(idx, 1.0 - w, 'REPLACE')
            if w > 0.0:
                g_bend.add(idx, w, 'REPLACE')

    def object_name(self, tl):
        temp = self.project.templates.get(tl.temp)
        if tl.name:
            return tl.name
        if tl.type == 'bodypart':
            # Modelbench muestra "New part" en las partes sin nombre
            return tl.model_part_name if tl.model_part_name.strip() else 'New part'
        if temp is not None:
            if temp.name:
                return temp.name
            if temp.model_name:
                return temp.model_name
            if temp.type == 'model':
                res = self.project.resource(temp.model_res)
                if res:
                    return os.path.splitext(os.path.basename(res.filename))[0]
            if temp.type == 'item' and temp.item_name:
                return temp.item_name.split('/')[-1]
            if temp.type == 'block' and temp.block_name:
                return temp.block_name
        return tl.type.capitalize() if tl.type else 'Object'

    # -- Suite integration helpers --------------------------------------------

    TYPE_CATEGORY = {
        'char': 'characters', 'entity': 'characters', 'spblock': 'blocks', 'block': 'blocks', 'model': 'models',
        'item': 'items', 'scenery': 'scenery', 'text': 'primitives',
        'pointlight': 'lights', 'spotlight': 'lights', 'camera': 'camera',
        'path': 'helpers', 'pathpoint': 'helpers', 'particles': 'helpers', 'folder': 'helpers',
    }

    def wants_type(self, tl):
        if tl.type in ('char', 'spblock', 'model', 'bodypart', 'item', 'block', 'text') or tl.type in mi_data.SHAPE_TYPES:
            return True
        return self.extras is not None and self.extras.handles(tl.type)

    def category_of(self, tl):
        """Suite category (characters, items, blocks, models, scenery, primitives, lights, camera, helpers)."""
        if tl.type == 'bodypart':
            root = self.project.timelines.get(tl.part_of) if tl.part_of else None
            return self.category_of(root) if root is not None else 'characters'
        if tl.type in mi_data.SHAPE_TYPES:
            return 'primitives'
        return self.TYPE_CATEGORY.get(tl.type, 'helpers')

    def basis_for(self, tl, Wd):
        """matrix_basis of an object from its decomposed world matrix (drops scale for cameras/lights)."""
        if tl.id in self.unit_scale_ids:
            loc, rot, _sca = Wd.decompose()
            Wd = Matrix.LocRotScale(loc, rot, (1.0, 1.0, 1.0))
        fix = self.local_fix.get(tl.id)
        return Wd @ fix if fix is not None else Wd

    def distribute_collections(self):
        """Move every created object into its Suite category collection (Characters, Items, ...)."""
        cc = self.opt.category_collections
        if not cc:
            return
        for tid, obj in self.objects.items():
            tl = self.project.timelines.get(tid)
            if tl is None:
                continue
            target = cc.get(self.category_of(tl))
            if target is None or target in obj.users_collection[:]:
                continue
            for current in obj.users_collection[:]:
                current.objects.unlink(obj)
            target.objects.link(obj)

    def run(self, context, collection=None):
        proj = self.project
        opt = self.opt
        self.log("Project: %s | %s | format %d%s" % (
            proj.name, proj.version_label, proj.file_format,
            '' if opt.version == 'AUTO' else ' (rules of %s)' % dict((v[0], v[1]) for v in mi_data.VERSIONS)[opt.version]))
        if self.assets.ok:
            self.log("Minecraft resources: %s" % self.assets.version)

        scene = context.scene
        if collection is not None:
            coll = collection   # the Suite creates the (uniquely named) import collection itself
        else:
            coll = bpy.data.collections.new(proj.name)
            scene.collection.children.link(coll)

        states = mi_data.evaluate(proj, opt.frame, self.part_lookup, ik=opt.ik)

        # Que timelines se crean (geometria o necesarios para la jerarquia)
        needed = {}

        cats = opt.categories

        def visit(tl):
            if tl.type == 'folder':
                keep = cats is None or 'helpers' in cats   # folders are structure: also kept for their children
            else:
                keep = self.wants_type(tl) and (cats is None or self.category_of(tl) in cats)
                if not keep and self.wants_type(tl):
                    self.skipped_category[self.category_of(tl)] = self.skipped_category.get(self.category_of(tl), 0) + 1
            for c in tl.children:
                if visit(c):
                    keep = True
            needed[tl.id] = keep
            return keep
        for r in proj.roots():
            visit(r)

        # Recorrido en árbol: los padres siempre se crean antes que sus hijos
        tree = []
        stack = list(reversed(proj.roots()))
        while stack:
            t = stack.pop()
            tree.append(t)
            stack.extend(reversed(t.children))

        # Esqueletos: una armadura por modelo (personaje o bloque especial). Los .mimodel
        # sueltos no llevan armadura: el bend se calcula con la geometría exacta de
        # Mine-imator (igual que la exportación de Modelbench), no con skinning.
        rig_roots = {}
        is_mimodel = os.path.splitext(proj.path)[1].lower() == '.mimodel'
        if is_mimodel and opt.rig:
            self.log(".mimodel: imported without an armature (exact Mine-imator bend)")
        if opt.rig and not is_mimodel:
            for tl in proj.order:
                if tl.type == 'bodypart' and tl.part_of and needed.get(tl.id) and self.part_lookup(tl) is not None:
                    root = proj.timelines.get(tl.part_of)
                    if root is not None and root.type in ('char', 'spblock', 'model'):
                        rig_roots.setdefault(root.id, []).append(tl)
            # Carpetas y primitivas que cuelgan de las partes del modelo (o de otras carpetas
            # del rig) también pasan a ser huesos de la misma armadura.
            owner = {t.id: rid for rid, ts in rig_roots.items() for t in ts}
            for tl in tree:
                if not tl.parent or not needed.get(tl.id):
                    continue
                if not (tl.type in ('folder', 'item', 'block') or tl.type in mi_data.SHAPE_TYPES):
                    continue
                rid = owner.get(tl.parent)
                if rid is not None:
                    owner[tl.id] = rid
                    rig_roots[rid].append(tl)
        # La escala de los miembros con IK (brazos con SCA 1,25...) no se aplica: el IK de Blender
        # no puede estirar la cadena y la pose se desplazaría; sin escala el IK reproduce la pose.
        self.no_scale = set()
        if rig_roots and opt.ik and opt.ik_keep_pose:
            for ts in rig_roots.values():
                for tl in ts:
                    if tl.type != 'bodypart' or not mi_data.supports_ik(self.part_lookup(tl)):
                        continue
                    if tl.value_default.get('IK_TARGET') or any(kf.values.get('IK_TARGET') for kf in tl.keyframes):
                        self.no_scale.add(tl.id)
            if self.no_scale:
                states = mi_data.evaluate(proj, opt.frame, self.part_lookup, ik=opt.ik, no_scale=self.no_scale)
                self.log("Scale ignored on %d IK limbs (Blender IK cannot stretch the chain)"
                         % len(self.no_scale))
        rest_states = mi_data.evaluate(proj, opt.frame, self.part_lookup, rest_parts=True) if rig_roots else None
        self.coll = coll
        self.frame_states = states

        created = 0
        for tl in tree:
            if not needed.get(tl.id):
                self.skipped[tl.type] = self.skipped.get(tl.type, 0) + 1
                continue
            st = states[tl.id]
            visible = (not tl.hide) and (st.inherit.get('VISIBLE', 1) or 0) > 0
            if not visible and not opt.import_hidden:
                continue
            W = self.conv.matrix(st.matrix_render)
            Wd, residual = decompose(W)
            name = self.object_name(tl)

            rig_part = tl.id in self.bone_of
            data = None
            if tl.id in rig_roots:
                data = bpy.data.armatures.new(name)
                data.display_type = 'STICK'  # bastones
            elif rig_part:
                rig = self.rigs[self.bone_of[tl.id]]
                bone, bend_bone = rig['bones'][tl.id][:2]
                try:
                    # Con "doblado con pesos" toda parte con bend lleva los dos huesos y sus pesos
                    # (editables). Si no, el bend que no cambia en todo el proyecto se hornea con la
                    # geometría exacta de Mine-imator (el skinning lineal pierde volumen en doblados fuertes)
                    polys = self.geometry(tl, st, rig=self.opt.bend_weights or not self.static_bend(tl))
                except Exception as e:
                    self.log("Error generating '%s' (%s): %s" % (name, tl.type, e))
                    polys = None
                if polys:
                    for p in polys:
                        p.mat = (p.mat or MatKey()).with_cull(not tl.backfaces)
                    data = self.make_mesh(name, polys, rig['rest_rel'][tl.id], groups=(bone, bend_bone))
                    if data is not None:
                        self.obj_alpha[tl.id] = max((p.mat.alpha if p.mat is not None else 1.0) for p in polys)
                if data is None:
                    continue  # la parte solo existe como hueso
                obj = bpy.data.objects.new(name, data)
                coll.objects.link(obj)
                self.assign_groups(obj)
                obj.parent = rig['obj']
                mod = obj.modifiers.new('Armature', 'ARMATURE')
                mod.object = rig['obj']
                obj['mineimator_id'] = tl.id
                obj['mineimator_type'] = tl.type
                self.apply_visibility(obj, tl, visible)
                self.objects[tl.id] = obj
                self.rig_meshes.add(tl.id)
                created += 1
                continue
            elif self.extras is not None and self.extras.handles(tl.type):
                try:
                    data = self.extras.make_data(self, tl, st, name)
                except Exception as e:  # a broken camera/light/scenery must not stop the whole import
                    self.log("Error creating '%s' (%s): %s" % (name, tl.type, e))
                    data = None
            elif tl.type == 'text':
                data = bpy.data.curves.new(name, 'FONT')
                data.body = str(st.values.get('TEXT') or tl.text or name)
                data.size = 8 * opt.unit_scale
                data.align_x = 'CENTER'
                data.extrude = 0.5 * opt.unit_scale
            else:
                try:
                    polys = self.geometry(tl, st)
                except Exception as e:  # una pieza rota no debe parar la conversion
                    self.log("Error generating '%s' (%s): %s" % (name, tl.type, e))
                    polys = None
                if polys:
                    # Mine-imator oculta las caras traseras salvo que el timeline active "backfaces"
                    cull = not tl.backfaces
                    for p in polys:
                        p.mat = (p.mat or MatKey()).with_cull(cull)
                    data = self.make_mesh(name, polys, residual)
                    if data is not None:
                        self.obj_alpha[tl.id] = max((p.mat.alpha if p.mat is not None else 1.0) for p in polys)
                elif tl.type == 'scenery':
                    self.skipped['scenery'] = self.skipped.get('scenery', 0) + 1

            obj = bpy.data.objects.new(name, data)
            if data is None:
                obj.empty_display_type = 'PLAIN_AXES'
                obj.empty_display_size = 4 * opt.unit_scale
            coll.objects.link(obj)

            # Padre: otro objeto o un hueso de una armadura
            link = self.parent_link_for(tl)
            if link is not None:
                pf = self.parent_frame(link, self.cur_worlds(), self.cur_poses())
                if link[0] == 'obj':
                    obj.parent = self.objects[link[1]]
                else:
                    obj.parent = self.rigs[link[1]]['obj']
                    obj.parent_type = 'BONE'
                    obj.parent_bone = link[2]
                obj.matrix_parent_inverse = pf.inverted_safe()
                self.links[tl.id] = link
                self.base_pf[tl.id] = pf
            if tl.type == 'text':
                self.local_fix[tl.id] = Matrix.Rotation(math.radians(90), 4, 'X')
            elif self.extras is not None and self.extras.handles(tl.type):
                fix, drop_scale = self.extras.fix_for(self, tl)
                if fix is not None:
                    self.local_fix[tl.id] = fix
                if drop_scale:
                    self.unit_scale_ids.add(tl.id)
            obj.matrix_basis = self.basis_for(tl, Wd)
            obj['mineimator_id'] = tl.id
            obj['mineimator_type'] = tl.type
            self.apply_visibility(obj, tl, visible)
            self.objects[tl.id] = obj
            self.base_world[tl.id] = Wd
            created += 1
            self.stats[tl.type] = self.stats.get(tl.type, 0) + 1
            if self.extras is not None and self.extras.handles(tl.type):
                try:
                    self.extras.finish(self, tl, st, obj)
                except Exception as e:
                    self.log("Error finishing '%s' (%s): %s" % (name, tl.type, e))

            if tl.id in rig_roots:
                self.build_rig(tl, obj, rig_roots[tl.id], rest_states)

        if self.rigs:
            self.pose_rigs(states)
            if opt.ik:
                self.add_ik(states)

        if opt.depth_masks:
            self.apply_depth_masks()

        if self.extras is not None:
            self.extras.finalize(self)
        self.distribute_collections()

        if opt.animation:
            self.bake_animation(scene)

        if opt.pack_textures:
            for img in self.textures.images.values():
                if img is not None and not img.packed_file:
                    try:
                        img.pack()
                    except RuntimeError:
                        pass

        scene.render.fps = int(proj.tempo) if proj.tempo else 24
        if opt.standard_view:
            try:
                scene.view_settings.view_transform = 'Standard'  # colores como en Mine-imator
            except TypeError:
                pass
        self.log("Objects created: %d" % created)
        if self.skipped:
            self.log("Not converted (no 3D geometry or unsupported): " +
                     ', '.join('%s x%d' % (k or '?', v) for k, v in sorted(self.skipped.items())))
        self.assets.close()
        return coll

    def apply_depth_masks(self):
        """Emula el truco de profundidad de Mine-imator.

        Mine-imator dibuja los objetos por orden de "depth" con test de
        profundidad. Un objeto casi invisible (alfa <= 0.05) dibujado antes que
        otro escribe en el buffer de profundidad y hace un "agujero" en los
        objetos que se dibujan después (p. ej. cuencas de los ojos que recortan
        la cara para que se vean los ojos). En Blender se reproduce con un
        modificador Boolean de diferencia.
        """
        proj = self.project
        masks = [tid for tid, a in self.obj_alpha.items()
                 if 0.001 < a <= 0.05 and tid in self.objects and self.objects[tid].type == 'MESH'
                 and not self.objects[tid].hide_viewport and not proj.timelines[tid].hq_hiding]
        if not masks:
            return

        def world_bbox(obj):
            pts = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
            return (Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts))),
                    Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts))))

        bpy.context.view_layer.update()
        boxes = {tid: world_bbox(o) for tid, o in self.objects.items() if o.type == 'MESH'}
        count = 0
        for mid in masks:
            mobj = self.objects[mid]
            mdepth = proj.timelines[mid].depth
            mn, mx = boxes[mid]
            used = False
            for tid, obj in self.objects.items():
                if tid == mid or obj.type != 'MESH' or tid in masks:
                    continue
                if proj.timelines[tid].depth <= mdepth:
                    continue
                if self.obj_alpha.get(tid, 1.0) <= 0.05:
                    continue
                bmn, bmx = boxes[tid]
                if any(bmx[i] < mn[i] or bmn[i] > mx[i] for i in range(3)):
                    continue
                mod = obj.modifiers.new('Máscara ' + mobj.name, 'BOOLEAN')
                mod.operation = 'DIFFERENCE'
                mod.object = mobj
                if hasattr(mod, 'solver'):
                    try:
                        mod.solver = 'EXACT'
                    except TypeError:
                        pass
                used = True
                count += 1
            if used:
                mobj.display_type = 'WIRE'
                mobj.hide_render = True
                mobj['mineimator_depth_mask'] = True
        if count:
            self.log("Depth masks emulated: %d cut-outs" % count)

    # -- visibilidad y jerarquia ------------------------------------------------

    def apply_visibility(self, obj, tl, visible):
        if not visible:
            obj.hide_viewport = True
            obj.hide_render = True
        if tl.hq_hiding:
            obj.hide_render = True  # oculto en el render de alta calidad de Mine-imator
        if tl.lq_hiding:
            obj.hide_viewport = True  # oculto en la vista previa de Mine-imator
        if self.obj_alpha.get(tl.id, 1.0) <= 0.001:
            obj.hide_render = True  # alfa 0: Mine-imator no lo dibuja
            obj.display_type = 'WIRE'
        # Las superficies interiores (formas invertidas, como el fondo de las cuencas de los ojos) y
        # las máscaras casi invisibles no proyectan sombra: si no, tapan la luz a lo que hay dentro
        # (los ojos se veían negros en Cycles).
        no_shadow = self.obj_alpha.get(tl.id, 1.0) <= 0.05
        if not no_shadow and tl.type == 'bodypart':
            part = self.part_lookup(tl)
            no_shadow = part is not None and any(getattr(s, 'invert', False) for s in part.shapes)
        if not no_shadow and tl.type in mi_data.SHAPE_TYPES:
            temp = self.project.templates.get(tl.temp)
            no_shadow = temp is not None and temp.shape_invert
        if no_shadow:
            obj.visible_shadow = False

    def parent_link_for(self, tl):
        p = tl.parent
        if not p:
            return None
        if p in self.bone_of:
            root = self.bone_of[p]
            bone, bend = self.rigs[root]['bones'][p][:2]
            return ('bone', root, bend if (bend and tl.lock_bend) else bone)
        if p in self.objects and p not in self.rig_meshes:
            return ('obj', p)
        return None

    def cur_worlds(self):
        return self.base_world

    def cur_poses(self):
        return {(r, n): m for r, rig in self.rigs.items() for n, m in rig['pose'].items()}

    def parent_frame(self, link, worlds, poses):
        if link[0] == 'obj':
            return self.basis_for(self.project.timelines[link[1]], worlds[link[1]])
        root, bone = link[1], link[2]
        L = self.rigs[root]['length'][bone]
        return worlds[root] @ poses[(root, bone)] @ Matrix.Translation((0.0, L, 0.0))

    # -- esqueleto ---------------------------------------------------------

    def static_bend(self, tl):
        """¿El doblado de la parte vale lo mismo en todos los fotogramas y no lo mueve un IK?"""
        if tl.type != 'bodypart':
            return False
        part = self.part_lookup(tl)
        if part is None or part.bend is None:
            return False
        proj = self.project
        chain = []
        cur = tl
        while cur is not None:
            chain.append(cur)
            if not cur.inherit_bend or not cur.parent:
                break
            cur = proj.timelines.get(cur.parent)
        frames = {0}
        for t in chain:
            frames.update(t.keyframe_positions())
            for kf in t.keyframes:
                if kf.values.get('IK_TARGET'):
                    return False
            if t.value_default.get('IK_TARGET'):
                return False
        seen = set()
        for f in frames:
            bx = by = bz = 0.0
            for t in chain:
                v = t.values_at(f)
                bx += v.get('BEND_ANGLE_X', 0.0) or 0.0
                by += v.get('BEND_ANGLE_Y', 0.0) or 0.0
                bz += v.get('BEND_ANGLE_Z', 0.0) or 0.0
            seen.add((round(bx, 3), round(by, 3), round(bz, 3)))
            if len(seen) > 1:
                return False
        return True

    def _bone_shape(self, part):
        """Dirección (MI) y longitudes (unidades MI) de los huesos de una parte."""
        dirs = {'lower': (0, 0, -1), 'upper': (0, 0, 1), 'left': (-1, 0, 0),
                'right': (1, 0, 0), 'back': (0, -1, 0), 'front': (0, 1, 0)}
        corners = []
        for sh in part.shapes:
            rotm = gm.rotation(*sh.rotation)
            for cx in (sh.frm[0], sh.to[0]):
                for cy in (sh.frm[1], sh.to[1]):
                    for cz in (sh.frm[2], sh.to[2]):
                        p = gm.transform_point(rotm, (cx, cy, cz))
                        corners.append((p[0] + sh.position[0], p[1] + sh.position[1], p[2] + sh.position[2]))
        if part.bend is not None:
            d = dirs[part.bend.part]
        else:
            d = None
            kids = [c.position for c in part.parts if any(abs(v) > 1e-4 for v in c.position)]
            if kids:
                avg = [sum(k[i] for k in kids) / len(kids) for i in range(3)]
                n = math.sqrt(sum(a * a for a in avg))
                if n > 1e-4:
                    d = tuple(a / n for a in avg)
            if d is None:
                best, d = -1.0, (0, 0, 1)
                for cand in dirs.values():
                    ext = max((sum(c[i] * cand[i] for i in range(3)) for c in corners), default=0.0)
                    if ext > best + 1e-6:
                        best, d = ext, cand
        ext = max((sum(c[i] * d[i] for i in range(3)) for c in corners), default=0.0)
        if part.bend is not None:
            piv = [0.0, 0.0, 0.0]
            piv[geo.PART_AXIS[part.bend.part]] = part.bend.offset
            dist = sum(piv[i] * d[i] for i in range(3))
            lower = ext - dist
            if mi_data.supports_ik(part):
                # Mine-imator resuelve el IK con el extremo a end_offset del pivote
                lower = part.bend.end_offset
            return d, max(dist, 0.5), max(lower, 0.5), piv
        return d, max(ext, 1.0), None, None

    def build_rig(self, root, arm_obj, parts, rest_states):
        u = self.conv.unit
        A_rest, _r = decompose(self.conv.matrix(rest_states[root.id].matrix_render))
        A_inv = A_rest.inverted_safe()
        rig = {'obj': arm_obj, 'bones': {}, 'rest_rel': {}, 'length': {}, 'rest': {}, 'parent': {},
               'order': [], 'pose': {}}
        self.rigs[root.id] = rig
        used = set()

        def uniq(n):
            n = (n or 'parte')[:55]
            base, i = n, 1
            while n in used:
                n = '%s.%03d' % (base, i)
                i += 1
            used.add(n)
            return n

        spec = []
        for b in parts:
            # Partes de modelo (con forma y bend) o elementos del rig (carpetas, primitivas...)
            part = self.part_lookup(b) if b.type == 'bodypart' else None
            R = A_inv @ self.conv.matrix(rest_states[b.id].matrix_render)
            loc, rot, _s = R.decompose()
            Fr = Matrix.LocRotScale(loc, rot, Vector((1, 1, 1)))
            if part is not None:
                d, L1, L2, piv = self._bone_shape(part)
                db = Vector((d[0], -d[1], d[2])).normalized()
                raxis = Vector((0, 1, 0)).rotation_difference(db).to_matrix().to_4x4()

            else:
                L1, L2, piv = 4.0, None, None
                raxis = Matrix.Identity(4)
            name = uniq(self.object_name(b))
            bend_name = uniq(name + '_bend') if (part is not None and part.bend is not None) else None
            Tp = Matrix.Translation((piv[0] * u, -piv[1] * u, piv[2] * u)) if piv else None
            rig['bones'][b.id] = (name, bend_name, raxis, Tp, part)
            rig['length'][name] = L1 * u
            if bend_name:
                rig['length'][bend_name] = L2 * u
            spec.append((b, name, bend_name, Fr @ raxis, (Fr @ Tp @ raxis) if Tp else None))
            self.bone_of[b.id] = root.id

        ids = {b.id for b in parts}
        for b, name, bend_name, _m, _mb in spec:
            pname = None
            if b.parent in ids:
                pbone, pbend = rig['bones'][b.parent][:2]
                pname = pbend if (pbend and b.lock_bend) else pbone
            rig['parent'][name] = pname
            if bend_name:
                rig['parent'][bend_name] = name

        # Huesos en modo edición
        view_layer = bpy.context.view_layer
        hidden = arm_obj.hide_viewport
        arm_obj.hide_viewport = False
        prev_active = view_layer.objects.active
        view_layer.objects.active = arm_obj
        bpy.ops.object.mode_set(mode='EDIT')
        ebs = arm_obj.data.edit_bones
        for b, name, bend_name, m, mb in spec:
            for n, mat in ((name, m), (bend_name, mb)):
                if n is None:
                    continue
                eb = ebs.new(n)
                eb.head = (0, 0, 0)
                eb.tail = (0, 1, 0)
                eb.matrix = mat
                eb.length = max(rig['length'][n], 1e-3)
        for n, pn in rig['parent'].items():
            if pn:
                ebs[n].parent = ebs[pn]
                ebs[n].use_connect = False
        bpy.ops.object.mode_set(mode='OBJECT')
        view_layer.objects.active = prev_active
        arm_obj.hide_viewport = hidden

        for bone in arm_obj.data.bones:
            rig['rest'][bone.name] = bone.matrix_local.copy()
            rig['length'][bone.name] = bone.length
        # orden padres -> hijos
        order, seen = [], set()

        def visit(n):
            if n in seen:
                return
            p = rig['parent'].get(n)
            if p:
                visit(p)
            seen.add(n)
            order.append(n)
        for n in rig['parent']:
            visit(n)
        rig['order'] = order
        rig['pose'] = self.rig_poses(root.id, self.frame_states, self.base_world[root.id])
        # Malla en espacio de reposo: la que, deformada por la pose del fotograma importado,
        # queda exactamente donde la pone Mine-imator (escala y cizallamiento incluidos).
        # Se usa la pose que cada hueso alcanza de verdad (con escala en la cadena, p. ej. el IK
        # de los brazos, difiere de la objetivo): así la pieza queda exacta en el fotograma importado.
        self.rig_bases(rig, rig['pose'])
        reached = rig['reached']
        A_inv_f = self.base_world[root.id].inverted_safe()
        for b in parts:
            name = rig['bones'][b.id][0]
            K = reached[name] @ rig['rest'][name].inverted_safe()
            rig['rest_rel'][b.id] = K.inverted_safe() @ A_inv_f @ self.conv.matrix(self.frame_states[b.id].matrix_render)
        arm_obj['mineimator_rig'] = True
        arm_obj.show_in_front = True  # huesos visibles a través de la malla

    def rig_poses(self, root_id, states, A):
        """Matrices de pose (espacio de la armadura) de cada hueso."""
        rig = self.rigs[root_id]
        A_inv = A.inverted_safe()
        F = Matrix.Diagonal((1.0, -1.0, 1.0, 1.0))
        poses = {}
        for tid, (bone, bend, raxis, Tp, part) in rig['bones'].items():
            D, _r = decompose(A_inv @ self.conv.matrix(states[tid].matrix_render))
            poses[bone] = D @ raxis
            if bend:
                bv = part._limit_bend(mi_data._vec_bend(states[tid].inherit))
                Rb = F @ gm_to_matrix(gm.build(0, 0, 0, bv[0], bv[1], bv[2], 1, 1, 1)) @ F
                poses[bend] = D @ Tp @ Rb @ raxis
        return poses

    def rig_bases(self, rig, poses):
        """matrix_basis de cada hueso para que alcance su matriz de pose.

        Blender reduce cada matrix_basis a posición/rotación/escala. Bajo un padre con escala
        no uniforme, un hijo girado necesitaría cizallamiento, que un hueso no admite; esos
        huesos se marcan (la primera vez) para no heredar escala (inherit_scale NONE), con lo
        que vuelven a ser exactos. Cada hijo se calcula respecto a la matriz que su padre
        alcanza de verdad, para que el error no se acumule.
        """
        noscale = rig.setdefault('noscale', {})
        out = {}
        got = {}
        for n in rig['order']:
            W = poses[n]
            rest = rig['rest'][n]
            p = rig['parent'].get(n)
            if p is None:
                loc, rot, sca = (rest.inverted_safe() @ W).decompose()
                out[n] = Matrix.LocRotScale(loc, rot, sca)
                got[n] = rest @ out[n]
                continue
            M = got[p] @ rig['rest'][p].inverted_safe() @ rest

            def full():
                loc, rot, sca = (M.inverted_safe() @ W).decompose()
                b = Matrix.LocRotScale(loc, rot, sca)
                return b, M @ b

            def none():
                # inherit_scale NONE: pose = T(M · loc) · (norm(padre) · offset3 · basis3)
                q3 = got[p].normalized().to_3x3() @ (rig['rest'][p].inverted_safe() @ rest).to_3x3()
                loc = M.inverted_safe() @ W.to_translation()
                _l, rot, sca = (q3.inverted_safe() @ W.to_3x3()).to_4x4().decompose()
                b = Matrix.LocRotScale(loc, rot, sca)
                return b, Matrix.Translation(M @ loc) @ (q3 @ b.to_3x3()).to_4x4()

            def err(G):
                return max(abs(G[i][j] - W[i][j]) for i in range(3) for j in range(4))

            if n in noscale:
                out[n], got[n] = none() if noscale[n] else full()
                continue
            bf, gf = full()
            if err(gf) < 1e-4:
                noscale[n] = False
                out[n], got[n] = bf, gf
                continue
            bn, gn = none()
            noscale[n] = err(gn) < err(gf)
            out[n], got[n] = (bn, gn) if noscale[n] else (bf, gf)
        rig['reached'] = got
        return out

    def pose_rigs(self, states):
        for root_id, rig in self.rigs.items():
            arm = rig['obj']
            bases = self.rig_bases(rig, rig['pose'])
            for n, flag in rig['noscale'].items():
                if flag:
                    arm.data.bones[n].inherit_scale = 'NONE'
            for n, basis in bases.items():
                pb = arm.pose.bones[n]
                pb.rotation_mode = 'QUATERNION'
                pb.matrix_basis = basis

    def add_ik(self, states):
        """Restricciones IK en las extremidades que lo usan en Mine-imator.

        La pose importada ya es la que resolvió Mine-imator y se mantiene. Para cada cadena
        se busca una configuración del IK de Blender (con polo, o sin polo y con una flexión
        previa mínima) que (1) no cambie esa pose y (2) pliegue hacia el lado del polo
        cuando el objetivo se acerca. Si no existe, la restricción se deja silenciada."""
        view_layer = bpy.context.view_layer
        made = muted = pose_muted = 0
        # sin mallas visibles la búsqueda del IK evalúa solo huesos (mucho más rápido)
        hidden = [(o, o.hide_viewport) for o in bpy.data.objects if o.type == 'MESH']
        for o, _h in hidden:
            o.hide_viewport = True
        for root_id, rig in self.rigs.items():
            arm = rig['obj']
            A_inv = self.base_world[root_id].inverted_safe()
            for tid, (bone, bend, raxis, Tp, part) in rig['bones'].items():
                if not bend or not mi_data.supports_ik(part):
                    continue
                v = states[tid].values
                target = v.get('IK_TARGET')
                if not target or target not in self.objects or target in self.rig_meshes:
                    continue
                pole = v.get('IK_TARGET_ANGLE')
                pole_obj = self.objects[pole] if (pole and pole in self.objects and pole not in self.rig_meshes) else None
                pole_loc = (A_inv @ self.base_world[pole]).to_translation() if pole_obj is not None else None
                c = arm.pose.bones[bend].constraints.new('IK')
                c.target = self._ik_goal(arm, rig, bend, self.objects[target])
                c.chain_count = 2
                made += 1
                pbb = arm.pose.bones[bend]
                base_basis = pbb.matrix_basis.copy()
                if not self._configure_ik(arm, rig, bone, bend, c, pole_obj, pole_loc, view_layer):
                    c.mute = True
                    muted += 1
                    view_layer.update()
                elif self.opt.ik_keep_pose:
                    # Con escala en la cadena (brazos) el IK de Blender no mantiene la pose: la mano
                    # se desplaza y se abre un hueco con el antebrazo. Se prioriza la pose exacta.
                    L = rig['length'][bend]
                    ref, now = rig['reached'][bend], pbb.matrix
                    disp = max((now @ Vector(p) - ref @ Vector(p)).length for p in ((0, 0, 0), (0, L, 0)))
                    chain = rig['length'][bone] + L
                    if disp / max(chain, 1e-6) > 0.02:
                        pbb.matrix_basis = base_basis
                        c.mute = True
                        pose_muted += 1
                        view_layer.update()
        for o, h in hidden:
            o.hide_viewport = h
        view_layer.update()
        if made:
            self.log("IK constraints created: %d" % made)
        if muted:
            self.log("IK disabled on %d limbs (no stable configuration was found)" % muted)
        if pose_muted:
            self.log("IK muted on %d limbs because it altered the imported pose (scaled chain); "
                     "enable it on the _bend bone or turn off 'Keep pose'" % pose_muted)

    def _ik_goal(self, arm, rig, bend, cube):
        """Objetivo del IK: un empty hijo del cubo colocado en la punta de la pose importada.

        En Mine-imator la punta no coincide con el centro del cubo; si el IK de Blender apuntara
        al cubo, tiraría de la mano al importar. Con el empty la pose de partida no cambia y el
        objetivo sigue al cubo igualmente."""
        bpy.context.view_layer.update()  # matrix_world al día antes de calcular la inversa del padre
        tip = arm.matrix_world @ (rig['reached'][bend] @ Vector((0.0, rig['length'][bend], 0.0)))
        goal = bpy.data.objects.new(cube.name + ' (IK)', None)
        goal.empty_display_type = 'SPHERE'
        goal.empty_display_size = max(cube.dimensions) * 0.15 if max(cube.dimensions) > 0 else 0.1
        colls = cube.users_collection
        (colls[0] if colls else bpy.context.scene.collection).objects.link(goal)
        goal.matrix_world = Matrix.Translation(tip)
        goal.parent = cube
        goal.matrix_parent_inverse = cube.matrix_world.inverted_safe()
        goal.hide_render = True
        bpy.context.view_layer.update()
        return goal

    def _ik_deviation(self, arm, rig, bone, bend):
        """Cuánto se aleja la cadena de la pose importada (relativo a la longitud del hueso)."""
        worst = 0.0
        for n in (bone, bend):
            G, W = arm.pose.bones[n].matrix, rig['reached'][n]
            L = max(rig['length'][n], 1e-6)
            for p in ((0, 0, 0), (0, L, 0), (L, 0, 0), (0, 0, L)):
                worst = max(worst, (G @ Vector(p) - W @ Vector(p)).length / L)
        return worst

    def _ik_folds_to_pole(self, arm, rig, bone, bend, c, pole_loc, view_layer):
        """¿Al acercar el objetivo, la articulación se desplaza hacia el lado del polo?"""
        pu, pb = arm.pose.bones[bone], arm.pose.bones[bend]
        cube = c.target
        saved = cube.matrix_world.copy()
        hip = pu.matrix.translation.copy()
        tgt = arm.matrix_world.inverted() @ saved.translation
        try:
            m = saved.copy()
            m.translation = arm.matrix_world @ (hip + (tgt - hip) * 0.6)
            cube.matrix_world = m
            view_layer.update()
            knee = pb.matrix.translation
            tip = pb.matrix @ Vector((0, rig['length'][bend], 0))
            a = tip - hip
            if a.length < 1e-6:
                return False
            a.normalize()
            off = (knee - hip) - a * (knee - hip).dot(a)
            pd = pole_loc - hip
            pd -= a * pd.dot(a)
            return off.length > 1e-6 and pd.length > 1e-6 and off.dot(pd) > 0
        finally:
            cube.matrix_world = saved
            view_layer.update()
    def _configure_ik(self, arm, rig, bone, bend, c, pole_obj, pole_loc, view_layer):
        """Busca la configuración del IK de Blender que conserva la pose importada y pliega
        hacia el polo al acercar el objetivo.

        Una cadena perfectamente recta no se pliega en Blender, así que se parte de una
        flexión previa mínima (probada hacia los dos lados); para cada una se barre el
        ángulo del polo (grueso y luego fino) y se elige el que menos altera la pose."""
        pu, pb = arm.pose.bones[bone], arm.pose.bones[bend]
        # La pose que el hueso alcanza de verdad (con escala IK en la cadena difiere de la objetivo
        # por el cizallamiento): es la referencia y a ella se vuelve, no a la matriz objetivo.
        base_basis = pb.matrix_basis.copy()
        imported = rig['reached'][bend].copy()
        # Con escala en la cadena el IK de Blender no reproduce la pose exactamente: se admite una
        # desviación pequeña (se prima que el polo y el IK funcionen). Si no hay una configuración
        # tan cercana, se descarta el polo o se silencia la restricción.
        tol_exact, tol_loose = 0.06, 0.3
        view_layer.update()

        def set_prebend(sign):
            pb.matrix_basis = base_basis
            view_layer.update()
            if pole_loc is None:
                return
            hip, knee = pu.matrix.translation.copy(), pb.matrix.translation.copy()
            tip = pb.matrix @ Vector((0, rig['length'][bend], 0))
            a = tip - hip
            if a.length < 1e-6 or (knee - hip).length < 1e-6 or (tip - knee).length < 1e-6:
                return
            a.normalize()
            if (knee - hip).angle(tip - knee, 0.0) > math.radians(3.0):
                return  # ya está doblada
            p = pole_loc - hip
            p -= a * a.dot(p)
            if p.length < 1e-6:
                return
            T = Matrix.Translation(knee)
            pb.matrix = T @ Matrix.Rotation(sign * math.radians(1.5), 4, a.cross(p.normalized())) @ T.inverted() @ imported
            view_layer.update()

        def dev_at(ang, sign):
            # el solver conserva memoria entre evaluaciones: se parte siempre del mismo estado
            set_prebend(sign)
            c.pole_angle = math.atan2(math.sin(ang), math.cos(ang))
            view_layer.update()
            return self._ik_deviation(arm, rig, bone, bend)

        if pole_obj is not None:
            c.pole_target = pole_obj
            found = []
            for sign in (1.0, -1.0):
                coarse = sorted((dev_at(-math.pi + k * math.pi / 12, sign), -math.pi + k * math.pi / 12)
                                for k in range(24))
                for d0, a0 in coarse[:2]:
                    if d0 > tol_loose:
                        continue
                    for j in range(-6, 7):
                        a = a0 + j * math.radians(2.5)
                        found.append((dev_at(a, sign) if j else d0, a, sign))
            found.sort()
            for d, a, sign in found[:8]:
                if d > tol_loose:
                    break
                dev_at(a, sign)
                set_prebend(sign)
                c.pole_angle = math.atan2(math.sin(a), math.cos(a))
                view_layer.update()
                if self._ik_folds_to_pole(arm, rig, bone, bend, c, pole_loc, view_layer):
                    set_prebend(sign)
                    c.pole_angle = math.atan2(math.sin(a), math.cos(a))
                    view_layer.update()
                    return True
            c.pole_target = None
        # sin polo: la flexión previa decide el plano; basta con no alterar la pose
        for sign in (1.0, -1.0):
            set_prebend(sign)
            if self._ik_deviation(arm, rig, bone, bend) <= tol_exact:
                return True
        pb.matrix_basis = base_basis
        view_layer.update()
        return False
    # -- animacion -----------------------------------------------------------

    def bake_animation(self, scene):
        proj = self.project
        start, end = proj.frame_range()
        if end <= start:
            self.log("The project has no animation (only one keyframe).")
            return
        scene.frame_start = start
        scene.frame_end = end
        step = self.opt.anim_step
        frames = list(range(start, end + 1, step))
        if frames[-1] != end:
            frames.append(end)
        last = {}
        last_bone = {}
        movable = [tid for tid in self.objects if tid not in self.rig_meshes]
        for f in frames:
            states = mi_data.evaluate(proj, f, self.part_lookup, ik=self.opt.ik, no_scale=getattr(self, 'no_scale', ()))
            worlds = {}
            for tid in movable:
                worlds[tid], _r = decompose(self.conv.matrix(states[tid].matrix_render))
            poses = {}
            for root_id, rig in self.rigs.items():
                rp = self.rig_poses(root_id, states, worlds[root_id])
                rig['frame_pose'] = rp
                for n, m in rp.items():
                    poses[(root_id, n)] = m
            for tid in movable:
                obj = self.objects[tid]
                tl = proj.timelines[tid]
                Wd = self.basis_for(tl, worlds[tid])
                link = self.links.get(tid)
                if link is not None:
                    basis = self.base_pf[tid] @ self.parent_frame(link, worlds, poses).inverted_safe() @ Wd
                else:
                    basis = Wd
                loc, rot, sca = basis.decompose()
                prev = last.get(tid)
                if obj.rotation_mode != 'QUATERNION':
                    obj.rotation_mode = 'QUATERNION'
                if prev is not None:
                    rot.make_compatible(prev[1])
                obj.location = loc
                obj.rotation_quaternion = rot
                obj.scale = sca
                vis = (not tl.hide) and (states[tid].inherit.get('VISIBLE', 1) or 0) > 0
                changed = prev is None or (loc - prev[0]).length > 1e-6 or rot.rotation_difference(prev[1]).angle > 1e-5 \
                    or (sca - prev[2]).length > 1e-6 or f == frames[-1]
                if changed:
                    obj.keyframe_insert('location', frame=f)
                    obj.keyframe_insert('rotation_quaternion', frame=f)
                    obj.keyframe_insert('scale', frame=f)
                if prev is None or prev[3] != vis:
                    obj.hide_viewport = not vis
                    obj.hide_render = not vis
                    obj.keyframe_insert('hide_viewport', frame=f)
                    obj.keyframe_insert('hide_render', frame=f)
                last[tid] = (loc.copy(), rot.copy(), sca.copy(), vis)
            # Huesos
            for root_id, rig in self.rigs.items():
                arm = rig['obj']
                for n, basis in self.rig_bases(rig, rig['frame_pose']).items():
                    pb = arm.pose.bones[n]
                    loc, rot, sca = basis.decompose()
                    prev = last_bone.get((root_id, n))
                    if prev is not None:
                        rot.make_compatible(prev[1])
                    changed = prev is None or (loc - prev[0]).length > 1e-6 or \
                        rot.rotation_difference(prev[1]).angle > 1e-5 or (sca - prev[2]).length > 1e-6 or f == frames[-1]
                    if changed:
                        pb.location = loc
                        pb.rotation_quaternion = rot
                        pb.scale = sca
                        pb.keyframe_insert('location', frame=f)
                        pb.keyframe_insert('rotation_quaternion', frame=f)
                        pb.keyframe_insert('scale', frame=f)
                    last_bone[(root_id, n)] = (loc.copy(), rot.copy(), sca.copy())
            # Partes de modelo: solo visibilidad
            for tid in self.rig_meshes:
                obj = self.objects[tid]
                tl = proj.timelines[tid]
                vis = (not tl.hide) and (states[tid].inherit.get('VISIBLE', 1) or 0) > 0
                prev = last.get(tid)
                if prev is None or prev != vis:
                    obj.hide_viewport = not vis
                    obj.hide_render = not vis
                    obj.keyframe_insert('hide_viewport', frame=f)
                    obj.keyframe_insert('hide_render', frame=f)
                last[tid] = vis
        scene.frame_set(self.opt.frame if start <= self.opt.frame <= end else start)
        self.log("Animation baked: frames %d-%d" % (start, end))


def as_bool_num(v):
    if v is None:
        return False
    if isinstance(v, str):
        return v.lower() in ('true', '1')
    return bool(v)


def import_project(context, filepath, options):
    imp = Importer(filepath, options)
    imp.run(context)
    return imp
