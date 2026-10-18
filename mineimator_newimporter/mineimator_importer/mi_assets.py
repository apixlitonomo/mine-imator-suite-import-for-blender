# Acceso a los recursos de Minecraft que trae Mine-imator (Data/Minecraft/*.zip
# y *.midata) y a los archivos del proyecto.

import glob
import json
import os
import re
import zipfile


def _version_key(path):
    name = os.path.splitext(os.path.basename(path))[0]
    return [int(n) for n in re.findall(r'\d+', name)] or [0]


def find_mineimator_dir(start_path, preferred='', version='AUTO'):
    """Busca la carpeta de instalación de Mine-imator (la que contiene Data/Minecraft)."""
    candidates = []
    if preferred:
        candidates.append(preferred)
    # Carpetas que contienen el proyecto y el propio conversor
    for base in (start_path, __file__):
        p = os.path.abspath(base)
        for _ in range(6):
            p = os.path.dirname(p)
            if not p:
                break
            candidates.append(p)
    # Ubicaciones habituales de instalacion (la version mas reciente primero)
    roots = [os.environ.get('ProgramFiles', ''), os.environ.get('ProgramFiles(x86)', ''),
             os.environ.get('LOCALAPPDATA', ''), os.path.expanduser('~')]
    roots += ['%s:\\' % d for d in 'CDEFGH']
    found = []
    for r in roots:
        if r and os.path.isdir(r):
            found += glob.glob(os.path.join(r, 'Mine-imator*')) + glob.glob(os.path.join(r, 'Mine imator*'))
    keyword = {'VEXEL': 'vexel', 'CB': 'community', 'MI_200': '2.0', 'MI_120': '1.2', 'MI_110': '1.1'}.get(version)
    found.sort(key=os.path.getmtime, reverse=True)
    if keyword:
        found.sort(key=lambda f: keyword not in os.path.basename(f).lower())
    candidates += found
    for c in candidates:
        if c and os.path.isdir(os.path.join(c, 'Data', 'Minecraft')):
            return c
    return None


class MinecraftAssets:
    """Lector del zip de recursos de Minecraft + archivo .midata de Mine-imator."""

    def __init__(self, mi_dir):
        self.ok = False
        self.zip = None
        self.midata = {}
        self.version = ''
        self.names = set()
        self.by_basename = {}
        self._json_cache = {}
        if not mi_dir:
            return
        mc = os.path.join(mi_dir, 'Data', 'Minecraft')
        zips = sorted(glob.glob(os.path.join(mc, '*.zip')), key=_version_key)
        if not zips:
            return
        zpath = zips[-1]
        self.version = os.path.splitext(os.path.basename(zpath))[0]
        self.zip = zipfile.ZipFile(zpath, 'r')
        for n in self.zip.namelist():
            self.names.add(n)
            self.by_basename.setdefault(os.path.basename(n).lower(), []).append(n)
        mid = os.path.join(mc, self.version + '.midata')
        if not os.path.isfile(mid):
            mids = sorted(glob.glob(os.path.join(mc, '*.midata')), key=_version_key)
            mid = mids[-1] if mids else None
        if mid:
            try:
                with open(mid, 'r', encoding='utf-8-sig') as f:
                    self.midata = json.load(f)
            except (OSError, ValueError):
                self.midata = {}
        self.ok = True

    def close(self):
        if self.zip:
            self.zip.close()
            self.zip = None

    # -- lectura -------------------------------------------------------------

    def read(self, name):
        if not self.zip or name not in self.names:
            return None
        return self.zip.read(name)

    def read_json(self, name):
        if name in self._json_cache:
            return self._json_cache[name]
        data = self.read(name)
        val = None
        if data is not None:
            try:
                val = json.loads(data.decode('utf-8-sig'))
            except ValueError:
                val = None
        self._json_cache[name] = val
        return val

    def texture_path(self, texname):
        """'entity/player/wide/steve' -> ruta dentro del zip."""
        t = texname.replace('minecraft:', '')
        if t.endswith('.png'):
            t = t[:-4]
        p = 'assets/minecraft/textures/%s.png' % t
        return p if p in self.names else None

    def find_model_file(self, filename, kind=None):
        cands = self.by_basename.get(os.path.basename(filename).lower(), [])
        if kind:
            for c in cands:
                if '/%s/' % kind in c:
                    return c
        return cands[0] if cands else None

    def extract(self, name, cache_dir):
        """Extrae un archivo del zip a la carpeta de caché y devuelve su ruta."""
        out = os.path.join(cache_dir, self.version, *name.split('/'))
        if not os.path.isfile(out):
            data = self.read(name)
            if data is None:
                return None
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, 'wb') as f:
                f.write(data)
        return out

    # -- midata --------------------------------------------------------------

    def _find_entry(self, listname, name):
        for e in self.midata.get(listname) or []:
            if isinstance(e, dict) and e.get('name') == name:
                return e
        return None

    def resolve_state_entry(self, entry, state):
        """Combina default_state + estado y devuelve archivo, texturas y ocultaciones."""
        from .mi_data import parse_state
        st = parse_state(entry.get('default_state'))
        st.update(state or {})
        res = {
            'file': entry.get('file'),
            'texture': entry.get('texture'),
            'shape_texture': {},
            'shape_hide': [],
            'shape_color': {},
            'hide': [],
            'state': st,
        }
        states = entry.get('states')
        if isinstance(states, dict):
            for sname, opts in states.items():
                if not isinstance(opts, list):
                    continue
                val = st.get(sname)
                chosen = None
                for o in opts:
                    if isinstance(o, dict) and str(o.get('value')) == str(val):
                        chosen = o
                        break
                if chosen is None:
                    continue
                if chosen.get('file'):
                    res['file'] = chosen['file']
                if chosen.get('texture'):
                    res['texture'] = chosen['texture']
                stx = chosen.get('shape_texture')
                if isinstance(stx, dict):
                    res['shape_texture'].update(stx)
                sh = chosen.get('shape_hide')
                if isinstance(sh, list):
                    res['shape_hide'].extend(sh)
                elif isinstance(sh, str):
                    res['shape_hide'].append(sh)
                sc = chosen.get('shape_color')
                if isinstance(sc, dict):
                    res['shape_color'].update(sc)
                hd = chosen.get('hide')
                if isinstance(hd, list):
                    res['hide'].extend(hd)
                elif isinstance(hd, str):
                    res['hide'].append(hd)
        return res

    def resolve_character(self, name, state, special=False):
        entry = None
        lists = ('special_blocks', 'characters') if special else ('characters', 'special_blocks')
        for ln in lists:
            entry = self._find_entry(ln, name)
            if entry:
                break
        if entry is None:
            return None
        info = self.resolve_state_entry(entry, state)
        if not info['file']:
            return None
        kind = 'special_block' if special else 'character'
        path = self.find_model_file(info['file'], kind)
        if not path:
            return None
        info['path'] = path
        return info

    def resolve_block(self, name, state):
        """Devuelve (ruta_blockstate, estado) de un bloque de Mine-imator."""
        entry = self._find_entry('blocks', name)
        st = dict(state or {})
        fname = None
        if entry:
            info = self.resolve_state_entry(entry, state)
            st = info['state']
            fname = info['file']
        if not fname:
            fname = name + '.json'
        path = 'assets/minecraft/blockstates/' + fname
        if path not in self.names:
            cand = self.by_basename.get(fname.lower(), [])
            cand = [c for c in cand if '/blockstates/' in c]
            path = cand[0] if cand else None
        return path, st

    def item_texture_name(self, slot):
        lst = self.midata.get('item_textures') or []
        if 0 <= slot < len(lst):
            return lst[slot]
        return None

    def item_slot(self, name):
        lst = self.midata.get('item_textures') or []
        try:
            return lst.index(name)
        except ValueError:
            return 0


def find_project_file(project_folder, filename):
    """Busca un archivo de recurso del proyecto (ruta relativa o por nombre)."""
    if not filename:
        return None
    p = filename if os.path.isabs(filename) else os.path.join(project_folder, filename)
    if os.path.isfile(p):
        return p
    base = os.path.basename(filename).lower()
    for root, _dirs, files in os.walk(project_folder):
        for f in files:
            if f.lower() == base:
                return os.path.join(root, f)
    return None
