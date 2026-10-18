# Importador de Mine-imator (.miproject, .miobject, .mimodel) para Blender.

bl_info = {
    "name": "Mine-imator Importer (.miproject/.miobject/.mimodel)",
    "author": "Alexandra https://x.com/_AlexandraM16",
    "version": (0, 18, 0),
    "blender": (3, 6, 0),
    "location": "Archivo > Importar > Mine-imator (.miproject/.miobject/.mimodel)",
    "description": "Importa proyectos, objetos y modelos de Mine-imator (1.1, 1.2, 2.0, Community y Vexel Build)",
    "category": "Import-Export",
}

EXTENSIONS = ('.miproject', '.miobject', '.mimodel')

if "bpy" in locals():
    import importlib
    from . import gm_math, mi_data, mi_assets, mi_geometry, mi_builder
    for _m in (gm_math, mi_data, mi_assets, mi_geometry, mi_builder):
        importlib.reload(_m)

import os

import bpy
from bpy.props import BoolProperty, CollectionProperty, EnumProperty, FloatProperty, IntProperty, StringProperty
from bpy_extras.io_utils import ImportHelper

from . import mi_data
from . import mi_builder


class MI_AddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    mineimator_dir: StringProperty(
        name="Carpeta de Mine-imator",
        description="Carpeta de instalación de Mine-imator (contiene Data/Minecraft). "
                    "Si se deja vacía se busca a partir de la ubicación del proyecto",
        subtype='DIR_PATH',
        default="",
    )

    def draw(self, context):
        self.layout.prop(self, "mineimator_dir")


def _prefs(context):
    try:
        return context.preferences.addons[__package__].preferences
    except (KeyError, AttributeError):
        return None


class IMPORT_OT_mineimator(bpy.types.Operator, ImportHelper):
    """Importa proyectos (.miproject), objetos (.miobject) y modelos (.mimodel) de Mine-imator"""
    bl_idname = "import_scene.mineimator"
    bl_label = "Importar Mine-imator"
    bl_options = {'REGISTER', 'UNDO', 'PRESET'}

    filename_ext = ".miproject"
    filter_glob: StringProperty(default="*.miproject;*.miobject;*.mimodel;*.backup*", options={'HIDDEN'})
    files: CollectionProperty(type=bpy.types.OperatorFileListElement, options={'HIDDEN', 'SKIP_SAVE'})
    directory: StringProperty(subtype='DIR_PATH', options={'HIDDEN', 'SKIP_SAVE'})

    mi_version: EnumProperty(
        name="Versión de Mine-imator",
        description="Versión de Mine-imator con la que se creó el proyecto",
        items=mi_data.VERSIONS,
        default='AUTO',
    )
    mineimator_dir: StringProperty(
        name="Carpeta de Mine-imator",
        description="Carpeta de instalación (para los modelos y texturas de Minecraft). Vacío = automático",
        subtype='DIR_PATH',
        default="",
    )
    frame: IntProperty(
        name="Fotograma",
        description="Fotograma del timeline cuya pose se importa",
        default=0, min=0,
    )
    animation: BoolProperty(
        name="Importar animación",
        description="Hornea los fotogramas clave de Mine-imator como animación de objetos",
        default=False,
    )
    anim_step: IntProperty(
        name="Paso",
        description="Cada cuántos fotogramas se crea un fotograma clave al hornear",
        default=1, min=1, max=50,
    )
    unit_scale: FloatProperty(
        name="Escala",
        description="Metros por unidad de Mine-imator (1/16 = un bloque mide 1 metro)",
        default=1.0 / 16.0, min=0.0001, max=100.0, precision=4,
    )
    import_hidden: BoolProperty(
        name="Importar objetos ocultos",
        description="Importa también los objetos ocultos o invisibles (quedan ocultos en Blender)",
        default=True,
    )
    pack_textures: BoolProperty(
        name="Empaquetar texturas",
        description="Guarda las texturas dentro del archivo .blend",
        default=True,
    )
    merge_vertices: BoolProperty(
        name="Unir vértices",
        description="Une los vértices coincidentes de cada pieza",
        default=True,
    )
    rig: BoolProperty(
        name="Crear esqueleto (armature)",
        description="Crea una armadura por modelo: un hueso por parte y otro por articulación con bend, "
                    "con la malla pesada para doblarse como en Mine-imator",
        default=True,
    )
    ik: BoolProperty(
        name="Restricciones IK",
        description="Resuelve el IK de Mine-imator al importar y añade restricciones IK de Blender hacia el mismo objetivo para poder moverlo",
        default=True,
    )
    bend_weights: BoolProperty(
        name="Doblado con pesos",
        description="Todas las partes con bend llevan dos huesos y pesos de influencia, también las de doblado "
                    "estático. Si se desactiva, el doblado estático se hornea con la geometría exacta (sin pesos)",
        default=True,
    )
    ik_keep_pose: BoolProperty(
        name="Conservar la pose",
        description="Ignora la escala de las extremidades con IK (el IK de Blender no puede estirar la cadena) "
                    "para que la pose importada no se desplace; si aun así el IK la altera, se silencia",
        default=True,
    )
    standard_view: BoolProperty(
        name="Vista de color 'Standard'",
        description="Usa la transformación de vista Standard para que los colores se vean como en Mine-imator",
        default=True,
    )
    depth_masks: BoolProperty(
        name="Emular máscaras de profundidad",
        description="Los objetos casi invisibles con 'depth' menor recortan (Boolean) a los que Mine-imator "
                    "dibuja después, como hacen las cuencas de ojos de muchos rigs",
        default=True,
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "mi_version")
        layout.prop(self, "mineimator_dir")
        layout.separator()
        layout.prop(self, "frame")
        layout.prop(self, "animation")
        row = layout.row()
        row.enabled = self.animation
        row.prop(self, "anim_step")
        layout.separator()
        layout.prop(self, "unit_scale")
        layout.prop(self, "import_hidden")
        layout.prop(self, "pack_textures")
        layout.prop(self, "rig")
        row = layout.row()
        row.enabled = self.rig
        row.prop(self, "bend_weights")
        row = layout.row()
        row.enabled = self.rig
        row.prop(self, "ik")
        row.prop(self, "ik_keep_pose")
        layout.prop(self, "merge_vertices")
        layout.prop(self, "depth_masks")
        layout.prop(self, "standard_view")

    def execute(self, context):
        mi_dir = self.mineimator_dir
        if not mi_dir:
            p = _prefs(context)
            mi_dir = p.mineimator_dir if p else ''
        opt = mi_builder.Options(
            version=self.mi_version, mi_dir=bpy.path.abspath(mi_dir) if mi_dir else '',
            frame=self.frame, animation=self.animation, anim_step=self.anim_step,
            unit_scale=self.unit_scale, import_hidden=self.import_hidden,
            pack_textures=self.pack_textures, merge_vertices=self.merge_vertices,
            depth_masks=self.depth_masks, rig=self.rig, ik=self.ik, ik_keep_pose=self.ik_keep_pose, bend_weights=self.bend_weights, standard_view=self.standard_view)
        paths = []
        if self.directory and self.files:
            paths = [os.path.join(self.directory, f.name) for f in self.files if f.name]
        if not paths and self.filepath:
            paths = [self.filepath]
        paths = [p for p in paths if p.lower().endswith(EXTENSIONS) or '.backup' in p.lower()]
        if not paths:
            self.report({'ERROR'}, "Selecciona un archivo .miproject, .miobject o .mimodel")
            return {'CANCELLED'}

        done = 0
        for path in paths:
            try:
                imp = mi_builder.import_project(context, path, opt)
            except (OSError, ValueError) as e:
                self.report({'ERROR'}, "%s: %s" % (os.path.basename(path), e))
                continue
            done += 1
            for m in imp.messages:
                if m.startswith(('Aviso', 'Error', 'Textura no', 'Recurso no', 'Modelo de', 'No se', 'Bloque no')):
                    self.report({'WARNING'}, m)
            self.report({'INFO'}, "%s: %s" % (os.path.basename(path), imp.messages[-1] if imp.messages else "importado"))
        return {'FINISHED'} if done else {'CANCELLED'}

    def invoke(self, context, event):
        # Al arrastrar un archivo al visor ya llega la ruta: mostrar solo las opciones
        if self.directory and self.files:
            return context.window_manager.invoke_props_dialog(self, width=360)
        return ImportHelper.invoke(self, context, event)


def menu_func_import(self, context):
    self.layout.operator(IMPORT_OT_mineimator.bl_idname, text="Mine-imator (.miproject/.miobject/.mimodel)")


classes = [MI_AddonPreferences, IMPORT_OT_mineimator]

# Arrastrar y soltar archivos en el visor (Blender 4.1+)
if hasattr(bpy.types, 'FileHandler'):
    class MI_FH_import(bpy.types.FileHandler):
        bl_idname = "MI_FH_import"
        bl_label = "Mine-imator"
        bl_import_operator = "import_scene.mineimator"
        bl_file_extensions = ";".join(EXTENSIONS)

        @classmethod
        def poll_drop(cls, context):
            return context.area is not None and context.area.type in ('VIEW_3D', 'OUTLINER')

    classes.append(MI_FH_import)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
