"""Suite-side import orchestration: Mine-imator Suite extras around the Mine-imator Importer core.

The geometry, rigs, IK, bends and optional animation come from ``mi_builder`` (the "new importer").
This module keeps what the Suite adds on top of it: the import collection tree and category filters,
cameras, lights, paths, particle placeholders, world scenery (Mineways / ``.meshcache``), the full
Mine-imator environment, MCprep material preparation, the startup-cube cleanup and the report.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json
import math
import os
import subprocess
import tempfile

import bpy
from mathutils import Matrix, Vector

from . import core, meshcache, mi_assets, mi_builder, mi_data

IMPORT_EXTENSIONS = (".miproject", ".miobject", ".mimodel")

CATEGORY_NAMES = {
    "characters": "Characters & Entities",
    "items": "Items",
    "blocks": "Blocks & Special Blocks",
    "models": "Custom Models",
    "scenery": "World Scenery",
    "primitives": "Shapes & Text",
    "lights": "Lights",
    "camera": "Camera",
    "environment": "Environment",
    "helpers": "Paths & Helpers",
}


@dataclass
class ImportOptions:
    # --- Suite ---------------------------------------------------------------------------
    categories: set[str] = field(default_factory=lambda: set(CATEGORY_NAMES))
    mineimator_path: str = ""
    asset_pack_path: str = ""
    mineways_path: str = ""
    use_mcprep: bool = True
    remove_startup_cube: bool = True
    mineimator_suite: bool = True
    outer_layers_3d: bool = False
    # --- Importer core (frame / animation / rigging) -----------------------------------------
    version: str = "AUTO"
    frame: int = 0
    animation: bool = False
    anim_step: int = 1
    unit_scale: float = 1.0 / 16.0
    import_hidden: bool = True
    pack_textures: bool = True
    merge_vertices: bool = True
    depth_masks: bool = True
    rig: bool = True
    ik: bool = True
    ik_keep_pose: bool = True
    bend_weights: bool = True
    standard_view: bool = True


@dataclass
class ImportReport:
    project_path: str = ""
    collection_name: str = ""
    animation_imported: bool = False
    imported: Counter = field(default_factory=Counter)
    created: Counter = field(default_factory=Counter)
    skipped: Counter = field(default_factory=Counter)
    fallbacks: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    placeholders: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def text(self) -> str:
        lines = [
            "Mine-imator Suite + Import report",
            f"Bridge version: {core.ADDON_VERSION}",
            f"Source: {self.project_path}",
            f"Collection: {self.collection_name or '(preflight)'}",
            "Animation: baked from the project's keyframes" if self.animation_imported
            else "Animation: not imported (single pose)",
            "",
            "Accounted root timelines:",
        ]
        if self.imported:
            lines.extend(f"  {key}: {value}" for key, value in sorted(self.imported.items()))
        else:
            lines.append("  none")
        if self.created:
            lines += ["", "Created datablocks:"]
            lines.extend(f"  {key}: {value}" for key, value in sorted(self.created.items()))
        for heading, values in (
            ("Fallbacks", self.fallbacks),
            ("Missing resources", self.missing),
            ("Placeholders", self.placeholders),
            ("Notes", self.notes),
        ):
            lines += ["", f"{heading} ({len(values)}):"]
            lines.extend(f"  - {value}" for value in values) if values else lines.append("  none")
        return "\n".join(lines)


def resolve_import_path(value: str | os.PathLike[str]) -> Path:
    """A .miproject/.miobject/.mimodel file, or a directory containing exactly one .miproject."""
    path = Path(value).expanduser()
    if path.is_file():
        if path.suffix.lower() not in IMPORT_EXTENSIONS and ".backup" not in path.name.lower():
            raise core.BridgeError(f"Not a Mine-imator project, object or model: {path}")
        return path.resolve()
    return core.resolve_project_path(path)



def _use_hashed_alpha(material: bpy.types.Material) -> None:
    """Mine-imator's alpha hashing: a fragment with 0 < alpha < 1 is discarded when
    alpha < hash(position, sample), otherwise it becomes fully opaque and writes
    depth.  Blender's Dithered (EEVEE Next) / Hashed (older EEVEE) is the same
    technique, so overlapping translucent shapes never need sorting."""
    if hasattr(material, "surface_render_method"):
        material.surface_render_method = "DITHERED"
    try:
        material.blend_method = "HASHED"
        material.shadow_method = "HASHED"
    except Exception:
        pass

def _hex_color(value: Any, alpha: float = 1.0) -> tuple[float, float, float, float]:
    raw = str(value or "#FFFFFF").lstrip("#")
    if len(raw) == 8:
        alpha = int(raw[6:8], 16) / 255.0
        raw = raw[:6]
    try:
        return tuple(int(raw[index:index + 2], 16) / 255.0 for index in (0, 2, 4)) + (alpha,)
    except (ValueError, IndexError):
        return (1.0, 1.0, 1.0, alpha)

def _remove_untouched_startup_cube(context: bpy.types.Context, report: ImportReport) -> bool:
    """Remove only Blender's pristine default cube, never a user-authored cube."""
    cube = bpy.data.objects.get("Cube")
    if not cube or cube.type != "MESH" or cube.parent or cube.animation_data:
        return False
    if tuple(cube.keys()) or cube.modifiers or cube.vertex_groups:
        return False
    if len(cube.material_slots) > 1:
        return False
    startup_material = cube.material_slots[0].material if cube.material_slots else None
    if startup_material and (
        startup_material.name != "Material"
        or any(abs(value - expected) > 1e-6 for value, expected in zip(startup_material.diffuse_color, (0.8, 0.8, 0.8, 1.0)))
    ):
        return False
    if len(cube.users_collection) != 1:
        return False
    collection = cube.users_collection[0]
    if collection.name != "Collection" or collection not in context.scene.collection.children[:]:
        return False
    siblings = {(obj.name, obj.type) for obj in collection.objects}
    if ("Camera", "CAMERA") not in siblings or ("Light", "LIGHT") not in siblings:
        return False
    if any(abs(value) > 1e-6 for value in cube.location):
        return False
    if any(abs(value) > 1e-6 for value in cube.rotation_euler):
        return False
    if any(abs(value - 1.0) > 1e-6 for value in cube.scale):
        return False
    mesh = cube.data
    if mesh.name != "Cube" or (len(mesh.vertices), len(mesh.edges), len(mesh.polygons)) != (8, 12, 6):
        return False
    expected = {
        (-1.0, -1.0, -1.0), (-1.0, -1.0, 1.0), (-1.0, 1.0, -1.0), (-1.0, 1.0, 1.0),
        (1.0, -1.0, -1.0), (1.0, -1.0, 1.0), (1.0, 1.0, -1.0), (1.0, 1.0, 1.0),
    }
    coordinates = {tuple(round(axis, 6) for axis in vertex.co) for vertex in mesh.vertices}
    if coordinates != expected:
        return False
    bpy.data.objects.remove(cube, do_unlink=True)
    if mesh.users == 0:
        bpy.data.meshes.remove(mesh)
    if startup_material and startup_material.users == 0:
        bpy.data.materials.remove(startup_material)
    report.notes.append("Removed Blender's untouched startup cube")
    report.created["startup_cube_removed"] += 1
    return True

def _link_object(obj: bpy.types.Object, collection: bpy.types.Collection) -> None:
    for current in list(obj.users_collection):
        current.objects.unlink(obj)
    collection.objects.link(obj)


def _new_empty(name: str, collection: bpy.types.Collection, display: str = "PLAIN_AXES") -> bpy.types.Object:
    obj = bpy.data.objects.new(name, None)
    obj.empty_display_type = display
    obj.empty_display_size = 0.25
    collection.objects.link(obj)
    return obj

def _new_material(name: str, image_path: Path | None = None, color: tuple[float, float, float, float] = (1, 1, 1, 1)) -> bpy.types.Material:
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    material.diffuse_color = color
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    principled = nodes.get("Principled BSDF")
    if principled:
        principled.inputs["Base Color"].default_value = color
        principled.inputs["Alpha"].default_value = color[3]
        principled.inputs["Roughness"].default_value = 0.8
    if image_path and image_path.is_file() and principled:
        try:
            image = bpy.data.images.load(str(image_path), check_existing=True)
            image.colorspace_settings.name = "sRGB"
            texture = nodes.new("ShaderNodeTexImage")
            texture.image = image
            texture.interpolation = "Closest"
            links.new(texture.outputs["Color"], principled.inputs["Base Color"])
            if color[3] < 1.0:
                # color_alpha / ALPHA scale the texture's own alpha; linking the
                # texture alone silently threw the shape's alpha away.
                scale = nodes.new("ShaderNodeMath")
                scale.operation = "MULTIPLY"
                scale.inputs[1].default_value = color[3]
                links.new(texture.outputs["Alpha"], scale.inputs[0])
                links.new(scale.outputs[0], principled.inputs["Alpha"])
            else:
                links.new(texture.outputs["Alpha"], principled.inputs["Alpha"])
            material.diffuse_color = (1, 1, 1, color[3])
        except Exception:
            pass
    _use_hashed_alpha(material)
    if hasattr(material, "use_transparency_overlap"):
        material.use_transparency_overlap = False
    material["mi_pixel_art"] = True
    return material

def _meshcache_atlas(assets: core.AssetStore, *, animated: bool) -> bpy.types.Image:
    """Build Mine-imator's exact 32-column block sheet inside Blender.

    Meshcache UVs point into this generated sheet rather than individual block
    textures. Mine-imator's metadata supplies the ordering, so the result also
    works when a future asset pack changes that order.
    """
    import numpy as np

    role = "animated" if animated else "static"
    image_name = f"MI {assets.asset_version} Block Atlas ({role})"
    existing = bpy.data.images.get(image_name)
    if existing and existing.get("mi_asset_zip") == str(assets.zip_path):
        return existing
    atlas_path = assets.extract_root / "bridge_generated" / f"block_atlas_{role}.png"
    if atlas_path.is_file():
        atlas = bpy.data.images.load(str(atlas_path), check_existing=True)
        atlas.name = image_name
        atlas.colorspace_settings.name = "sRGB"
        atlas.alpha_mode = "STRAIGHT"
        atlas["mi_bridge_version"] = core.ADDON_VERSION
        atlas["mi_asset_zip"] = str(assets.zip_path)
        atlas["mi_source_setting"] = f"block_texture_atlas_{role}"
        return atlas

    names = assets.metadata.get("block_textures_animated" if animated else "block_textures", [])
    if not isinstance(names, list):
        raise RuntimeError(f"Mine-imator asset metadata has no {role} block texture list")
    sheet_width = 32
    sheet_height = 2 if animated else 32
    tile_size = 16
    pixels = np.zeros((sheet_height * tile_size, sheet_width * tile_size, 4), dtype=np.float32)
    for slot, raw_name in enumerate(names[:sheet_width * sheet_height]):
        clean_name = str(raw_name).replace(" opaque", "").replace(" noalpha", "")
        path = assets.materialize(f"textures/{clean_name}.png")
        if not path:
            continue
        source = bpy.data.images.load(str(path), check_existing=False)
        try:
            width, height = source.size
            if width != tile_size or height < tile_size:
                continue
            # Blender defers decoding file-backed image pixels. Touching one
            # value ensures foreach_get sees decoded data instead of zeros.
            _ = source.pixels[0]
            values = np.empty(width * height * 4, dtype=np.float32)
            source.pixels.foreach_get(values)
            source_pixels = values.reshape((height, width, 4))
            # Minecraft animated strips put frame zero at the top of the PNG.
            tile = source_pixels[height - tile_size:height, :tile_size, :]
            x = (slot % sheet_width) * tile_size
            y = (slot // sheet_width) * tile_size
            pixels[y:y + tile_size, x:x + tile_size, :] = tile
        finally:
            bpy.data.images.remove(source)

    atlas = bpy.data.images.new(f"{image_name} (building)", width=sheet_width * tile_size, height=sheet_height * tile_size, alpha=True)
    atlas.colorspace_settings.name = "sRGB"
    atlas.alpha_mode = "STRAIGHT"
    # Image settings trigger an internal reset in Blender 5.2, so write pixels
    # only after all settings and metadata are assigned.
    atlas.pixels[:] = pixels.ravel()
    atlas_path.parent.mkdir(parents=True, exist_ok=True)
    atlas.filepath_raw = str(atlas_path)
    atlas.file_format = "PNG"
    atlas.save()
    bpy.data.images.remove(atlas)
    atlas = bpy.data.images.load(str(atlas_path), check_existing=True)
    atlas.name = image_name
    atlas.colorspace_settings.name = "sRGB"
    atlas.alpha_mode = "STRAIGHT"
    atlas["mi_bridge_version"] = core.ADDON_VERSION
    atlas["mi_asset_zip"] = str(assets.zip_path)
    atlas["mi_source_setting"] = f"block_texture_atlas_{role}"
    return atlas


_MESHCACHE_TINT_DEFAULTS = {
    "normal": (1.0, 1.0, 1.0, 1.0),
    "animated": (1.0, 1.0, 1.0, 1.0),
    "grass": (0.5686, 0.7412, 0.3490, 1.0),
    "foliage": (0.4667, 0.6706, 0.1843, 1.0),
    "leaves_oak": (0.4667, 0.6706, 0.1843, 1.0),
    "leaves_spruce": (0.3843, 0.6588, 0.3412, 1.0),
    "leaves_birch": (0.3843, 0.6588, 0.3412, 1.0),
    "leaves_jungle": (0.4667, 0.6706, 0.1843, 1.0),
    "leaves_acacia": (0.4667, 0.6706, 0.1843, 1.0),
    "leaves_dark_oak": (0.4667, 0.6706, 0.1843, 1.0),
    "leaves_mangrove": (0.4667, 0.6706, 0.1843, 1.0),
    "water": (0.2431, 0.4588, 0.8824, 0.72),
}


def _meshcache_material(name: str, image: bpy.types.Image, depth: int, role: str) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    material.diffuse_color = _MESHCACHE_TINT_DEFAULTS[role]
    material["mi_pixel_art"] = True
    material["mi_meshcache_tint_role"] = role
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    principled.name = "MI Meshcache Shader"
    principled.inputs["Roughness"].default_value = 0.9 if role != "water" else 0.12
    texture = nodes.new("ShaderNodeTexImage")
    texture.name = "MI Block Atlas"
    texture.image = image
    texture.interpolation = "Closest"
    texture.extension = "CLIP"
    tint = nodes.new("ShaderNodeRGB")
    tint.name = "MI Tint"
    tint.outputs[0].default_value = _MESHCACHE_TINT_DEFAULTS[role]
    multiply = nodes.new("ShaderNodeMixRGB")
    multiply.name = "MI Texture Tint"
    multiply.blend_type = "MULTIPLY"
    multiply.inputs[0].default_value = 1.0
    links.new(texture.outputs["Color"], multiply.inputs[1])
    links.new(tint.outputs[0], multiply.inputs[2])
    links.new(multiply.outputs[0], principled.inputs["Base Color"])
    if depth > 0:
        if role == "water":
            alpha = nodes.new("ShaderNodeMath")
            alpha.operation = "MULTIPLY"
            alpha.inputs[1].default_value = _MESHCACHE_TINT_DEFAULTS[role][3]
            links.new(texture.outputs["Alpha"], alpha.inputs[0])
            links.new(alpha.outputs[0], principled.inputs["Alpha"])
            principled.inputs["Metallic"].default_value = 0.0
            principled.inputs["IOR"].default_value = 1.333
        else:
            links.new(texture.outputs["Alpha"], principled.inputs["Alpha"])
        if hasattr(material, "surface_render_method"):
            material.surface_render_method = "DITHERED"
        if hasattr(material, "use_transparency_overlap"):
            material.use_transparency_overlap = False
    links.new(principled.outputs[0], output.inputs["Surface"])
    return material


def _meshcache_object(raw: meshcache.RawMesh, name: str, material: bpy.types.Material, collection: bpy.types.Collection) -> bpy.types.Object:
    """Convert one native Mine-imator VertexBuffer mesh with bulk Blender APIs."""
    import numpy as np

    vertex_dtype = np.dtype([
        ("co", "<f4", (3,)),
        ("normal", "<u4"),
        ("color", "<u4"),
        ("uv", "<f4", (2,)),
        ("data", "<u4"),
        ("tangent", "<u4"),
    ])
    vertices = np.frombuffer(raw.vertices, dtype=vertex_dtype, count=raw.vertex_count)
    indices = np.frombuffer(raw.indices, dtype="<u4", count=raw.index_count)
    coordinates = np.empty((raw.vertex_count, 3), dtype=np.float32)
    coordinates[:, 0] = vertices["co"][:, 0] / core.MI_UNITS_PER_BLOCK
    coordinates[:, 1] = -vertices["co"][:, 1] / core.MI_UNITS_PER_BLOCK
    coordinates[:, 2] = vertices["co"][:, 2] / core.MI_UNITS_PER_BLOCK

    mesh = bpy.data.meshes.new(f"{name} Mesh")
    mesh.vertices.add(raw.vertex_count)
    mesh.vertices.foreach_set("co", coordinates.ravel())
    mesh.loops.add(raw.index_count)
    mesh.loops.foreach_set("vertex_index", indices)
    polygon_count = raw.index_count // 3
    mesh.polygons.add(polygon_count)
    mesh.polygons.foreach_set("loop_start", np.arange(0, raw.index_count, 3, dtype=np.int32))
    mesh.polygons.foreach_set("loop_total", np.full(polygon_count, 3, dtype=np.int32))
    uv_layer = mesh.uv_layers.new(name="UVMap")
    loop_uv = np.ascontiguousarray(vertices["uv"][indices], dtype=np.float32)
    uv_layer.data.foreach_set("uv", loop_uv.ravel())
    mesh.materials.append(material)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    obj["mi_meshcache_depth"] = raw.depth
    obj["mi_meshcache_buffer"] = raw.buffer_name
    return obj

def _mcprep_texture(name: str) -> Path | None:
    """Last-resort lookup in MCprep's bundled default resource pack."""
    raw = str(name).replace("minecraft:", "").replace("\\", "/")
    raw = raw.removeprefix("textures/")
    if not Path(raw).suffix:
        raw += ".png"
    candidates: list[Path] = []
    scripts = Path(bpy.utils.user_resource("SCRIPTS"))
    addon_roots = [scripts / "addons" / "MCprep_addon", scripts / "addons_core" / "MCprep_addon"]
    for addon_root in addon_roots:
        pack = addon_root / "MCprep_resources" / "resourcepacks" / "mcprep_default" / "assets" / "minecraft" / "textures"
        candidates.extend([pack / raw, pack / Path(raw).name])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None



def _tag(obj: bpy.types.Object, project_path: Path | str, tl: Any = None, extra: Path | str | None = None) -> None:
    """Suite metadata on an object (kept from the original bridge so existing tooling still finds it)."""
    obj["mi_bridge_version"] = core.ADDON_VERSION
    obj["mi_source_project"] = str(project_path)
    if tl is not None:
        obj["mi_id"] = str(tl.id)
        obj["mi_type"] = str(tl.type)
        obj["mi_template"] = str(tl.temp or "null")
    if extra:
        obj["mi_resource_path"] = str(extra)


def _import_meshcache_world(
    cache_path: Path,
    name: str,
    root: bpy.types.Object,
    tl: Any,
    template: Any,
    collection: bpy.types.Collection,
    project_path: Path,
    assets: core.AssetStore,
    report: ImportReport,
) -> bpy.types.Object:
    """Recover world scenery from a Mine-imator format-2 ``.meshcache`` under an existing root empty."""
    header = meshcache.read_header(cache_path)
    _tag(root, project_path, tl, cache_path)
    root["mi_meshcache_format"] = header.format
    root["mi_meshcache_size"] = list(header.size)

    static_atlas = _meshcache_atlas(assets, animated=False)
    animated_atlas = _meshcache_atlas(assets, animated=True)
    # Mine-imator first subtracts the template's automatic scenery pivot (half of the crop in X/Y), then
    # renders the cached vertices through render_world_block's legacy basis. In engine space that basis is
    # (x, y, z) -> (y, -x + sizeY, z). Reflecting engine Y into Blender Y produces the matrices below. Both
    # stages are required: omitting the pivot rotates the crop around a corner and leaves its terrain
    # hundreds of blocks away from the camera.
    legacy_basis = (
        Matrix.Translation((0.0, -float(header.size[1]), 0.0))
        @ Matrix.Rotation(math.radians(90.0), 4, "Z")
    )
    automatic_pivot = Matrix.Translation(
        (-float(header.size[0]) / 2.0, float(header.size[1]) / 2.0, 0.0)
    )
    scenery_basis = automatic_pivot @ legacy_basis
    materials: dict[tuple[int, str], bpy.types.Material] = {}
    imported_vertices = imported_triangles = 0
    for raw in meshcache.iter_raw_meshes(cache_path):
        if not raw.vertex_count or not raw.index_count:
            continue
        role = raw.buffer_name
        key = (raw.depth, role)
        if key not in materials:
            atlas = animated_atlas if role in {"animated", "water"} else static_atlas
            materials[key] = _meshcache_material(
                f"MI World {role.replace('_', ' ').title()} D{raw.depth}", atlas, raw.depth, role
            )
            materials[key]["mi_source_project"] = str(project_path)
            materials[key]["mi_bridge_version"] = core.ADDON_VERSION
            materials[key]["mi_suite_id"] = report.collection_name
        object_name = f"{name} - {role.replace('_', ' ').title()} {raw.mesh_index + 1}"
        obj = _meshcache_object(raw, object_name, materials[key], collection)
        obj.parent = root
        obj.matrix_basis = scenery_basis
        _tag(obj, project_path, tl, cache_path)
        imported_vertices += raw.vertex_count
        imported_triangles += raw.index_count // 3
        report.created["scenery_mesh"] += 1

    report.fallbacks.append(f"World scenery recovered from Mine-imator meshcache: {cache_path}")
    report.notes.append(
        f"Meshcache world: {header.size[0]} x {header.size[1]} x {header.size[2]} blocks, "
        f"{imported_vertices:,} vertices, {imported_triangles:,} triangles"
    )
    return root


# Mine-imator cameras look along their local +Y (engine space), which is -Y after the engine->Blender
# reflection, with +Z up. Blender cameras look along -Z with +Y up, so this fixed rotation (applied on
# the right of the camera's world matrix) re-aims the Blender camera. It reproduces the Suite's original
# yaw/pitch aim exactly and additionally honours roll and parenting.
CAMERA_FIX = Matrix((
    (-1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
))


class SuiteExtras:
    """Hooks called by ``mi_builder.Importer`` for the timeline types the importer core does not build."""

    LIGHTS = ("pointlight", "spotlight")
    TYPES = ("camera", "pointlight", "spotlight", "scenery", "path", "pathpoint", "particles")

    def __init__(self, owner: "SceneImporter"):
        self.owner = owner
        self._raw: dict[str, dict[str, Any]] | None = None
        self._paths: list[tuple[Any, bpy.types.Object]] = []

    # -- hooks ---------------------------------------------------------------------------

    def handles(self, timeline_type: str) -> bool:
        return timeline_type in self.TYPES

    def make_data(self, imp: mi_builder.Importer, tl: Any, st: Any, name: str) -> Any:
        if tl.type in self.LIGHTS:
            return self._light_data(imp, tl, st, name)
        if tl.type == "camera":
            return self._camera_data(st, name)
        return None   # scenery, paths and particles are empties that finish() fills in

    def fix_for(self, imp: mi_builder.Importer, tl: Any) -> tuple[Matrix | None, bool]:
        if tl.type == "camera":
            return CAMERA_FIX, True
        if tl.type in self.LIGHTS:
            return None, True
        return None, False

    def finish(self, imp: mi_builder.Importer, tl: Any, st: Any, obj: bpy.types.Object) -> None:
        report = self.owner.report
        if tl.type == "camera":
            self.owner.context.scene.camera = obj
            report.created["camera"] += 1
        elif tl.type in self.LIGHTS:
            report.created["light"] += 1
        elif tl.type == "path":
            self._paths.append((tl, obj))
        elif tl.type == "pathpoint":
            obj.empty_display_type = "SPHERE"
        elif tl.type == "particles":
            obj.empty_display_type = "SPHERE"
            obj.name = f"Particle Spawner - {obj.name}"
            obj["mi_placeholder_reason"] = "particle simulation is not converted"
            report.placeholders.append(f"{obj.name}: particle simulation is not converted")
            report.created["placeholder"] += 1
        elif tl.type == "scenery":
            self._scenery(imp, tl, obj)

    def finalize(self, imp: mi_builder.Importer) -> None:
        for tl, root in self._paths:
            self._build_path(imp, tl, root)
        for tid, obj in imp.objects.items():
            tl = imp.project.timelines.get(tid)
            if tl is not None:
                _tag(obj, imp.project.path, tl)

    # -- helpers -------------------------------------------------------------------------

    def _raw_timeline(self, imp: mi_builder.Importer, tl: Any) -> dict[str, Any]:
        if self._raw is None:
            self._raw = {}
            for item in imp.project.root.get("timelines") or []:
                if isinstance(item, dict):
                    self._raw[mi_data.save_id(item.get("id")) or ""] = item
        return self._raw.get(tl.id, {})

    def _collection(self, category: str) -> bpy.types.Collection:
        return self.owner.collections.get(category) or self.owner.root_collection

    def _light_data(self, imp: mi_builder.Importer, tl: Any, st: Any, name: str) -> bpy.types.Light:
        values = st.values
        is_spot = tl.type == "spotlight"
        data = bpy.data.lights.new(name or ("Spot Light" if is_spot else "Point Light"), "SPOT" if is_spot else "POINT")
        color = values.get("LIGHT_COLOR") or (1.0, 1.0, 1.0)
        data.color = tuple(color)[:3]
        strength = max(0.0, float(values.get("LIGHT_STRENGTH", 1.0) or 0.0))
        light_range = max(0.01, float(values.get("LIGHT_RANGE", 250.0) or 250.0) / core.MI_UNITS_PER_BLOCK)
        data.energy = strength * max(25.0, light_range * light_range * 10.0)
        data.cutoff_distance = light_range
        data.use_shadow = bool(self._raw_timeline(imp, tl).get("shadows", True))
        data.specular_factor = float(values.get("LIGHT_SPECULAR_STRENGTH", 1.0) or 0.0)
        data.shadow_soft_size = max(0.0, float(values.get("LIGHT_FADE_SIZE", 0.0) or 0.0) / core.MI_UNITS_PER_BLOCK)
        if is_spot:
            data.spot_size = min(math.pi, math.radians(float(values.get("LIGHT_SPOT_RADIUS", 50.0) or 50.0) * 2.0))
            sharpness = values.get("LIGHT_SPOT_SHARPNESS")
            data.spot_blend = 0.15 if sharpness is None else min(1.0, max(0.0, 1.0 - float(sharpness)))
        return data

    def _camera_data(self, st: Any, name: str) -> bpy.types.Camera:
        values = st.values
        data = bpy.data.cameras.new(name or "Mine-imator Camera")
        fov = math.radians(float(values.get("CAM_FOV", 45.0) or 45.0))
        # Mine-imator stores a vertical FOV; Blender's generic `angle` setter is horizontal under the
        # default sensor fit.
        data.sensor_fit = "VERTICAL"
        data.lens = data.sensor_height / (2.0 * math.tan(max(0.001, min(fov, math.radians(170.0))) / 2.0))
        if values.get("CAM_DOF"):
            data.dof.use_dof = True
            if values.get("CAM_DOF_DEPTH") is not None:
                data.dof.focus_distance = max(0.01, float(values["CAM_DOF_DEPTH"]) / core.MI_UNITS_PER_BLOCK)
        return data

    def _descendants(self, tl: Any) -> list[Any]:
        result: list[Any] = []
        pending = list(tl.children)
        while pending:
            item = pending.pop(0)
            result.append(item)
            pending.extend(item.children)
        return result

    def _build_path(self, imp: mi_builder.Importer, tl: Any, root: bpy.types.Object) -> None:
        report = self.owner.report
        points = [p for p in self._descendants(tl) if p.type == "pathpoint" and p.id in imp.base_world]
        report.created["path"] += 1
        if len(points) < 2:
            root["mi_path_note"] = "Path has fewer than two visible points"
            return
        inverse = imp.base_world[tl.id].inverted_safe()
        curve = bpy.data.curves.new(f"{root.name} Geometry", "CURVE")
        curve.dimensions = "3D"
        curve.bevel_depth = 0.025
        curve.bevel_resolution = 1
        spline = curve.splines.new("POLY")
        spline.points.add(len(points) - 1)
        for index, point in enumerate(points):
            coordinate = (inverse @ imp.base_world[point.id]).to_translation()
            spline.points[index].co = (*coordinate, 1.0)
        obj = bpy.data.objects.new(curve.name, curve)
        self._collection("helpers").objects.link(obj)
        obj.parent = root     # identity basis + identity parent inverse: points are in the path's local space
        curve.materials.append(_new_material("MI Path", color=(0.2, 0.8, 1.0, 1.0)))

    # -- world scenery ---------------------------------------------------------------------

    def _scenery(self, imp: mi_builder.Importer, tl: Any, root: bpy.types.Object) -> None:
        owner = self.owner
        report = owner.report
        project_path = Path(imp.project.path)
        template = imp.project.templates.get(tl.temp)
        resource = imp.project.resource(template.scenery) if template is not None else None
        collection = self._collection("scenery")
        if resource is None:
            # Mine-imator creates an empty scenery slot for a new project. It has no visible geometry, so
            # a rendered missing-resource cube is misleading. Keep the slot for source accounting only.
            root["mi_scenery_note"] = "No scenery resource is assigned; nothing visible to import"
            root.empty_display_size = 0.05
            root.show_name = False
            root.hide_viewport = True
            root.hide_render = True
            report.notes.append(f"{root.name}: empty scenery slot contains no visible world geometry")
            report.created["empty_scenery"] += 1
            return
        data = resource.data
        bounds = core.world_bounds({"start": data.get("world_box_start"), "end": data.get("world_box_end")})
        region_dir = Path(str(data.get("world_regions_dir", "")))
        world_dir = region_dir.parent
        cache_path = meshcache.discover(project_path, str(resource.filename))
        world_name = tl.name or resource.filename or "World Scenery"
        mineways = core.find_mineways(owner.options.mineways_path)
        world_ok = bool(bounds) and region_dir.is_dir() and world_dir.is_dir()
        if (not mineways or not world_ok) and cache_path and owner.assets:
            _import_meshcache_world(cache_path, world_name, root, tl, template, collection, project_path, owner.assets, report)
            return
        if not mineways:
            details = "Mineways is not configured and no matching Mine-imator meshcache was found"
            if bounds:
                details += f"; export X/Y/Z {bounds[0]} to {bounds[1]} from {data.get('world_regions_dir', '')}"
            self._placeholder(root, world_name, details)
            return
        if not world_ok:
            details = "saved world path or crop is invalid and no matching Mine-imator meshcache was found"
            if cache_path and not owner.assets:
                details = f"meshcache found at {cache_path}, but the matching Mine-imator asset pack is unavailable"
            self._placeholder(root, world_name, details)
            return
        export_dir = Path(tempfile.gettempdir()) / "mineimator_mcprep_bridge" / "mineways"
        export_dir.mkdir(parents=True, exist_ok=True)
        stem = core.safe_filename(f"{project_path.stem}-{tl.id or 'scenery'}")
        obj_path = export_dir / f"{stem}.obj"
        script_path = export_dir / f"{stem}.mwscript"
        log_path = export_dir / f"{stem}.log"
        start, end = bounds
        script = "\n".join((
            f"Save log file: {log_path}",
            "Show informational: script",
            "Show warning: script",
            "Show error: script",
            "Set render type: Wavefront OBJ absolute indices",
            f"Minecraft world: {world_dir}",
            f"Selection location min to max: {start[0]}, {start[1]}, {start[2]} to {end[0]}, {end[1]}, {end[2]}",
            "File type: Export all textures to three large images",
            "Center model: YES",
            "Make Z the up direction instead of Y: no",
            "Use biomes: yes",
            f"Export for Rendering: {obj_path}",
            "",
        ))
        script_path.write_text(script, encoding="utf-8")
        completed = subprocess.run(
            [str(mineways), "-headless", "-s", str(world_dir.parent), str(script_path)],
            capture_output=True,
            text=True,
            timeout=300,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if completed.returncode != 0 or not obj_path.is_file():
            log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
            message = (log_text or completed.stderr or completed.stdout or f"exit code {completed.returncode}").strip().replace("\n", " ")
            raise RuntimeError(f"Mineways export failed: {message[:400]}")
        before = set(bpy.data.objects)
        bpy.ops.wm.obj_import(filepath=str(obj_path), forward_axis="NEGATIVE_Z", up_axis="Y")
        imported = [obj for obj in bpy.data.objects if obj not in before]
        if not imported:
            raise RuntimeError("Mineways exported an OBJ but Blender imported no objects")
        _tag(root, project_path, tl, obj_path)
        root["mi_world_region_dir"] = str(region_dir)
        root["mi_world_bounds_xyz"] = json.dumps(bounds)
        # Mine-imator's world builder maps Minecraft Z->engine X and Minecraft X->engine Y. Blender's OBJ
        # importer instead produces (world X, -world Z, world Y), so transpose the horizontal axes into
        # Mine-imator's (world Z, -world X, world Y) local basis.
        world_basis = Matrix((
            (0.0, -1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ))
        for obj in imported:
            _link_object(obj, collection)
            obj.parent = root
            obj.matrix_basis = world_basis @ obj.matrix_basis
            for material in getattr(obj.data, "materials", []):
                if material and material.use_nodes:
                    for node in material.node_tree.nodes:
                        if node.type == "TEX_IMAGE":
                            node.interpolation = "Closest"
            report.created["scenery_mesh"] += 1
        report.notes.append(f"Mineways export: {obj_path}")

    def _placeholder(self, root: bpy.types.Object, name: str, message: str) -> None:
        # The root empty keeps the hierarchy and transform for whatever hangs off a missing resource,
        # and draws nothing.
        root["mi_placeholder_reason"] = message
        root.name = f"MISSING - {name}"
        self.owner.report.placeholders.append(f"{name}: {message}")
        self.owner.report.created["placeholder"] += 1


class SceneImporter:
    """Imports one project/object/model: Suite collections + Importer core + Suite extras + environment."""

    def __init__(self, context: bpy.types.Context, path: str | os.PathLike[str], options: ImportOptions):
        self.context = context
        self.path = resolve_import_path(path)
        self.options = options
        self.report = ImportReport(project_path=str(self.path))
        self.assets: core.AssetStore | None = None
        self.root_collection: bpy.types.Collection | None = None
        self.collections: dict[str, bpy.types.Collection] = {}
        self.project: core.ProjectIndex | None = None   # Suite view of the project (environment, format 33+)
        self.importer: mi_builder.Importer | None = None

    # -- Suite project view ------------------------------------------------------------------

    def load_suite_project(self) -> None:
        """The Suite environment needs the raw project index; it only exists for .miproject 2.0+."""
        if self.path.suffix.lower() != ".miproject":
            return
        try:
            data = core.load_json(self.path)
            fmt = int(data.get("format", -1))
        except (core.BridgeError, TypeError, ValueError) as exc:
            self.report.fallbacks.append(f"Suite environment unavailable: {exc}")
            return
        if fmt < 33:
            self.report.notes.append(
                f"Project format {fmt}: the Mine-imator Suite environment needs a 2.0+ project (format 33+); "
                "geometry was imported normally"
            )
            return
        self.project = core.ProjectIndex(self.path, data)
        if fmt != core.SUPPORTED_FORMAT:
            self.report.notes.append(f"Project format {fmt}: the Suite environment is tuned for format {core.SUPPORTED_FORMAT}")

    def prepare_assets(self) -> None:
        install = core.find_mineimator_install(self.path, self.options.mineimator_path)
        asset_path = Path(self.options.asset_pack_path) if self.options.asset_pack_path else None
        if asset_path and asset_path.is_file() and asset_path.suffix.lower() == ".zip":
            metadata_path = asset_path.with_suffix(".midata")
            if metadata_path.is_file():
                try:
                    base = install or asset_path.parent
                    self.assets = core.AssetStore.open_paths(base, asset_path.stem, metadata_path, asset_path)
                    self.report.notes.append(f"Mine-imator assets: {self.assets.asset_version} ({self.assets.zip_path})")
                    return
                except core.BridgeError as exc:
                    self.report.missing.append(str(exc))
        if not install:
            self.report.missing.append("Mine-imator installation not found; bundled characters and special blocks will use placeholders")
            return
        try:
            self.assets = core.AssetStore.open(install)
            self.report.notes.append(f"Mine-imator assets: {self.assets.asset_version} ({self.assets.zip_path})")
        except core.BridgeError as exc:
            self.report.missing.append(str(exc))


    def mineimator_dir(self) -> str:
        install = core.find_mineimator_install(self.path, self.options.mineimator_path)
        if install:
            return str(install)
        asset = Path(self.options.asset_pack_path) if self.options.asset_pack_path else None
        if asset and asset.is_file():
            return str(asset.parent.parent.parent)      # <install>/Data/Minecraft/<version>.zip
        return self.options.mineimator_path or ""

    def create_collections(self) -> None:
        name_source = self.importer.project.name if self.importer else self.path.stem
        base = f"Mine-imator - {name_source}"
        name = core.unique_name(base, bpy.data.collections.keys())
        self.root_collection = bpy.data.collections.new(name)
        self.context.scene.collection.children.link(self.root_collection)
        self.root_collection["mi_bridge_version"] = core.ADDON_VERSION
        self.root_collection["mi_source_project"] = str(self.path)
        self.root_collection["mi_created_in"] = self.importer.project.created_in if self.importer else ""
        self.root_collection["mi_3d_outer_layers"] = self.options.outer_layers_3d
        self.context.scene["mi_bridge_last_import_collection"] = name
        self.report.collection_name = name
        for key, label in CATEGORY_NAMES.items():
            if key in self.options.categories:
                child = bpy.data.collections.new(label)
                self.root_collection.children.link(child)
                self.collections[key] = child

    def import_scene(self) -> ImportReport:
        options = self.options
        self.load_suite_project()
        self.prepare_assets()
        if options.outer_layers_3d:
            self.report.notes.append("3D character outer layers: enabled for player-skin hats, jackets, sleeves, and pants")
        mi_options = mi_builder.Options(
            version=options.version, mi_dir=self.mineimator_dir(), asset_zip=options.asset_pack_path,
            frame=options.frame, animation=options.animation, anim_step=options.anim_step,
            unit_scale=options.unit_scale, import_hidden=options.import_hidden,
            pack_textures=options.pack_textures, merge_vertices=options.merge_vertices,
            depth_masks=options.depth_masks, rig=options.rig, ik=options.ik, ik_keep_pose=options.ik_keep_pose,
            bend_weights=options.bend_weights, standard_view=options.standard_view,
            outer_layers_3d=options.outer_layers_3d, categories=set(options.categories),
        )
        extras = SuiteExtras(self)
        # Parse and validate first, so a bad file fails before the scene is touched.
        self.importer = mi_builder.Importer(str(self.path), mi_options, extras=extras)
        if options.remove_startup_cube:
            _remove_untouched_startup_cube(self.context, self.report)
        self.create_collections()
        mi_options.category_collections = self.collections
        self.importer.run(self.context, collection=self.root_collection)
        self._collect_report(self.importer)
        self._audit_world_size()
        self._environment()
        self._mcprep_materials()
        if not options.animation:
            self._ensure_static()
        if options.outer_layers_3d:
            self.context.scene["mi_bridge_last_3d_outer_layer_count"] = int(self.report.created["3d_outer_layer"])
        write_report(self.report)
        return self.report

    def _collect_report(self, imp: mi_builder.Importer) -> None:
        report = self.report
        report.animation_imported = bool(self.options.animation)
        report.notes.append(f"Importer core: {imp.project.version_label} (format {imp.project.file_format})")
        for timeline in imp.project.roots():
            if timeline.type == "audio":
                report.skipped["audio"] += 1
            else:
                report.imported[timeline.type] += 1
        for kind, count in imp.stats.items():
            if kind in ("3d_outer_layer", "3d_outer_polys"):
                report.created[kind] += count
            else:
                report.created[f"object_{kind}"] += count
        if imp.rigs:
            report.created["armature_rig"] += len(imp.rigs)
        for kind, count in imp.skipped.items():
            report.skipped[kind or "unknown"] += count
        for category, count in imp.skipped_category.items():
            report.skipped[f"category:{category}"] += count
        for message in imp.messages:
            lowered = message.lower()
            if message.startswith("Error"):
                report.fallbacks.append(message)
            elif message.startswith("Warning") or "not found" in lowered or "could not be loaded" in lowered:
                report.missing.append(message)
            else:
                report.notes.append(message)
        if self.options.outer_layers_3d and not imp.stats.get("3d_outer_layer"):
            report.notes.append("3D outer layers were enabled but no player-skin outer layer was found in this import")

    def _audit_world_size(self, max_blocks: float = 3.5, max_distance: float = 14.0) -> None:
        """List meshes that end up unusually large or far from the origin.

        The report names them (with their local scale), which is the quickest way to
        pin down a misplaced shape without guessing.
        """
        try:
            bpy.context.view_layer.update()
        except Exception:
            return
        flagged: list[str] = []
        for collection in self.collections.values():
            for obj in collection.all_objects:
                if obj.type != "MESH" or len(obj.data.vertices) < 2:
                    continue
                corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
                lows = [min(c[i] for c in corners) for i in range(3)]
                highs = [max(c[i] for c in corners) for i in range(3)]
                size = max(highs[i] - lows[i] for i in range(3))
                center = [(lows[i] + highs[i]) * 0.5 for i in range(3)]
                if size > max_blocks or math.sqrt(sum(c * c for c in center)) > max_distance:
                    scale = obj.matrix_basis.to_scale()
                    flagged.append(
                        f"audit: '{obj.name}' (parent '{obj.parent.name if obj.parent else '-'}') spans {size:.1f} blocks at "
                        f"({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}), local scale ({scale[0]:.3f}, {scale[1]:.3f}, {scale[2]:.3f})"
                    )
        self.report.fallbacks.extend(flagged[:15])
        if len(flagged) > 15:
            self.report.fallbacks.append(f"audit: ... and {len(flagged) - 15} more oversized/far objects")

    def _environment(self) -> None:
        if "environment" not in self.options.categories:
            return
        if self.project is None:
            self.report.notes.append("No Suite project index for this file type/format; environment import skipped")
            return
        if self.options.mineimator_suite:
            from . import suite_environment
            suite_environment.build_suite(
                self.context,
                self.collections["environment"],
                self.project,
                self.assets,
                self.report,
            )
            return
        background = self.project.data.get("background", {})
        if not isinstance(background, dict):
            return
        world = self.context.scene.world or bpy.data.worlds.new("Mine-imator World")
        self.context.scene.world = world
        world.use_nodes = True
        nodes = world.node_tree.nodes
        node = nodes.get("Background")
        color = background.get("color") or background.get("sky_color") or "#87A9FF"
        if node:
            node.inputs["Color"].default_value = _hex_color(color)
            node.inputs["Strength"].default_value = float(background.get("brightness", 0.8))
        if background.get("fog_show"):
            distance = max(1.0, float(background.get("fog_distance", 10000.0)) / core.MI_UNITS_PER_BLOCK)
            size = max(1.0, float(background.get("fog_size", 2000.0)) / core.MI_UNITS_PER_BLOCK)
            height = max(10.0, float(background.get("fog_height", 1250.0)) / core.MI_UNITS_PER_BLOCK)
            extent = distance + size
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, height / 2.0))
            fog = self.context.active_object
            fog.name = "Mine-imator Fog Volume"
            fog.dimensions = (extent * 2.0, extent * 2.0, height)
            _link_object(fog, self.collections["environment"])
            material = bpy.data.materials.new("Mine-imator Fog")
            material.use_nodes = True
            fog_nodes = material.node_tree.nodes
            fog_nodes.clear()
            output = fog_nodes.new("ShaderNodeOutputMaterial")
            volume = fog_nodes.new("ShaderNodeVolumePrincipled")
            volume.inputs["Color"].default_value = _hex_color(background.get("fog_color", color))
            volume.inputs["Density"].default_value = min(0.02, 0.12 / extent)
            material.node_tree.links.new(volume.outputs["Volume"], output.inputs["Volume"])
            fog.data.materials.append(material)
            fog.display_type = "WIRE"
            fog.hide_set(True)
            fog.hide_render = False
            fog["mi_bridge_version"] = core.ADDON_VERSION
            self.report.created["fog"] += 1
        if background.get("sunlight", background.get("sun", True)):
            data = bpy.data.lights.new("Mine-imator Sun", "SUN")
            data.energy = float(background.get("sunlight_strength", 1.0))
            sun = bpy.data.objects.new(data.name, data)
            self.collections["environment"].objects.link(sun)
            sun.rotation_euler = (math.radians(35), 0, math.radians(-35))
            sun["mi_bridge_version"] = core.ADDON_VERSION
            self.report.created["sun"] += 1
        if background.get("ground_show"):
            bpy.ops.mesh.primitive_plane_add(size=2000.0)
            ground = self.context.active_object
            ground.name = "Mine-imator Ground"
            _link_object(ground, self.collections["environment"])
            ground_name = str(background.get("ground_name", "block/grass_block_top"))
            texture = self.assets.materialize(f"textures/{ground_name}.png") if self.assets else None
            if not texture:
                texture = _mcprep_texture(ground_name)
            ground.data.materials.append(_new_material("Mine-imator Ground", texture, _hex_color(background.get("grass_color", "#91BD59"))))
            ground["mi_bridge_version"] = core.ADDON_VERSION
            self.report.created["ground"] += 1
        self.context.scene["mi_environment_source"] = json.dumps(background, separators=(",", ":"))[:60000]


    def _mcprep_materials(self) -> None:
        if not self.options.use_mcprep:
            return
        meshes = [
            obj for obj in self.root_collection.all_objects
            if obj.type == "MESH" and not obj.get("mi_suite_id")
        ] if self.root_collection else []
        # MCprep's material preparation rebuilds the node tree around the texture, which would discard the
        # Multiply/Add/Mix nodes that reproduce Mine-imator's color on shapes and tinted parts.
        tinted = [obj for obj in meshes if any(slot.material and slot.material.get("mi_tinted") for slot in obj.material_slots)]
        if tinted:
            meshes = [obj for obj in meshes if obj not in tinted]
            self.report.notes.append(
                f"MCprep material preparation skipped on {len(tinted)} colored objects to keep their Mine-imator color nodes"
            )
        if not meshes:
            return
        try:
            bpy.ops.object.select_all(action="DESELECT")
            for obj in meshes:
                obj.select_set(True)
            self.context.view_layer.objects.active = meshes[0]
            if hasattr(bpy.ops, "mcprep") and hasattr(bpy.ops.mcprep, "prep_materials"):
                # MCprep exposes this hidden flag for meta-operators; it also
                # avoids its optional telemetry wrapper affecting the return.
                result = bpy.ops.mcprep.prep_materials(skipUsage=True)
                self.report.notes.append(f"MCprep material preparation: {','.join(result)}")
            else:
                self.report.notes.append("MCprep material operator was not registered; bridge materials remain pixel-art ready")
        except Exception as exc:
            self.report.fallbacks.append(f"MCprep material preparation skipped: {exc}")


    def _ensure_static(self) -> None:
        if not self.root_collection:
            return
        for obj in self.root_collection.all_objects:
            if obj.animation_data:
                obj.animation_data_clear()
            if getattr(obj.data, "animation_data", None):
                obj.data.animation_data_clear()



def preflight(project_value: str, options: ImportOptions) -> ImportReport:
    path = resolve_import_path(project_value)
    project = mi_data.Project(str(path), options.version)
    report = ImportReport(project_path=str(path))
    report.notes.append(f"{project.version_label}: project format {project.file_format}")
    install = core.find_mineimator_install(path, options.mineimator_path)
    if install:
        try:
            store = core.AssetStore.open(install)
            report.notes.append(f"Mine-imator {store.asset_version} asset pack found: {store.zip_path}")
        except core.BridgeError as exc:
            report.missing.append(str(exc))
    else:
        found = mi_assets.find_mineimator_dir(str(path), options.mineimator_path, options.version)
        if found:
            report.notes.append(f"Mine-imator resources found: {found}")
        else:
            report.missing.append("Mine-imator installation was not detected")
    mineways = core.find_mineways(options.mineways_path)
    if mineways:
        report.notes.append(f"Mineways found: {mineways}")
    else:
        report.notes.append("Mineways is not configured; scenery uses a matching .meshcache when available, otherwise a placeholder")
    for timeline in project.roots():
        if timeline.type == "audio":
            report.skipped["audio"] += 1
        else:
            report.imported[timeline.type] += 1
    for resource in project.resources.values():
        if resource.type in {"skin", "downloadskin", "downloadedskin", "model", "texture"} and not mi_assets.find_project_file(project.folder, resource.filename):
            report.missing.append(f"Project resource missing: {resource.filename or resource.id}")
    return report


def write_report(report: ImportReport) -> bpy.types.Text:
    base = "Mine-imator Bridge Report"
    name = core.unique_name(base, bpy.data.texts.keys())
    text = bpy.data.texts.new(name)
    text.write(report.text())
    return text
