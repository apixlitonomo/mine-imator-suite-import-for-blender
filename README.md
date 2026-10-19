# Mine-imator Suite + Import for Blender 0.4.0

The Mine-imator Suite (environment, world scenery, camera and lights, Render Export, MCprep) merged with the
Mine-imator Importer core (by Alexandra). The importer core now builds all geometry; the Suite supplies
everything around it.

## Known Bugs (may be fixed later)

1. Some materials will appear white if "Run Mcprep material preparation" is active while importing
2. Imported grass blocks or transparency blocks will appear black (this can be manually fixed by changing the option in the transparency material from hashed to blended)
3. some UVs from imported terrain are upside down (lmao how did that happen)

## What comes from where

**Importer core (replaces the old Suite importing code):** characters, special blocks, custom `.mimodel`
models, items, blocks, primitive shapes and text; armatures with exact bends; Mine-imator IK; depth masks;
optional keyframe baking; `.miproject` (formats 1.1 to Vexel), `.miobject` and `.mimodel` files; drag and drop.

**Suite (kept):** Mine-imator environment (sky, sun, moon, clouds, fog, biome, ground) with live sidebar
controls, world scenery (Mineways or `.meshcache`), camera, lights, paths, particle placeholders, category
collections and filters, 3D character outer layers, MCprep preparation, startup-cube cleanup, report,
preflight and Render Export.

## Install and use

1. **Edit > Preferences > Get Extensions > Install from Disk**, pick the zip, enable it.
2. **File > Import > Mine-imator (.miproject/.miobject/.mimodel)**, or the **MI Bridge** sidebar tab.
3. Use **Preflight** to validate, then **Import Scene**; read the *Mine-imator Bridge Report* text.

## Changes in 0.4.0

- Importer core replaces the Suite's model, bend, item, block, shape and text builders. The removed modules are
  `modelgeom.py`, `shapegen.py` and `scenemath.py`.
- New sidebar section **Pose, Rigging and Animation**: frame, animation baking, armatures, weighted bends, IK,
  depth masks, scale, packing and Standard view. Animation is off by default, so imports stay a single pose.
- 3D outer layers were reimplemented on the new geometry path (`voxelize_block`), still one texel deep.
- Cameras, lights and paths use the importer's evaluated transforms, so parenting, inheritance flags and roll
  are honoured. Camera values use Mine-imator's real keys (`CAM_FOV`, `LIGHT_RANGE`, ...).
- **Use frame-0 item swaps** was removed: the importer core applies custom item slots natively.
- Importer messages and labels were translated to English (code comments remain Spanish).
- The Suite environment needs a 2.0+ project (format 33+); older formats import geometry only.

## Render Export

The integrated still-image exporter is available from **File > Export Image...**
and from **3D View > Sidebar > MI Bridge > Render Export**. Choose a size preset
or custom resolution, select a camera, optionally remove the background or
include render-hidden objects, then choose **Render and Save PNG**. It renders
the current frame and restores the scene's original render settings afterward.

## 0.3.0 Render Export integration

- Integrates Simple Render Export 1.0.0 into the MI Bridge package.
- Adds nine image-size presets plus custom dimensions, camera selection,
  transparent-background PNG output, and optional hidden-object rendering.
- Adds matching File-menu and MI Bridge sidebar controls without requiring a
  second add-on installation.

## 0.3.1 Project name

- Adopts the public name **Mine-imator Suite + Import for Blender** while
  retaining the stable extension ID for seamless upgrades.

## 0.3.2 Scenery cache and sunlight fixes

- Recovers world scenery directly from Mine-imator format-2 `.meshcache`
  files when the original Minecraft save path is missing or Mineways is not
  available. Matching caches can be discovered in sibling Mine-imator project
  folders after a project is duplicated with Save As.
- Rebuilds Mine-imator's exact static and animated block atlases from the
  installed Minecraft asset pack, including transparent blocks, biome-tinted
  grass and leaves, and water.
- Keys the bundled sun and moon textures' opaque black backgrounds to
  transparency.
- Calibrates the real Blender Sun from Mine-imator's time-of-day direction,
  twilight color, horizon fade, strength, shadow angle, and shallow-incidence
  compensation so dawn and dusk still cast useful directional light.

## 0.3.3 Scenery placement fix

- Reproduces Mine-imator's automatic center pivot and legacy 90-degree cache
  basis in the correct order. Meshcache worlds now retain the saved position
  and orientation instead of rotating around a crop corner and exposing the
  selection walls or ceiling near the scene.

## 0.3.4 3D character outer layers

- Adds an optional **3D character outer layers** import setting. It converts
  opaque pixels in player hats, jackets, sleeves, and pants into editable 3D
  surface geometry, with real depth around transparent gaps and silhouette
  edges.
- The setting is disabled by default and affects player-skin models only;
  custom models and entity shells keep their authored Mine-imator geometry.

## 0.3.5 Live environment and visible voxel depth

- Gives 3D player outer-layer pixels a full texel of depth, matching the
  silhouette thickness of the 3D Skin Layers style and remaining visible at
  ordinary camera distances.
- Makes directional sunlight strength linear and visibly responsive while
  retaining the time-of-day fade at sunrise, sunset, and night.
- Clarifies that **Time of day** and **Sky rotation** position the sun, while
  **Shadow softness** controls the Blender Sun's angular diameter and the Sun
  texture rotation only spins the visible disc.
- Adds **Apply Current Values** to force-refresh the active World and Sun, plus
  an import-status message showing how many voxel-layer parts were generated.
- Respects Mine-imator's saved ground visibility. Projects with ground disabled
  now create the editable suite ground hidden instead of forcing it visible.

## 0.3.6 Persistent 3D-layer default

- Adds a persistent **Default 3D outer layers** preference in both the MI
  Bridge panel and Blender's extension preferences.
- The preference controls the starting state for new scenes and File > Import;
  the existing **3D character outer layers** checkbox remains a per-import
  override.

The **Use frame-0 item swaps** option is disabled by default because some
Mine-imator projects contain stale `ITEM_NAME` compatibility hints. Enable the
option only when the frame-0 swap is known to be intentional.

## 0.2.0 Mine-imator Suite

- Adds a default-enabled full environment import with Minecraft grass ground,
  day/night sky, sunlight, textured sun and moon, moon phases, pixel stars,
  Minecraft clouds, biome colors, and fog.
- Adds a live `MI Environment` sidebar modeled after Mine-imator's environment
  panels. Appearance settings update Blender immediately while wind and motion
  speeds remain stored static values.
- Repeated suite imports preserve older environments but make the newest suite
  active. Disabling Mine-imator Suite retains the legacy environment importer.

## 0.1.5 fixes

- Detect and remove Blender's untouched default startup cube during import.
  The check requires its original name, mesh, transform, collection, Camera,
  and Light, so user-created or edited cubes are preserved.
- Added an enabled-by-default option to retain the startup cube if desired.

## 0.1.4 fixes

- Hide Mine-imator's unused empty-world slot from the viewport and renders,
  while retaining it as a metadata-only Outliner entry for source accounting.

## 0.1.3 fixes

- Rebuilt pixel items using Mine-imator's bottom-up texture coordinates,
  fixed one-MI-unit extrusion depth, and saved rotation point.
- Removed internal per-pixel cube faces and generate only the visible front,
  back, and silhouette boundary surfaces.
- Corrected the apparent sword, axe, bow, and arrow angles by using
  Mine-imator's bottom-centered item pivot rather than a centered mesh origin.

## 0.1.2 fixes

- Per-instance frame-0 skin and model texture overrides.
- Correct mixed-axis scene rotation order and declared bend-angle limits.
- Correct local positions and rotations for multi-shape custom `.mimodel`
  files.
- Template item identity by default, with opt-in frame-0 item swaps.
- Empty Mine-imator scenery slots now remain non-rendering viewport markers
  instead of producing an unexplained placeholder cube.

## 0.1.1 fixes

- Exact mixed-axis Mine-imator rotation composition for frame-zero poses.
- Correct Mine-imator cuboid UV face layout and mirrored textures.
- Blocky bend-plane splitting, eliminating stretched cuboid faces.
- Relative hierarchy linking for held items and nested timelines.
- Automatic Mineways detection plus Mine-imator/Minecraft world-axis
  correction for scenery placement.
