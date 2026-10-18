# houdini-skin-glb

Assemble a **skinned glTF 2.0** `.glb` from Houdini/KineFX capture weights, without the
KineFX *GLTF Character* node.

Houdini **Apprentice cannot export skinned glTF** — only Core/FX can:

```
glTF export is only supported in Houdini Core and Houdini FX versions.
```

The plain glTF ROP still writes static meshes, but it drops the skin: no `skins`, no
`JOINTS_0`/`WEIGHTS_0`. This tool fills that gap. Houdini does the work it is good at — fit,
decimate, capture weights — then dumps JSON; a dependency-free Python script builds the glTF.

## How it works

```
Houdini (any license)                       Python (any machine)
┌────────────────────────────┐              ┌──────────────────────────┐
│ mesh + Capture Attribute   │  skindata.   │ glb_from_houdini.py      │
│ Unpack + KineFX skeleton   │───json──────▶│  → <asset>_fit.glb       │
│ houdini_dump_skin.py       │              │  (skin + joints + tex)   │
└────────────────────────────┘              └──────────────────────────┘
```

The JSON carries:

| field | contents |
|---|---|
| `joints` | `{name, parent, matrix}` — 4×4 **row-major world** matrix per joint |
| `positions`, `normals`, `uvs` | flat per triangle corner (the mesh is de-indexed) |
| `joints4`, `weights4` | four influences per corner, normalised |
| `indices` | `0..n-1` |
| `texture` | optional path to the base-colour image (PNG/JPEG) |

The converter computes `inverseBindMatrices` from the world matrices, writes joint nodes
with `matrix`, one skinned mesh node, the skin, a Principled-compatible material and embeds
the texture. Pure standard library: no numpy, no trimesh.

## Houdini side

The mesh must have capture weights from KineFX. The expected chain is:

```
mesh (File + Clean + Normal + Triangulate + Polyreduce)
  └ kinefx::jointcaptureproximity   (input 1 = skeleton)
     └ captureattribunpack
        └ attribwrangle               (prune to 4 influences, renormalise)
           └ [dump from here]
skeleton: points with name + transform (matrix3), two-point polys parent → child
```

```python
import sys
sys.path.append("/path/to/houdini-skin-glb")
from houdini_dump_skin import dump_skin_data

dump_skin_data(
    hou.node("/obj/EQ_armor/OUT"),          # UNPACKED capture arrays
    hou.node("/obj/EQ_armor/skeleton"),     # KineFX skeleton
    "/tmp/armor/skindata.json",
    "armor_male_apprentice",
    "/tmp/armor/texture.png",
)
```

Then, outside Houdini:

```bash
python3 glb_from_houdini.py /tmp/armor/skindata.json /tmp/armor/armor_fit.glb
```

It also runs from `hython` directly, since it is pure Python.

## Three.js

The exported skin uses the same joint names as the body's glTF (Mixamo names here), so one
animation drives both:

```js
const body = await loader.loadAsync("base_male.glb");
const gear = await loader.loadAsync("armor_male_apprentice_fit.glb");

const clip = SkeletonUtils.clone(body.scene);
const bones = {};
clip.traverse(o => { if (o.isBone) bones[o.name] = o; });

gear.scene.traverse(o => {
  if (o.isSkinnedMesh) {
    o.bind(new THREE.Skeleton(Object.values(bones), clip.skeleton.boneInverses), clip.matrixWorld);
  }
});
clip.add(gear.scene);
```

## Notes

- `JOINTS_0` is `UNSIGNED_BYTE`, so keep the skeleton under 256 joints (Mixamo biped ≈ 60).
- Weights are renormalised on export; more than four influences are pruned upstream.
- UVs are per-corner; the mesh is de-indexed rather than welded, which keeps UV seams exact
  and costs vertices, not correctness.
- The texture is embedded; pass nothing and the material is flat white.
- The world matrices come from the skeleton's rest pose, so the mesh must be dumped in that
  same pose (the capture pose), not after posing.

## License

MIT
