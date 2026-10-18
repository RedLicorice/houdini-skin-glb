#!/usr/bin/env python3
"""Assemble a skinned .glb from the JSON that embervale_pipeline.py dumps out of Houdini.

Houdini Apprentice cannot export skinned glTF (the KineFX GLTF Character node is Core/FX
only), so the pipeline writes the fitted mesh, its capture weights, the skeleton and the
texture path to skindata.json; this turns that into a standard glTF 2.0 binary with:

  - one skinned mesh (POSITION, NORMAL, TEXCOORD_0, JOINTS_0, WEIGHTS_0, indices)
  - a skin with inverseBindMatrices and the Mixamo joint hierarchy
  - an embedded base-color texture

Pure standard library (no numpy, no trimesh), so it also runs inside hython:

  python3 tools/glb_from_houdini.py skindata.json out.glb
"""

from __future__ import annotations

import base64
import json
import struct
import sys
from pathlib import Path

# ---------------------------------------------------------------- matrix helpers (row-major 4x4)

def mat_mul(a, b):
    return [[sum(a[r][k] * b[k][c] for k in range(4)) for c in range(4)] for r in range(4)]


def mat_inv(m):
    """4x4 inverse via Gauss-Jordan (rows, floats)."""
    n = 4
    aug = [list(m[r]) + [1.0 if r == c else 0.0 for c in range(n)] for r in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) < 1e-12:
            raise ValueError("singular matrix")
        aug[col], aug[piv] = aug[piv], aug[col]
        d = aug[col][col]
        aug[col] = [v / d for v in aug[col]]
        for r in range(n):
            if r != col and abs(aug[r][col]) > 1e-15:
                f = aug[r][col]
                aug[r] = [a - f * b for a, b in zip(aug[r], aug[col])]
    return [row[n:] for row in aug]


def col_major(m):
    """Row-major 4x4 -> glTF column-major 16 floats."""
    return [m[r][c] for c in range(4) for r in range(4)]


# ---------------------------------------------------------------- binary writer

class Buf:
    def __init__(self):
        self.data = bytearray()
        self.views = []
        self.accessors = []

    def _pad(self):
        while len(self.data) % 4:
            self.data.append(0)

    def view(self, blob, target=None):
        self._pad()
        off = len(self.data)
        self.data.extend(blob)
        v = {"buffer": 0, "byteOffset": off, "byteLength": len(blob)}
        if target:
            v["target"] = target
        self.views.append(v)
        return len(self.views) - 1

    def accessor(self, blob, comp, ctype, count, target=None, minmax=False):
        v = self.view(blob, target)
        a = {"bufferView": v, "componentType": comp, "count": count, "type": ctype}
        if minmax:
            arr = struct.unpack("<" + "f" * (len(blob) // 4), blob)
            comps = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[ctype]
            mins = [min(arr[i::comps]) for i in range(comps)]
            maxs = [max(arr[i::comps]) for i in range(comps)]
            a["min"] = mins
            a["max"] = maxs
        self.accessors.append(a)
        return len(self.accessors) - 1


FLOAT, UBYTE, UINT = 5126, 5121, 5125


def build_glb(data: dict, out: Path):
    joints = data["joints"]
    joints_index = {j["name"]: i for i, j in enumerate(joints)}

    # world matrices, locals and inverse binds
    world = []
    for j in joints:
        m = j["matrix"]                       # row-major 4x4
        world.append([list(r) for r in m])
    ibm = [mat_inv(w) for w in world]

    b = Buf()
    pos = data["positions"]
    nrm = data["normals"]
    uv = data["uvs"]
    jj = data["joints4"]
    ww = data["weights4"]
    idx = data["indices"]
    count = len(pos) // 3

    a_pos = b.accessor(struct.pack("<%df" % len(pos), *pos), FLOAT, "VEC3", count, 34962, True)
    a_nrm = b.accessor(struct.pack("<%df" % len(nrm), *nrm), FLOAT, "VEC3", count, 34962)
    a_uv = b.accessor(struct.pack("<%df" % len(uv), *uv), FLOAT, "VEC2", count, 34962)
    a_j = b.accessor(struct.pack("<%dB" % len(jj), *jj), UBYTE, "VEC4", count, 34962)
    a_w = b.accessor(struct.pack("<%df" % len(ww), *ww), FLOAT, "VEC4", count, 34962)
    a_i = b.accessor(struct.pack("<%dI" % len(idx), *idx), UINT, "SCALAR", len(idx), 34963)
    a_ibm = b.accessor(struct.pack("<%df" % (16 * len(ibm)), *[v for m in ibm for v in col_major(m)]),
                       FLOAT, "MAT4", len(ibm))

    # texture
    tex_path = data.get("texture")
    img_view = None
    mime = "image/png"
    if tex_path and Path(tex_path).exists():
        blob = Path(tex_path).read_bytes()
        mime = "image/jpeg" if blob[:2] == b"\xff\xd8" else "image/png"
        img_view = b.view(blob)

    # nodes: joints first, then the mesh
    nodes = []
    for i, j in enumerate(joints):
        n = {"name": j["name"], "matrix": col_major(world[i])}
        nodes.append(n)
    for i, j in enumerate(joints):
        p = j.get("parent")
        if p not in (None, "", -1):
            pi = joints_index.get(p) if isinstance(p, str) else p
            if pi is not None:
                nodes[pi].setdefault("children", []).append(i)
    mesh_node = len(nodes)
    nodes.append({"name": data.get("name", "gear"), "mesh": 0, "skin": 0})

    gltf = {
        "asset": {"version": "2.0", "generator": "embervale glb_from_houdini"},
        "scene": 0,
        "scenes": [{"nodes": [mesh_node]}],
        "nodes": nodes,
        "meshes": [{
            "name": data.get("name", "gear"),
            "primitives": [{
                "attributes": {"POSITION": a_pos, "NORMAL": a_nrm, "TEXCOORD_0": a_uv,
                               "JOINTS_0": a_j, "WEIGHTS_0": a_w},
                "indices": a_i,
                "material": 0,
                "mode": 4,
            }],
        }],
        "skins": [{
            "inverseBindMatrices": a_ibm,
            "joints": list(range(len(joints))),
            "skeleton": 0,
        }],
        "materials": [{
            "name": data.get("name", "gear") + "_mat",
            "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0.0, "roughnessFactor": 0.9},
        }],
        "accessors": b.accessors,
        "bufferViews": b.views,
        "buffers": [{"byteLength": len(b.data)}],
    }
    if img_view is not None:
        gltf["images"] = [{"bufferView": img_view, "mimeType": mime}]
        gltf["samplers"] = [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}]
        gltf["textures"] = [{"sampler": 0, "source": 0}]
        gltf["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"] = {"index": 0}

    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    bin_blob = bytes(b.data)
    bin_blob += b"\x00" * ((4 - len(bin_blob) % 4) % 4)

    total = 12 + 8 + len(js) + 8 + len(bin_blob)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, total))
        f.write(struct.pack("<I4s", len(js), b"JSON"))
        f.write(js)
        f.write(struct.pack("<I4s", len(bin_blob), b"BIN\x00"))
        f.write(bin_blob)
    return out


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    data = json.loads(Path(argv[1]).read_text())
    build_glb(data, Path(argv[2]))
    print(f"wrote {argv[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
