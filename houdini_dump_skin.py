#!/usr/bin/env python3
"""Dump a skinned KineFX mesh from Houdini to JSON for glb_from_houdini.py.

Use inside Houdini / hython, after the mesh has capture weights:

    from houdini_dump_skin import dump_skin_data
    dump_skin_data(mesh_sop, skeleton_sop, "/path/asset/skindata.json",
                   "armor_male_apprentice", "/path/asset/texture.png")

  mesh_sop     a SOP with triangle polygons, point N, vertex uv, and the UNPACKED
               capture arrays (Capture Attribute Unpack): boneCapture_index /
               boneCapture_data, plus the detail boneCapture_pCaptPath.
  skeleton_sop points with `name` + `transform` (matrix3); the hierarchy is read from
               two-point polygons (parent vertex -> child vertex).

The JSON holds what glTF needs:

  joints     [{name, parent, matrix: 4x4 row-major world}]
  positions  flat xyz per triangle corner
  normals    flat xyz per corner
  uvs        flat uv per corner
  joints4    four joint indices per corner (ubyte-safe, < 256 joints)
  weights4   four normalised weights per corner
  indices    0..n-1 (corners are already de-indexed)
"""

from __future__ import annotations

import json
from pathlib import Path


def dump_skin_data(mesh_sop, skel_sop, out_json, name, texture=""):
    import hou  # noqa: F401  (only exists inside Houdini)

    g = mesh_sop.geometry()
    gs = skel_sop.geometry()

    joints, jindex = [], {}
    for p in gs.points():
        m3 = [float(v) for v in p.attribValue("transform").asTuple()]
        pos = p.position()
        m = [[m3[0], m3[1], m3[2], pos.x()],
             [m3[3], m3[4], m3[5], pos.y()],
             [m3[6], m3[7], m3[8], pos.z()],
             [0.0, 0.0, 0.0, 1.0]]
        jindex[p.attribValue("name")] = len(joints)
        joints.append({"name": p.attribValue("name"), "matrix": m, "parent": ""})
    for prim in gs.prims():
        vs = prim.vertices()
        if len(vs) == 2:
            child = vs[1].point().attribValue("name")
            parent = vs[0].point().attribValue("name")
            joints[jindex[child]]["parent"] = parent

    paths = list(g.attribValue("boneCapture_pCaptPath"))

    def joint_of(path):
        path = str(path)
        for nm in jindex:
            if nm == path or path.endswith(nm):
                return jindex[nm]
        return 0

    has_uv = g.findVertexAttrib("uv") is not None
    has_vn = g.findVertexAttrib("N") is not None

    positions, normals, uvs, jj, ww, indices = [], [], [], [], [], []
    for prim in g.prims():
        vs = prim.vertices()
        if len(vs) != 3:
            continue
        for v in vs:
            p = v.point()
            pos = p.position()
            positions += [pos.x(), pos.y(), pos.z()]
            n = v.attribValue("N") if has_vn else p.attribValue("N")
            normals += [n[0], n[1], n[2]]
            t = v.attribValue("uv") if has_uv else (0.0, 0.0)
            uvs += [t[0], t[1]]

            idxs = p.attribValue("boneCapture_index")
            wts = p.attribValue("boneCapture_data")
            pairs = sorted(((float(w), joint_of(paths[i])) for i, w in zip(idxs, wts)), reverse=True)[:4]
            tot = sum(w for w, _ in pairs) or 1.0
            js = [j for _, j in pairs] + [0] * (4 - len(pairs))
            ws = [w / tot for w, _ in pairs] + [0.0] * (4 - len(pairs))
            jj += js
            ww += ws
        indices += list(range(len(indices), len(indices) + 3))

    data = {"name": name, "texture": str(texture) if texture else "",
            "joints": joints, "positions": positions, "normals": normals, "uvs": uvs,
            "joints4": jj, "weights4": ww, "indices": indices}
    Path(out_json).write_text(json.dumps(data))
    return out_json
