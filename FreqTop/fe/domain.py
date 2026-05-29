"""
FreqTop/fe/domain.py
====================
MeshDomain  — mesh topology: node/element indexing, DOF matrix, edge accessors.
BeamDomain  — extends MeshDomain with JSON-driven BCs, loads, and passive flanges.

Architecture
------------
    MeshDomain          pure geometry — no BCs, no material data
        └─ BeamDomain   parses parameters.json; implements the problem interface
                        used by FESolver and TopOptSolver:
                            nelx, nely, ndof          (from MeshDomain)
                            get_fixed_dofs()
                            get_load_vector()
                            get_passive_elements()
                            get_active_elements()
                            has_passive_elements()
                            correct_design_variable(rho, volfrac)

Node numbering  (column-major, iy fastest)
------------------------------------------
    node(ix, iy) = ix * (nely + 1) + iy
    ix ∈ [0, nelx],  iy ∈ [0, nely]

    Example — 3×2 grid (nelx=3, nely=2):
        col ix=0  ix=1  ix=2  ix=3
        iy=2:  2     5     8    11
        iy=1:  1     4     7    10
        iy=0:  0     3     6     9

Element numbering  (column-major, ely fastest)
----------------------------------------------
    element(elx, ely) = elx * nely + ely
    elx ∈ [0, nelx-1],  ely ∈ [0, nely-1]

DOF numbering
-------------
    Node n  →  DOF 2*n   (x-displacement)
               DOF 2*n+1 (y-displacement)
    Total DOFs = 2 * (nelx+1) * (nely+1)
"""

from __future__ import annotations

import json
import numpy as np


# ---------------------------------------------------------------------------
# MeshDomain
# ---------------------------------------------------------------------------

class MeshDomain:
    """
    Structured 2-D quad mesh: node numbering, element numbering, and DOF indexing.

    Contains pure-geometry helpers only — no boundary conditions, no material
    data, and no optimisation logic.  BeamDomain (below) adds those on top.
    """

    def __init__(
        self,
        nelx: int,
        nely: int,
        length_x: float = 1.0,
        length_y: float = 1.0,
        gamma: float = 1.4,
    ):
        self.nelx  = int(nelx)
        self.nely  = int(nely)
        self.gamma = float(gamma)

        self.nnx   = self.nelx + 1          # nodes in x-direction
        self.nny   = self.nely + 1          # nodes in y-direction
        self.nelxy = self.nelx * self.nely  # total elements
        self.nnxy  = self.nnx  * self.nny   # total nodes

        self.length_x       = float(length_x)
        self.length_y       = float(length_y)
        self.elem_length    = self.length_y / self.nely
        self.volume_element = (self.length_x * self.length_y) / self.nelxy
        self.volume         = self.length_x * self.length_y

    # ------------------------------------------------------------------
    # Property
    # ------------------------------------------------------------------

    @property
    def ndof(self) -> int:
        """Total degrees of freedom: 2 per node."""
        return 2 * self.nnxy

    # ------------------------------------------------------------------
    # Node / element indexing
    # ------------------------------------------------------------------

    def node_index(self, ix: int, iy: int) -> int:
        """
        Global node number at grid position (ix, iy).
        ix ∈ [0, nelx],  iy ∈ [0, nely].
        """
        return ix * self.nny + iy

    def element_index(self, elx: int, ely: int) -> int:
        """
        Global element number for element (elx, ely).
        elx ∈ [0, nelx-1],  ely ∈ [0, nely-1].
        """
        return elx * self.nely + ely

    def node_at_relative(self, rx: float, ry: float) -> int:
        """
        Global node index at relative domain position (rx, ry) ∈ [0, 1]².
        Snaps to the nearest grid node.

            rx = 0 → left edge,   rx = 1 → right edge
            ry = 0 → bottom edge, ry = 1 → top edge
        """
        ix = int(round(rx * self.nelx))
        iy = int(round(ry * self.nely))
        ix = max(0, min(self.nelx, ix))
        iy = max(0, min(self.nely, iy))
        return self.node_index(ix, iy)

    # ------------------------------------------------------------------
    # Edge node arrays
    # ------------------------------------------------------------------

    def left_nodes(self) -> np.ndarray:
        """All nodes on the left edge (ix = 0)."""
        return np.array([self.node_index(0, iy) for iy in range(self.nny)], dtype=int)

    def right_nodes(self) -> np.ndarray:
        """All nodes on the right edge (ix = nelx)."""
        return np.array([self.node_index(self.nelx, iy) for iy in range(self.nny)], dtype=int)

    def bottom_nodes(self) -> np.ndarray:
        """All nodes on the bottom edge (iy = 0)."""
        return np.array([self.node_index(ix, 0) for ix in range(self.nnx)], dtype=int)

    def top_nodes(self) -> np.ndarray:
        """All nodes on the top edge (iy = nely)."""
        return np.array([self.node_index(ix, self.nely) for ix in range(self.nnx)], dtype=int)

    # ------------------------------------------------------------------
    # Edge element arrays
    # ------------------------------------------------------------------

    def bottom_elems(self, thickness: int = 1) -> np.ndarray:
        """
        Element indices for the bottom *thickness* rows
        (ely = 0 .. thickness-1, all elx).
        """
        return np.array(
            [self.element_index(elx, ely)
             for ely in range(thickness)
             for elx in range(self.nelx)],
            dtype=int,
        )

    def top_elems(self, thickness: int = 1) -> np.ndarray:
        """
        Element indices for the top *thickness* rows
        (ely = nely-thickness .. nely-1, all elx).
        """
        return np.array(
            [self.element_index(elx, ely)
             for ely in range(self.nely - thickness, self.nely)
             for elx in range(self.nelx)],
            dtype=int,
        )

    def left_elems(self) -> np.ndarray:
        """Element indices for the left column (elx = 0)."""
        return np.array(
            [self.element_index(0, ely) for ely in range(self.nely)],
            dtype=int,
        )

    def right_elems(self) -> np.ndarray:
        """Element indices for the right column (elx = nelx-1)."""
        return np.array(
            [self.element_index(self.nelx - 1, ely) for ely in range(self.nely)],
            dtype=int,
        )

    # ------------------------------------------------------------------
    # DOF matrix
    # ------------------------------------------------------------------

    def generate_edofMat(self) -> np.ndarray:
        """
        Build and return the (nelxy, 8) element→DOF connectivity matrix.

        Row *el* holds the 8 global DOF indices for element *el*, in the
        order required by ``lk()`` in elements.py::

            [2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3, 2*n2, 2*n2+1, 2*n1, 2*n1+1]

        where  n1 = node_index(elx, ely),  n2 = node_index(elx+1, ely).
        """
        edofMat = np.zeros((self.nelxy, 8), dtype=int)
        for elx in range(self.nelx):
            for ely in range(self.nely):
                el = self.element_index(elx, ely)
                n1 = self.node_index(elx,     ely)
                n2 = self.node_index(elx + 1, ely)
                edofMat[el, :] = [
                    2*n1+2, 2*n1+3,
                    2*n2+2, 2*n2+3,
                    2*n2,   2*n2+1,
                    2*n1,   2*n1+1,
                ]
        return edofMat

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------

    def visualisation_reshape(self, field: np.ndarray) -> np.ndarray:
        """Reshape flat (nelxy,) element field to (nely, nelx) for imshow."""
        return field.reshape((self.nelx, self.nely)).T

    def visualisation_reshape_nodes(self, field: np.ndarray) -> np.ndarray:
        """Reshape flat (nnxy,) nodal field to (nny, nnx) for imshow."""
        return field.reshape((self.nnx, self.nny)).T


# ---------------------------------------------------------------------------
# BeamDomain
# ---------------------------------------------------------------------------

class BeamDomain(MeshDomain):
    """
    Structured beam domain with JSON-driven BCs, loads, and passive flanges.

    Implements the problem interface required by FESolver::

        nelx, nely, ndof         (inherited from MeshDomain)
        get_fixed_dofs()         assembled from _supports
        get_load_vector()        assembled from self-weight

    Exposes passive-region helpers for TopOptSolver::

        get_passive_elements()
        get_active_elements()
        has_passive_elements()
        correct_design_variable(rho, volfrac)

    Supported BC types (JSON ``bc.supports`` list)
    -----------------------------------------------
    ``{"type": "clamp",  "location": "left"|"right"|"bottom"|"top"}``
        All nodes on that edge, both DOFs fixed by default.

    ``{"type": "middle", "location": "left"|"right"|"bottom"|"top"}``
        Single midpoint node of that edge, both DOFs fixed by default.
        Use this for a knife-edge / pin support at the beam mid-height.

    ``{"type": "node", "rx": 0.5, "ry": 0.0}``
        Node nearest to relative domain position (rx, ry).

    All types accept an optional ``"constraint"`` key:
        ``"xy"``  fix both DOFs (default)
        ``"x"``   fix x-DOF only
        ``"y"``   fix y-DOF only

    Design improvements over TopOptPython fem.py
    ---------------------------------------------
    * **Supports**: stored as ``list[tuple[int, str]]`` — no fragile boolean
      mask arrays, no ``+=`` bug (numpy element-wise add vs. concatenate).
    * **Passive elements**: stored as ``set[int]`` — duplicates impossible,
      no ``np.concatenate`` chains.
    * **correct_design_variable**: counts passive area via ``len(_passive_set)``
      instead of ``nelx * 2 * thickness`` — correct for asymmetric flanges and
      single-flange configurations.
    * **get_fixed_dofs**: reads actual constraint type per node instead of
      always forcing "xy" regardless of the support definition.
    """

    def __init__(self, path_or_params: "str | dict" = "parameters.json"):
        if isinstance(path_or_params, str):
            with open(path_or_params, "r", encoding="utf-8") as fh:
                params = json.load(fh)
        else:
            params = path_or_params
        domain = params["domain"]
        mesh_p = domain["mesh"]
        size_p = domain["size"]

        super().__init__(
            nelx     = int(mesh_p["nelx"]),
            nely     = int(mesh_p["nely"]),
            length_x = float(size_p["length"]),
            length_y = float(size_p["height"]),
            gamma    = float(mesh_p.get("FEM_massmatrix_gamma", 1.4)),
        )

        # Physical material properties (read once, stored for downstream use)
        mat_cfg    = params.get("materials", {}).get("base", {})
        units      = params.get("meta", {}).get("units", {}).get("elastic_modulus", "")
        self.E     = float(mat_cfg.get("E", 205.0))
        if units == "GPa":
            self.E *= 1e9
        elif units == "MPa":
            self.E *= 1e6
        self.nu    = float(mat_cfg.get("nu", 0.3))
        self.rho   = float(mat_cfg.get("rho", 7850.0))

        # List of (node_id, constraint_type) with constraint_type ∈ {"x","y","xy"}
        self._supports: list[tuple[int, str]] = []

        # Set of passive element indices — set prevents duplicates automatically
        self._passive_set: set[int] = set()

        # Self-weight: per-node y force (negative = downward)
        self._selfweight_fy: float = 0.0

        # Concentrated point forces: list of {node_id, dof_offset, value}
        # dof_offset: 0 = x-DOF, 1 = y-DOF
        self._concentrated_forces: list[dict] = []

        # Modal (eigenvector-based) forces — populated by _append_modal_force
        self._modal_force_specs: list = []

        self._parse_passive_regions(domain.get("passive_regions", {}))
        self._parse_supports(params["bc"]["supports"])
        self._parse_loads(
            params["bc"]["loads"],
            params.get("materials", {}).get("base", {}),
        )
        self._parse_concentrated_forces(
            params["bc"]["loads"].get("concentrated_forces", [])
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_json(cls, path: str = "parameters.json") -> "BeamDomain":
        """Create a BeamDomain by reading *path* (a JSON parameters file)."""
        return cls(path)

    @classmethod
    def from_dict(cls, params: dict) -> "BeamDomain":
        """Create a BeamDomain from an already-loaded parameters dict.

        Used by the sweep runner so each sweep point avoids re-reading the
        JSON file and can pass an override-applied dict directly.
        """
        return cls(params)

    # ------------------------------------------------------------------
    # JSON parsing
    # ------------------------------------------------------------------

    def _parse_passive_regions(self, passive_cfg: dict) -> None:
        """Populate _passive_set from 'domain.passive_regions'."""
        flange = passive_cfg.get("flange", {})

        if flange.get("is_bottom_passive_flag", False):
            rel_h     = float(flange.get("bottom_relative_height", 0.0))
            thickness = max(1, int(round(rel_h * self.nely)))
            self._passive_set.update(self.bottom_elems(thickness).tolist())

        if flange.get("is_top_passive_flag", False):
            rel_h     = float(flange.get("top_relative_height", 0.0))
            thickness = max(1, int(round(rel_h * self.nely)))
            self._passive_set.update(self.top_elems(thickness).tolist())

    def _parse_supports(self, supports_cfg: list) -> None:
        """
        Populate _supports from 'bc.supports'.

        Each entry becomes one or more (node_id, constraint_type) tuples
        in _supports.  No boolean mask arrays are used.

        Supported type / location combinations
        --------------------------------------
        ``{"type": "clamp"|"clamped", "location": "left"|"right"|"bottom"|"top"}``
            All nodes on that edge, both DOFs fixed.

        ``{"type": "clamp"|"clamped",
           "location": "left_midpoint"|"right_midpoint"|"bottom_midpoint"|"top_midpoint"}``
            Single mid-height / mid-span node of that edge — useful for
            knife-edge / simply-supported conditions (SS, CS Olhoff cases).

        ``{"type": "middle", "location": "left"|"right"|"bottom"|"top"}``
            Alias: single midpoint node of the named edge.

        ``{"type": "node", "rx": float, "ry": float}``
            Node nearest to relative domain position (rx, ry) ∈ [0, 1]².

        All types accept an optional ``"constraint"`` key (default ``"xy"``):
            ``"xy"`` — fix both DOFs  |  ``"x"`` — x only  |  ``"y"`` — y only
        """
        _edge_nodes = {
            "left":   self.left_nodes,
            "right":  self.right_nodes,
            "bottom": self.bottom_nodes,
            "top":    self.top_nodes,
        }
        # Midpoint relative positions for each edge and their "_midpoint" aliases
        _midpoints = {
            "left":              (0.0, 0.5),
            "right":             (1.0, 0.5),
            "bottom":            (0.5, 0.0),
            "top":               (0.5, 1.0),
            "left_midpoint":     (0.0, 0.5),
            "right_midpoint":    (1.0, 0.5),
            "bottom_midpoint":   (0.5, 0.0),
            "top_midpoint":      (0.5, 1.0),
        }

        for entry in supports_cfg:
            stype      = entry.get("type",       "").lower()
            location   = entry.get("location",   "").lower()
            constraint = entry.get("constraint", "xy")

            if stype in ("clamp", "clamped"):
                if location in _edge_nodes:
                    # Full-edge clamp: fix every node on the named edge.
                    for n in _edge_nodes[location]():
                        self._supports.append((int(n), constraint))
                elif location in _midpoints:
                    # Midpoint-only pin: single node at mid-height / mid-span.
                    # Covers "left_midpoint", "right_midpoint", etc. as used
                    # in the Olhoff SS / CS benchmark cases.
                    rx, ry = _midpoints[location]
                    self._supports.append((self.node_at_relative(rx, ry), constraint))
                else:
                    raise ValueError(
                        f"Unknown clamp location: {location!r}. "
                        "Valid edge names: 'left', 'right', 'bottom', 'top'. "
                        "Valid midpoint names: 'left_midpoint', 'right_midpoint', "
                        "'bottom_midpoint', 'top_midpoint'."
                    )

            elif stype == "middle":
                if location not in _midpoints:
                    raise ValueError(
                        f"Unknown middle location: {location!r}. "
                        "Valid: 'left', 'right', 'bottom', 'top'."
                    )
                rx, ry = _midpoints[location]
                self._supports.append((self.node_at_relative(rx, ry), constraint))

            elif stype == "node":
                rx = float(entry.get("rx", 0.0))
                ry = float(entry.get("ry", 0.0))
                self._supports.append((self.node_at_relative(rx, ry), constraint))

            else:
                raise ValueError(
                    f"Unknown support type: {stype!r}. "
                    "Valid types: 'clamp', 'clamped', 'middle', 'node'."
                )

    def _parse_loads(self, loads_cfg: dict, material_cfg: dict) -> None:
        """Set up self-weight and concentrated forces from 'bc.loads' and 'materials.base'."""
        if loads_cfg.get("is_selfweight", True):
            rho     = float(material_cfg.get("rho", 7850.0))
            gravity = float(loads_cfg.get("gravity", 9.81))
            # Gravitational force on one element distributed uniformly to all
            # nodes — same simplification as TopOptPython deadload().
            self._selfweight_fy = -(self.volume_element * rho * gravity)

    def _parse_concentrated_forces(self, forces_cfg: list) -> None:
        """
        Populate _concentrated_forces from 'bc.loads.concentrated_forces'.

        Each JSON entry has the form::

            {
                "edge": "top" | "bottom" | "left" | "right",
                "relative_horizontal_position": float,   // top / bottom edges
                "relative_vertical_position":   float,   // left / right edges
                "direction": "x" | "y",
                "value": float    // force in Newtons  (negative = downward/leftward)
            }

        Multiple forces at the same node accumulate (+=).

        Design notes vs TopOptPython add_top_concentrated_forces
        ---------------------------------------------------------
        * Generalised to all four edges, not just the top.
        * Node position computed directly via node_index() — no boolean
          mask arrays or += list bug.
        * Stored as plain dicts; applied in get_load_vector().
        """
        for entry in forces_cfg:
            if entry.get("type", "point").lower() == "modal":
                self._append_modal_force(entry)
                continue

            edge      = entry.get("edge", "top").lower()
            direction = entry.get("direction", "y").lower()
            value     = float(entry.get("value", 0.0))

            if edge in ("top", "bottom"):
                rx  = float(entry.get("relative_horizontal_position", 0.5))
                rx  = max(0.0, min(1.0, rx))
                ix  = max(0, min(self.nelx, int(rx * self.nelx)))
                iy  = self.nely if edge == "top" else 0
            elif edge in ("left", "right"):
                ry  = float(entry.get("relative_vertical_position", 0.5))
                ry  = max(0.0, min(1.0, ry))
                iy  = max(0, min(self.nely, int(ry * self.nely)))
                ix  = 0 if edge == "left" else self.nelx
            else:
                raise ValueError(
                    f"Unknown concentrated force edge: {edge!r}. "
                    "Valid: 'top', 'bottom', 'left', 'right'."
                )

            if direction not in ("x", "y"):
                raise ValueError(
                    f"Unknown force direction: {direction!r}. Valid: 'x', 'y'."
                )

            self._concentrated_forces.append({
                "node_id":    self.node_index(ix, iy),
                "dof_offset": 0 if direction == "x" else 1,
                "value":      value,
            })

    # ------------------------------------------------------------------
    # Modal force helpers  (consumed by ModalFESolver)
    # ------------------------------------------------------------------

    def _append_modal_force(self, entry: dict) -> None:
        """Parse a ``{"type": "modal", "number_of_modes": N}`` JSON entry
        and append a ``ModalForceSpec`` to ``_modal_force_specs``.

        The import is deferred to avoid a circular-import issue between
        domain.py and modal_load.py.
        """
        from .modal_load import ModalForceSpec     # local import — intentional
        mode_index = int(entry.get("number_of_modes", 1))
        self._modal_force_specs.append(ModalForceSpec(mode_index=mode_index))

    def has_modal_forces(self) -> bool:
        """Return True when at least one modal force spec is registered."""
        return len(self._modal_force_specs) > 0

    def get_modal_force_specs(self) -> list:
        """Return a shallow copy of the list of ModalForceSpec objects."""
        return list(self._modal_force_specs)

    # ------------------------------------------------------------------
    # Problem interface  (required by FESolver and TopOptSolver)
    # ------------------------------------------------------------------

    def get_fixed_dofs(self) -> np.ndarray:
        """
        Assemble sorted array of constrained DOF indices.

        For each entry (node_id, ctype) in _supports:
            "x"  → fixes DOF 2*node_id
            "y"  → fixes DOF 2*node_id + 1
            "xy" → fixes both
        """
        fixed: set[int] = set()
        for node_id, ctype in self._supports:
            if "x" in ctype:
                fixed.add(2 * node_id)
            if "y" in ctype:
                fixed.add(2 * node_id + 1)
        return np.array(sorted(fixed), dtype=int)

    def get_load_vector(self, constant: bool | None = None) -> np.ndarray:
        """
        Assemble global load vector, shape (ndof, 1).

        Parameters
        ----------
        constant : bool or None
            True  — concentrated forces only.
            False — self-weight only.
            None  — combined (default, used by FESolver).
        """
        f = np.zeros((self.ndof, 1))
        if constant is not True:
            if self._selfweight_fy != 0.0:
                f += self.deadload()
        if constant is not False:
            for force in self._concentrated_forces:
                dof = 2 * force["node_id"] + force["dof_offset"]
                f[dof, 0] += force["value"]
        return f

    # ------------------------------------------------------------------
    # Passive-region interface  (used by TopOptSolver.run)
    # ------------------------------------------------------------------

    def get_passive_elements(self) -> np.ndarray:
        """Sorted array of passive (flange) element indices."""
        return np.array(sorted(self._passive_set), dtype=int)

    def get_active_elements(self) -> np.ndarray:
        """Sorted array of element indices that the optimiser may change."""
        return np.setdiff1d(np.arange(self.nelxy), self.get_passive_elements())

    def has_passive_elements(self) -> bool:
        return len(self._passive_set) > 0

    # ------------------------------------------------------------------
    # Volume-fraction correction for passive regions
    # ------------------------------------------------------------------

    def correct_design_variable(self, rho: np.ndarray, volfrac: float) -> np.ndarray:
        """
        Rescale active densities to satisfy the overall volume-fraction target,
        then force all passive (flange) elements to density = 1.

        Volume balance::

            volfrac * nelxy  =  Σ_active(rho)  +  n_passive * 1.0

        Required active volume fraction::

            volfrac_active  =  (volfrac * nelxy  −  n_passive) / n_active

        Fixes vs. original fem.py
        -------------------------
        * Uses ``len(_passive_set)`` instead of ``nelx * 2 * thickness`` —
          correct for asymmetric flanges and single-flange configurations.
        * Rescales only ``rho[active]`` — passive slots are never touched by
          the rescaling step.
        * Clips ``rho[active]`` to [0, 1] after rescaling.

        Parameters
        ----------
        rho : ndarray, shape (nelxy,)
            Current density field, modified in-place.
        volfrac : float
            Overall target volume fraction ∈ (0, 1).

        Returns
        -------
        rho : ndarray
        """
        if not self.has_passive_elements():
            return rho

        passive   = self.get_passive_elements()
        active    = self.get_active_elements()
        n_passive = len(passive)
        n_active  = len(active)

        volfrac_active = (volfrac * self.nelxy - n_passive) / n_active

        if not (0.0 <= volfrac_active <= 1.0):
            raise ValueError(
                f"Infeasible passive configuration: {n_passive} passive elements "
                f"out of {self.nelxy} total. Overall volfrac={volfrac:.3f} requires "
                f"active volfrac={volfrac_active:.3f}, which is outside [0, 1]."
            )

        active_sum = float(rho[active].sum())
        if active_sum > 1e-12:
            rho[active] = np.clip(
                rho[active] * (volfrac_active * n_active / active_sum),
                0.0, 1.0,
            )

        rho[passive] = 1.0
        return rho

    # ------------------------------------------------------------------
    # Load vectors  (ported from TopOptPython fem.py BeamDomain)
    # ------------------------------------------------------------------


    def convert_elemental_field_to_nodal_field(self, field: np.ndarray) -> np.ndarray:
        """
        Map a flat element field (nelxy,) to a nodal DOF field (ndof, 1).

        Each node collects 1/4 of the value from each of its (up to 4) adjacent
        elements — a bilinear averaging stencil.  Boundary nodes naturally receive
        contributions from fewer elements; the pad with zeros handles them without
        special-casing.

        After averaging the result is *sharpened* (raised to the power ``gamma``)
        and *renormalized* to preserve the original field sum.  This compresses
        near-zero values towards zero while keeping the total load constant, which
        prevents void regions from carrying spurious inertial loads.

        The output is duplicated across x- and y-DOF channels so it can be
        multiplied element-wise with a (ndof, n_columns) force matrix.

        Parameters
        ----------
        field : ndarray, shape (nelxy,)
            Element density (or any element-wise scalar field).

        Returns
        -------
        nodal : ndarray, shape (ndof, 1)
            Nodal field ready for element-wise multiplication with a load matrix.
        """
        S_orig = float(field.sum())

        # Pad the 2-D element grid by one row/column of zeros on all sides so
        # that boundary nodes (which touch fewer than 4 elements) are handled
        # identically to interior nodes.
        padded = np.pad(
            self.visualisation_reshape(field),  # (nely, nelx)
            pad_width=1,
            mode="constant",
            constant_values=0.0,
        )

        # Each node (i, j) averages its four adjacent element corners.
        # After padding, element (i, j) of the padded array corresponds to
        # original element (i-1, j-1), so:
        nodal_2d = (
            padded[:-1, :-1] +   # bottom-left  element
            padded[:-1, 1:]  +   # bottom-right element
            padded[1:,  :-1] +   # top-left     element
            padded[1:,  1:]      # top-right    element
        ) / 4.0                  # shape: (nely+1, nelx+1) = (nny, nnx)

        # Flatten in Fortran (column-major) order to match node_index convention,
        # then duplicate for x- and y-DOF at each node.
        nodal_flat = nodal_2d.ravel(order="F")                # (nnxy,)
        nodal      = np.repeat(nodal_flat, 2).reshape(-1, 1)  # (ndof, 1)

        # Sharpening: raise to gamma to suppress near-zero values
        nodal_sharp = nodal ** self.gamma

        # Renormalize to preserve sum of the original field
        S_new = float(nodal_sharp.sum())
        if S_new > 1e-12:
            nodal_sharp *= S_orig / S_new

        return nodal_sharp

    def deadload(self) -> np.ndarray:
        """
        Constant (density-independent) self-weight load vector, shape (ndof, 1).

        Applies ``_selfweight_fy`` uniformly to every y-DOF.  The x-component
        is always zero (gravity acts in the −y direction only).

        Used directly by ``get_load_vector`` and as the base vector scaled by
        ``dynamic_f`` during frequency optimisation.
        """
        f = np.zeros((self.ndof, 1))
        f[1::2, 0] = self._selfweight_fy
        return f

    def dynamic_f(self, field: np.ndarray) -> np.ndarray:
        """
        Density-weighted load vector for frequency / dynamic analysis,
        shape (ndof, 1).

        Scales the static self-weight load by the local element density so that
        void regions carry no inertial load.  This is the correct body-force
        assembly for frequency topology optimisation where the mass varies with
        the design field.

        Algorithm (after TopOptPython fem.py ``dynamic_f``)::

            nodal_rho = convert_elemental_field_to_nodal_field(field)
            f_dynamic = nodal_rho * deadload()

        Parameters
        ----------
        field : ndarray, shape (nelxy,)
            Physical element density (0 = void, 1 = solid).

        Returns
        -------
        f : ndarray, shape (ndof, 1)
            Density-weighted load vector.

        Raises
        ------
        RuntimeError
            If self-weight is not configured (``_selfweight_fy == 0``).
        """
        if self._selfweight_fy == 0.0:
            raise RuntimeError(
                "dynamic_f requires self-weight. "
                "Set 'bc.loads.is_selfweight: true' in the JSON."
            )
        nodal_rho = self.convert_elemental_field_to_nodal_field(field)
        return nodal_rho * self.deadload()
