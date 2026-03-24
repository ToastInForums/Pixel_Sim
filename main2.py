"""
Optimised Pixel Simulation
===========================
Fix (vs previous optimised version)
--------------------------------------
BUG: Cells that could not move would set active_next[r,c]=True but never
     called _mark_chunk(), so their chunk was not dirty next frame.
     The collect loop skips cells whose chunk is clean → the cell was never
     processed → water/sand/lava appeared to freeze mid-air.

FIX: Every `active_next[r, c] = True` now has a paired
     `_mark_chunk(dirty_next, r, c, CH, CW)`.
"""

import moderngl
import moderngl_window as mglw
import numpy as np
import numba as nb

# ============================================================
# WINDOW / GRID CONSTANTS
# ============================================================
WINDOW_WIDTH  = 1280
WINDOW_HEIGHT = 800
GRID_WIDTH    = 320
GRID_HEIGHT   = 200

CHUNK_SIZE = 8
CHUNK_W    = GRID_WIDTH  // CHUNK_SIZE   # 40
CHUNK_H    = GRID_HEIGHT // CHUNK_SIZE   # 25

# ============================================================
# COLOR PALETTE  (R, G, B)
# ============================================================
EMPTY    = ( 2,  2,  8)
SAND     = (220, 190, 120)
WATER    = ( 40, 100, 220)
STONE    = (120, 120, 130)
FIRE     = (240,  80,  20)
LAVA     = (207,  70,  20)
OBSIDIAN = ( 30,  20,  45)
STEAM    = (180, 200, 220)
SMOKE    = ( 60,  60,  70)
OIL      = (180, 140, 60)

ELEMENTS: dict = {}

class Element:
    def __init__(self, name, density, move, color, viscosity=0.0):
        self.name      = name
        self.density   = density
        self.viscosity = viscosity
        self.move      = move
        self.color     = color
        eid = len(ELEMENTS)
        ELEMENTS[eid] = dict(name=name, density=density,
                             viscosity=viscosity, move=move, color=color)

Sand     = Element("Sand",     1.6, 1, SAND)
Water    = Element("Water",    1.0, 2, WATER,    viscosity=0.05)
Stone    = Element("Stone",    2.5, 0, STONE)
Lava     = Element("Lava",     2.2, 2, LAVA,     viscosity=0.75)
Obsidian = Element("Obsidian", 3.5, 1, OBSIDIAN)
Fire     = Element("Fire",     0.0, 4, FIRE)
Steam    = Element("Steam",    0.0, 3, STEAM)
Smoke    = Element("Smoke",    0.0, 3, SMOKE)
Oil      = Element("Oil",      0.85, 2, OIL,     viscosity=0.02)

VERT = """
#version 430
in  vec2 in_pos;
out vec2 uv;
void main() {
    uv = in_pos * 0.5 + 0.5;
    uv.y = 1.0 - uv.y;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

FRAG = """
#version 430
uniform sampler2D grid_tex;
in  vec2 uv;
out vec4 frag_color;
void main() {
    frag_color = vec4(texture(grid_tex, uv).rgb, 1.0);
}
"""

# ============================================================
# NUMBA KERNELS
# ============================================================

@nb.njit(cache=True, fastmath=True, inline='always')
def _mark_chunk(dirty_next, r, c, CH, CW):
    cr = r >> 3
    cc = c >> 3
    r0 = max(0, cr - 1);  r1 = min(CH - 1, cr + 1)
    c0 = max(0, cc - 1);  c1 = min(CW - 1, cc + 1)
    for nr in range(r0, r1 + 1):
        for nc in range(c0, c1 + 1):
            dirty_next[nr, nc] = True


@nb.njit(cache=True, fastmath=True, inline='always')
def _activate(active_next, r, c, H, W):
    r0 = max(0, r - 1);  r1 = min(H - 1, r + 1)
    c0 = max(0, c - 1);  c1 = min(W - 1, c + 1)
    for nr in range(r0, r1 + 1):
        for nc in range(c0, c1 + 1):
            active_next[nr, nc] = True


@nb.njit(cache=True, fastmath=True, inline='always')
def _can_displace(mover, target, density_lut):
    return target == 0 or density_lut[mover] > density_lut[target]


@nb.njit(cache=True, fastmath=True, inline='always')
def _keep_active(active_next, dirty_next, r, c, H, W, CH, CW):
    """
    Re-queue a cell for next frame AND ensure its chunk stays dirty.
    This is the paired helper that was missing before – always call this
    instead of writing active_next[r,c]=True directly.
    """
    active_next[r, c] = True
    _mark_chunk(dirty_next, r, c, CH, CW)


@nb.njit(cache=True, fastmath=True, inline='always')
def _swap(grid, moved, active_next, dirty_next,
          r1, c1, r2, c2, H, W, CH, CW):
    tmp = grid[r1, c1]
    grid[r1, c1] = grid[r2, c2]
    grid[r2, c2] = tmp
    moved[r2, c2] = True
    _activate(active_next, r1, c1, H, W)
    _activate(active_next, r2, c2, H, W)
    _mark_chunk(dirty_next, r1, c1, CH, CW)
    _mark_chunk(dirty_next, r2, c2, CH, CW)


@nb.njit(cache=True, fastmath=True, inline='always')
def _swap_gas(grid, moved, active_next, dirty_next, metadata,
              r1, c1, r2, c2, H, W, CH, CW):
    metadata[r2, c2] = metadata[r1, c1]
    metadata[r1, c1] = 0
    _swap(grid, moved, active_next, dirty_next, r1, c1, r2, c2, H, W, CH, CW)


@nb.njit(cache=True, fastmath=True)
def _react(grid, metadata, active_next, dirty_next,
           r, c, obs_id, stm_id, wat_id, H, W, CH, CW):
    if r < H - 1 and grid[r + 1, c] == 0:
        return

    for di in range(4):
        if   di == 0: nr, nc = r - 1, c
        elif di == 1: nr, nc = r + 1, c
        elif di == 2: nr, nc = r,     c - 1
        else:         nr, nc = r,     c + 1

        if not (0 <= nr < H and 0 <= nc < W):
            continue
        if grid[nr, nc] != wat_id:
            continue

        grid[r, c]     = obs_id
        metadata[r, c] = 10

        grid[nr, nc]     = stm_id
        metadata[nr, nc] = 235 + np.random.randint(0, 20)
        active_next[nr, nc] = True
        _mark_chunk(dirty_next, nr, nc, CH, CW)

        burst = 12 + np.random.randint(0, 6)
        for _ in range(burst):
            sr = nr - np.random.randint(0, 9)
            sc = nc + np.random.randint(-3, 4)
            if 0 <= sr < H and 0 <= sc < W and grid[sr, sc] == 0:
                grid[sr, sc]     = stm_id
                metadata[sr, sc] = 228 + np.random.randint(0, 27)
                active_next[sr, sc] = True
                _mark_chunk(dirty_next, sr, sc, CH, CW)

        active_next[r, c] = True
        _mark_chunk(dirty_next, r, c, CH, CW)
        return


@nb.njit(cache=True, fastmath=True)
def _steam_neighbor_count(grid, r, c, steam_id, H, W):
    count = 0
    r0 = max(0, r - 1);  r1 = min(H - 1, r + 1)
    c0 = max(0, c - 1);  c1 = min(W - 1, c + 1)
    for nr in range(r0, r1 + 1):
        for nc in range(c0, c1 + 1):
            if grid[nr, nc] == steam_id:
                count += 1
    return count - 1


@nb.njit(cache=True, fastmath=True)
def _steam_patch_bias(grid, r, c, steam_id, H, W):
    ls = 0;  rs = 0
    for dr in range(-1, 2):
        nr = r + dr
        if not (0 <= nr < H):
            continue
        if c - 1 >= 0 and grid[nr, c - 1] == steam_id: ls += 3
        if c - 2 >= 0 and grid[nr, c - 2] == steam_id: ls += 2
        if c - 3 >= 0 and grid[nr, c - 3] == steam_id: ls += 1
        if c + 1 <  W and grid[nr, c + 1] == steam_id: rs += 3
        if c + 2 <  W and grid[nr, c + 2] == steam_id: rs += 2
        if c + 3 <  W and grid[nr, c + 3] == steam_id: rs += 1
    if ls > rs + 1: return -1
    if rs > ls + 1: return  1
    return 0


@nb.njit(cache=True, fastmath=True)
def _obsidian_supported(grid, metadata, r, c, H, W, move_lut):
    if metadata[r, c] > 0:
        metadata[r, c] -= 1
        return True
    for i in range(5):
        if   i == 0: nr, nc = r + 1, c
        elif i == 1: nr, nc = r + 1, c - 1
        elif i == 2: nr, nc = r + 1, c + 1
        elif i == 3: nr, nc = r,     c - 1
        else:        nr, nc = r,     c + 1
        if not (0 <= nr < H and 0 <= nc < W):
            continue
        t = grid[nr, nc]
        if t != 0 and move_lut[t] != 3:
            return True
    return False

@nb.njit(cache=True, fastmath=True)
def _react_oil_fire(grid, metadata, active_next, dirty_next, 
                    r, c, oil_id, fire_id, H, W, CH, CW):
    #Oil touching fire ignites: oil cell becomes fire
    for di in range(4):
        if di == 0: nr, nc = r - 1, c
        elif di == 1: nr, nc = r + 1, c
        elif di == 2: nr, nc = r, c - 1
        else: nr, nc = r, c + 1

        if not (0 <= nr < H and 0 <= nc < W): continue
        if grid[nr, nc] != fire_id: continue

        grid[r, c] = fire_id
        metadata[r, c] = 80 + np.random.randint(0, 80)
        active_next[r, c] = True
        _mark_chunk(dirty_next, r, c, CH, CW)
        return

# ============================================================
# MAIN SIMULATION KERNEL
# ============================================================
@nb.njit(cache=True, fastmath=True)
def simulate_step(grid, active, active_next,
                  metadata, moved,
                  dirty, dirty_next,
                  density_lut, move_lut, viscosity_lut,
                  steam_id, water_id, lava_id, obs_id,
                  oil_id, fire_id, smoke_id, 
                  idx_buf):

    H,  W  = grid.shape
    CH, CW = dirty.shape

    active_next[:] = False
    dirty_next[:]  = False
    moved[:]       = False

    # ------------------------------------------------------------------
    # Collect active cells inside dirty chunks
    # ------------------------------------------------------------------
    n = 0
    for cr in range(CH):
        row_dirty = False
        for cc2 in range(CW):
            if dirty[cr, cc2]:
                row_dirty = True
                break
        if not row_dirty:
            continue

        for cc in range(CW):
            if not dirty[cr, cc]:
                continue
            r_start = cr * CHUNK_SIZE
            c_start = cc * CHUNK_SIZE
            for r in range(r_start, r_start + CHUNK_SIZE):
                for c in range(c_start, c_start + CHUNK_SIZE):
                    if active[r, c] and grid[r, c] != 0:
                        idx_buf[n] = r * W + c
                        n += 1

    # Fisher-Yates shuffle
    for i in range(n - 1, 0, -1):
        j = np.random.randint(0, i + 1)
        tmp        = idx_buf[i]
        idx_buf[i] = idx_buf[j]
        idx_buf[j] = tmp

    # ------------------------------------------------------------------
    # Per-cell update
    # ------------------------------------------------------------------
    for i in range(n):
        idx  = idx_buf[i]
        r    = idx // W
        c    = idx - r * W

        cell = grid[r, c]
        if cell == 0 or moved[r, c]:
            continue

        if cell == lava_id and r < H - 1:
            _react(grid, metadata, active_next, dirty_next,
                   r, c, obs_id, steam_id, water_id, H, W, CH, CW)
            cell = grid[r, c]
            if cell == 0:
                continue
        
        if cell == oil_id:
            _react_oil_fire(grid, metadata, active_next, dirty_next, 
            r, c, oil_id, fire_id, H, W, CH, CW)
            cell = grid[r, c] # may have changed to fire
            if cell == 0:
                continue

        move = move_lut[cell]

        # ---- POWDER  (move == 1) ----
        if move == 1:
            if r >= H - 1:
                _keep_active(active_next, dirty_next, r, c, H, W, CH, CW)
                continue

            if cell == obs_id:
                if _obsidian_supported(grid, metadata, r, c, H, W, move_lut):
                    _keep_active(active_next, dirty_next, r, c, H, W, CH, CW)
                    continue
                if grid[r + 1, c] == 0:
                    _swap(grid, moved, active_next, dirty_next,
                          r, c, r + 1, c, H, W, CH, CW)
                else:
                    if np.random.randint(0, 2) != 0:
                        dx1 = -1;  dx2 = 1
                    else:
                        dx1 =  1;  dx2 = -1
                    nc1 = c + dx1;  nc2 = c + dx2
                    if 0 <= nc1 < W and grid[r + 1, nc1] == 0:
                        _swap(grid, moved, active_next, dirty_next,
                              r, c, r + 1, nc1, H, W, CH, CW)
                    elif 0 <= nc2 < W and grid[r + 1, nc2] == 0:
                        _swap(grid, moved, active_next, dirty_next,
                              r, c, r + 1, nc2, H, W, CH, CW)
                    else:
                        _keep_active(active_next, dirty_next, r, c, H, W, CH, CW)
            else:
                if _can_displace(cell, grid[r + 1, c], density_lut):
                    _swap(grid, moved, active_next, dirty_next,
                          r, c, r + 1, c, H, W, CH, CW)
                else:
                    if np.random.randint(0, 2) != 0:
                        dx1 = -1;  dx2 = 1
                    else:
                        dx1 =  1;  dx2 = -1
                    nc1 = c + dx1;  nc2 = c + dx2
                    if 0 <= nc1 < W and _can_displace(cell, grid[r + 1, nc1], density_lut):
                        _swap(grid, moved, active_next, dirty_next,
                              r, c, r + 1, nc1, H, W, CH, CW)
                    elif 0 <= nc2 < W and _can_displace(cell, grid[r + 1, nc2], density_lut):
                        _swap(grid, moved, active_next, dirty_next,
                              r, c, r + 1, nc2, H, W, CH, CW)
                    else:
                        _keep_active(active_next, dirty_next, r, c, H, W, CH, CW)

        # ---- LIQUID  (move == 2) ----
        elif move == 2:
            if r >= H - 1:
                _keep_active(active_next, dirty_next, r, c, H, W, CH, CW)
                continue

            if np.random.random() < viscosity_lut[cell]:
                _keep_active(active_next, dirty_next, r, c, H, W, CH, CW)
                continue

            if _can_displace(cell, grid[r + 1, c], density_lut):
                _swap(grid, moved, active_next, dirty_next,
                      r, c, r + 1, c, H, W, CH, CW)
            else:
                if np.random.randint(0, 2) != 0:
                    dx1 = -1;  dx2 = 1
                else:
                    dx1 =  1;  dx2 = -1
                nc1 = c + dx1;  nc2 = c + dx2

                moved_liq = False
                if 0 <= nc1 < W and _can_displace(cell, grid[r + 1, nc1], density_lut):
                    _swap(grid, moved, active_next, dirty_next,
                          r, c, r + 1, nc1, H, W, CH, CW)
                    moved_liq = True
                elif 0 <= nc2 < W and _can_displace(cell, grid[r + 1, nc2], density_lut):
                    _swap(grid, moved, active_next, dirty_next,
                          r, c, r + 1, nc2, H, W, CH, CW)
                    moved_liq = True

                if not moved_liq:
                    if 0 <= nc1 < W and grid[r, nc1] == 0:
                        _swap(grid, moved, active_next, dirty_next,
                              r, c, r, nc1, H, W, CH, CW)
                        moved_liq = True
                    elif 0 <= nc2 < W and grid[r, nc2] == 0:
                        _swap(grid, moved, active_next, dirty_next,
                              r, c, r, nc2, H, W, CH, CW)
                        moved_liq = True

                if not moved_liq:
                    _keep_active(active_next, dirty_next, r, c, H, W, CH, CW)

        # ---- GAS  (move == 3) ----
        elif move == 3:
            is_steam  = (cell == steam_id)
            near_ceil = r < 16
            very_top  = r < 6
            above     = r - 1

            life = metadata[r, c]
            if is_steam:
                decay_p = 0.05 if very_top else (0.10 if near_ceil else 0.40)
            else:
                decay_p = 0.70

            if np.random.random() < decay_p:
                life -= 1

            if life <= 0:
                grid[r, c] = 0
                _mark_chunk(dirty_next, r, c, CH, CW)
                continue

            metadata[r, c] = life

            local_steam = 0
            patch_bias  = 0
            if is_steam and near_ceil:
                local_steam = _steam_neighbor_count(grid, r, c, steam_id, H, W)
                patch_bias  = _steam_patch_bias(grid, r, c, steam_id, H, W)

                if very_top and life <= 80 and local_steam >= 4:
                    age_f   = (80 - life) / 80.0
                    dense_f = min(local_steam - 3, 4) / 4.0
                    cchance = 0.0006 + 0.0035 * dense_f + 0.010 * age_f
                    if r < 3:
                        cchance += 0.0015
                    if np.random.random() < cchance:
                        grid[r, c]     = water_id
                        metadata[r, c] = 0
                        _activate(active_next, r, c, H, W)
                        _mark_chunk(dirty_next, r, c, CH, CW)
                        continue

                hold = 0.16 + 0.06 * min(local_steam, 5)
                if very_top:
                    hold += 0.10
                if local_steam >= 3 and np.random.random() < hold:
                    _keep_active(active_next, dirty_next, r, c, H, W, CH, CW)
                    continue

            moved_gas = False

            if is_steam and near_ceil and patch_bias != 0:
                nc = c + patch_bias
                if 0 <= nc < W and grid[r, nc] == 0:
                    _swap_gas(grid, moved, active_next, dirty_next, metadata,
                              r, c, r, nc, H, W, CH, CW)
                    moved_gas = True

            if not moved_gas and above >= 0 and grid[above, c] == 0:
                if (not near_ceil) or np.random.random() < 0.72:
                    _swap_gas(grid, moved, active_next, dirty_next, metadata,
                              r, c, above, c, H, W, CH, CW)
                    moved_gas = True

            if np.random.randint(0, 2) != 0:
                dx1 = -1;  dx2 = 1
            else:
                dx1 =  1;  dx2 = -1
            if patch_bias == -1: dx1 = -1;  dx2 = 1
            elif patch_bias == 1: dx1 = 1;  dx2 = -1

            if not moved_gas and above >= 0:
                nc = c + dx1
                if 0 <= nc < W and grid[above, nc] == 0:
                    _swap_gas(grid, moved, active_next, dirty_next, metadata,
                              r, c, above, nc, H, W, CH, CW)
                    moved_gas = True
                else:
                    nc = c + dx2
                    if 0 <= nc < W and grid[above, nc] == 0:
                        _swap_gas(grid, moved, active_next, dirty_next, metadata,
                                  r, c, above, nc, H, W, CH, CW)
                        moved_gas = True

            if not moved_gas:
                max_spread = 2 if is_steam else 1
                for dist in range(1, max_spread + 1):
                    nc = c + dx1 * dist
                    if 0 <= nc < W and grid[r, nc] == 0:
                        _swap_gas(grid, moved, active_next, dirty_next, metadata,
                                  r, c, r, nc, H, W, CH, CW)
                        moved_gas = True
                        break
                    nc = c + dx2 * dist
                    if 0 <= nc < W and grid[r, nc] == 0:
                        _swap_gas(grid, moved, active_next, dirty_next, metadata,
                                  r, c, r, nc, H, W, CH, CW)
                        moved_gas = True
                        break

            if not moved_gas:
                _keep_active(active_next, dirty_next, r, c, H, W, CH, CW)

        # ---- FIRE  (move == 4) ----
        elif move == 4:
            life = metadata[r, c]

            # 1. Water check - extinguish first
            extinguished = False
            for di in range(4):
                if   di == 0: nr, nc = r - 1, c
                elif di == 1: nr, nc = r + 1, c
                elif di == 2: nr, nc = r,     c - 1
                else:         nr, nc = r,     c + 1
                if not (0 <= nr < H and 0 <= nc < W): continue
                if grid[nr, nc] == water_id:
                    grid[r, c]       = 0
                    grid[nr, nc]     = steam_id
                    metadata[nr, nc] = 200 + np.random.randint(0, 55)
                    _activate(active_next, r, c, H, W)
                    _mark_chunk(dirty_next, r, c, CH, CW)
                    _mark_chunk(dirty_next, nr, nc, CH, CW)
                    extinguished = True
                    break
            if extinguished:
                continue

            # 2. Lifetime decay — always decrement, fire burns fast
            life -= 1
            if life <= 0:
                # Dying fire becomes smoke
                grid[r, c]    = smoke_id
                metadata[r,c] = 40 + np.random.randint(0, 40)
                _activate(active_next, r, c, H, W)
                _mark_chunk(dirty_next, r, c, CH, CW)
                continue
            metadata[r, c] = life

            # 3. Spread to flammable neighbours BEFORE moving
            for di in range(4):
                if   di == 0: nr, nc = r - 1, c
                elif di == 1: nr, nc = r,     c - 1
                elif di == 2: nr, nc = r,     c + 1
                else:         nr, nc = r + 1, c
                if not (0 <= nr < H and 0 <= nc < W): continue
                if grid[nr, nc] == oil_id and np.random.random() < 0.15:
                    grid[nr, nc]     = fire_id
                    metadata[nr, nc] = 80 + np.random.randint(0, 80)
                    _activate(active_next, nr, nc, H, W)
                    _mark_chunk(dirty_next, nr, nc, CH, CW)

            # 4. Movement — fire rises upward like a flame
            above = r - 1
            moved_fire = False

            # Try straight up
            if above >= 0 and grid[above, c] == 0:
                _swap(grid, moved, active_next, dirty_next,
                      r, c, above, c, H, W, CH, CW)
                moved_fire = True

            # Try diagonal up (gives the flickering sideways lean)
            if not moved_fire and above >= 0:
                if np.random.randint(0, 2) != 0:
                    dx1, dx2 = -1, 1
                else:
                    dx1, dx2 =  1, -1
                nc1 = c + dx1
                if 0 <= nc1 < W and grid[above, nc1] == 0:
                    _swap(grid, moved, active_next, dirty_next,
                          r, c, above, nc1, H, W, CH, CW)
                    moved_fire = True
                else:
                    nc2 = c + dx2
                    if 0 <= nc2 < W and grid[above, nc2] == 0:
                        _swap(grid, moved, active_next, dirty_next,
                              r, c, above, nc2, H, W, CH, CW)
                        moved_fire = True

            # Blocked from rising — spread sideways on the ground
            if not moved_fire:
                if np.random.randint(0, 2) != 0:
                    dx1, dx2 = -1, 1
                else:
                    dx1, dx2 =  1, -1
                nc1 = c + dx1
                if 0 <= nc1 < W and grid[r, nc1] == 0:
                    _swap(grid, moved, active_next, dirty_next,
                          r, c, r, nc1, H, W, CH, CW)
                    moved_fire = True
                elif 0 <= c + dx2 < W and grid[r, c + dx2] == 0:
                    _swap(grid, moved, active_next, dirty_next,
                          r, c, r, c + dx2, H, W, CH, CW)
                    moved_fire = True

            if not moved_fire:
                _keep_active(active_next, dirty_next, r, c, H, W, CH, CW)
    return n


# ============================================================
# SIMULATION WINDOW
# ============================================================
class Sim(mglw.WindowConfig):
    gl_version  = (4, 3)
    title       = "Pixel Sim – optimised"
    window_size = (WINDOW_WIDTH, WINDOW_HEIGHT)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.sim_speed   = 1/30   # run sim at 30 ticks/sec — lower = slower
        self._sim_accum  = 0.0    # existing lines below this unchanged

        self.grid        = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.uint8)
        self.active      = np.ones ((GRID_HEIGHT, GRID_WIDTH), dtype=bool)
        self.active_next = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=bool)
        self.metadata    = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.uint8)
        self.moved       = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=bool)

        self.dirty      = np.ones ((CHUNK_H, CHUNK_W), dtype=bool)
        self.dirty_next = np.zeros((CHUNK_H, CHUNK_W), dtype=bool)

        self._idx_buf = np.empty(GRID_HEIGHT * GRID_WIDTH, dtype=np.int32)

        self.selected  = 1
        self.brush_r   = 2
        self.sim_steps = 1

        lut_size = len(ELEMENTS) + 1
        self.color_lut     = np.zeros((lut_size, 3), dtype=np.uint8)
        self.density_lut   = np.zeros(lut_size,      dtype=np.float32)
        self.move_lut      = np.zeros(lut_size,      dtype=np.uint8)
        self.viscosity_lut = np.zeros(lut_size,      dtype=np.float32)
        self.color_lut[0]  = EMPTY
        for eid, data in ELEMENTS.items():
            idx = eid + 1
            self.color_lut[idx]     = data["color"]
            self.density_lut[idx]   = data["density"]
            self.move_lut[idx]      = data["move"]
            self.viscosity_lut[idx] = data["viscosity"]

        ids = {data["name"]: eid + 1 for eid, data in ELEMENTS.items()}
        self._steam_id = ids["Steam"]
        self._water_id = ids["Water"]
        self._lava_id  = ids["Lava"]
        self._obs_id   = ids["Obsidian"]
        self.oil_id    = ids["Oil"]
        self.fire_id   = ids["Fire"]
        self.smoke_id  = ids["Smoke"]

        self._row_top_boost = np.clip(
            1.0 - np.arange(GRID_HEIGHT, dtype=np.float32)[:, None] / 22.0,
            0.0, 1.0)

        self.prog = self.ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
        quad = np.array([-1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0], dtype=np.float32)
        self.vbo = self.ctx.buffer(quad.tobytes())
        self.vao = self.ctx.vertex_array(self.prog, [(self.vbo, "2f", "in_pos")])
        self.texture = self.ctx.texture(size=(GRID_WIDTH, GRID_HEIGHT), components=3)
        self.texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.texture.use(location=0)
        self.prog["grid_tex"] = 0

        print("Warming up Numba kernels (first run only)…")
        self._run_sim_step()
        print("Done – simulation running.")

    def _run_sim_step(self):
        n = simulate_step(
            self.grid, self.active, self.active_next,
            self.metadata, self.moved,
            self.dirty, self.dirty_next,
            self.density_lut, self.move_lut, self.viscosity_lut,
            self._steam_id, self._water_id, self._lava_id, self._obs_id,
            self.oil_id, self.fire_id, self.smoke_id,
            self._idx_buf,
        )
        self.active,  self.active_next = self.active_next, self.active
        self.dirty,   self.dirty_next  = self.dirty_next,  self.dirty
        return n

    def _window_to_grid(self, wx, wy):
        gx = int(wx / WINDOW_WIDTH  * GRID_WIDTH)
        gy = int(wy / WINDOW_HEIGHT * GRID_HEIGHT)
        return (gx, gy) if 0 <= gx < GRID_WIDTH and 0 <= gy < GRID_HEIGHT else (-1, -1)

    def _paint(self, wx, wy, eid):
        gx, gy = self._window_to_grid(wx, wy)
        if gx == -1:
            return
        r = self.brush_r
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT:
                    self.grid[ny, nx] = eid
                    self.active[ny, nx] = True
                    if eid > 0:
                        name = ELEMENTS[eid - 1]["name"]
                        if name == "Fire":
                            self.metadata[ny, nx] = np.random.randint(80, 160)
                        elif name == "Steam":
                            self.metadata[ny, nx] = np.random.randint(225, 255)
                        elif name == "Smoke":
                            self.metadata[ny, nx] = np.random.randint(70, 130)
                        else:
                            self.metadata[ny, nx] = 0
                    cr = ny // CHUNK_SIZE
                    cc = nx // CHUNK_SIZE
                    self.dirty[cr, cc] = True

    def _build_frame_rgb(self):
        frame      = self.color_lut[self.grid].copy()

        fire_mask = (self.grid == self.fire_id)
        if np.any(fire_mask):
            life_f = self.metadata.astype(np.float32) / 160.0 # normalise to 0-1
            life_f = np.clip(life_f, 0.0, 1.0)
            rows, cols = np.nonzero(fire_mask)
            t = life_f[rows, cols]
            # High life = hot white-yellow core, low life = dim red ember
            frame[rows, cols, 0] = np.clip(200 + 55 * t, 0, 255).astype(np.uint8)
            frame[rows, cols, 1] = np.clip(30 + 130 * t, 0, 255).astype(np.uint8)
            frame[rows, cols, 2] = np.clip( 50 * t * t, 0, 255).astype(np.uint8)

        steam_mask = (self.grid == self._steam_id)
        if not np.any(steam_mask):
            return frame

        steam_u8 = steam_mask.astype(np.uint8)
        padded   = np.pad(steam_u8, 1, mode="constant")
        density  = (
            padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
            padded[1:-1, :-2]+ padded[1:-1, 1:-1]+ padded[1:-1, 2:] +
            padded[2:,  :-2] + padded[2:,  1:-1] + padded[2:,  2:]
        ).astype(np.float32)

        life         = self.metadata.astype(np.float32) / 255.0
        density_fac  = np.clip((density - 1.5) / 5.5, 0.0, 1.0)
        cloud_factor = density_fac * density_fac
        vapor        = 0.12 + 0.26 * life + 0.42 * cloud_factor + 0.28 * self._row_top_boost
        vapor        = np.clip(vapor, 0.0, 1.0)

        rows, cols = np.nonzero(steam_mask)
        v = vapor[rows, cols]
        frame[rows, cols, 0] = (EMPTY[0] + (232 - EMPTY[0]) * v).astype(np.uint8)
        frame[rows, cols, 1] = (EMPTY[1] + (238 - EMPTY[1]) * v).astype(np.uint8)
        frame[rows, cols, 2] = (EMPTY[2] + (248 - EMPTY[2]) * v).astype(np.uint8)
        return frame

    def on_render(self, time, frame_time):
        self._sim_accum += frame_time
        n_active = 0
        while self._sim_accum >= self.sim_speed:
            n_active = self._run_sim_step()
            self._sim_accum -= self.sim_speed

        fps = 1.0 / max(frame_time, 1e-6)
        self.wnd.title = (
            f"Pixel Sim  |  {fps:5.0f} FPS  |  "
            f"{n_active:,} active cells  |  "
            f"1–8 select  R reset"
        )
        self.texture.write(self._build_frame_rgb().tobytes())
        self.ctx.clear()
        self.vao.render(moderngl.TRIANGLE_STRIP)

    def on_mouse_drag_event(self, x, y, dx, dy):
        if self.wnd.mouse_states.left:
            self._paint(x, y, self.selected)
        elif self.wnd.mouse_states.right:
            self._paint(x, y, 0)

    def on_key_event(self, key, action, modifiers):
        if action == self.wnd.keys.ACTION_PRESS:
            keys    = self.wnd.keys
            mapping = {
                keys.NUMBER_1: 1, keys.NUMBER_2: 2, #Sand, Water
                keys.NUMBER_3: 3, keys.NUMBER_4: 4, #Stone, Lava
                keys.NUMBER_5: 5, keys.NUMBER_6: 6, #Idk and Idk
                keys.NUMBER_7: 7, keys.NUMBER_8: 8, #Idk and Idk
                keys.NUMBER_9: 9, #Oil
            }
            if key in mapping:
                self.selected = mapping[key]
            if key == keys.R:
                self.grid[:]     = 0
                self.metadata[:] = 0
                self.active[:]   = True
                self.dirty[:]    = True
            if key == keys.UP:
                self.sim_speed = max(1/120, self.sim_speed / 1.5)  # faster
            if key == keys.DOWN:
                self.sim_speed = min(1.0, self.sim_speed * 1.5)    # slower


if __name__ == "__main__":
    mglw.run_window_config(Sim)