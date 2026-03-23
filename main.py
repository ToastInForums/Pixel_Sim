# ============================================================
#  PIXEL SIM
#  A falling-sand style cellular automaton
#
#  ARCHITECTURE OVERVIEW
#  ┌─────────────┐     ID array      ┌──────────────┐
#  │  ELEMENTS   │ ──────────────── │     GRID     │
#  │  registry   │  ELEMENTS dict    │  numpy uint8 │
#  └─────────────┘                  └──────┬───────┘
#                                          │ color_lut[grid]
#                                   ┌──────▼───────┐
#                                   │   TEXTURE    │
#                                   │  RGB uint8   │
#                                   └──────┬───────┘
#                                          │ sampler2D
#                                   ┌──────▼───────┐
#                                   │   SHADERS    │
#                                   │  fullscreen  │
#                                   │    quad      │
#                                   └─────────────-┘
#
#  ADDING A NEW ELEMENT
#  1. Add its RGB color constant in the "Color Values" section
#  2. Instantiate it:  MyElement = Element("Name", density, move_type, COLOR)
#     - density : float  — heavier sinks below lighter  (sand=1.6, water=1.0)
#     - move    : int    — movement class
#                           0 = static  (stone, obsidian)
#                           1 = powder  (sand)
#                           2 = liquid  (water, lava)
#                           3 = gas     (steam, smoke)
#                           4 = fire    (stationary, spreads, burns out)
#  3. Add update logic in Sim._update_sim() if needed
# ============================================================


# ============================================================
# IMPORTS
# ============================================================
import moderngl
import moderngl_window as mglw
import numpy as np
from PIL import ImageFont


# ============================================================
# WINDOW / GRID CONSTANTS
# ============================================================
WINDOW_WIDTH  = 1280
WINDOW_HEIGHT = 800

# Grid is lower resolution than the window — each cell = 4x4 pixels
# Change these to trade detail for performance
GRID_WIDTH  = 320
GRID_HEIGHT = 200


# ============================================================
# COLOR PALETTE  (R, G, B)
# ============================================================
EMPTY    = (  2,   2,   8)
SAND     = (220, 190, 120)
WATER    = ( 40, 100, 220)
STONE    = (120, 120, 130)
FIRE     = (240,  80,  20)
LAVA     = (207,  70,  20)
OBSIDIAN = ( 30,  20,  45)
STEAM    = (180, 200, 220)
SMOKE    = ( 60,  60,  70)


# ============================================================
# ELEMENT REGISTRY
#
#  Populated automatically when Element() instances are created.
#  Key   = element ID (int, auto-assigned, 0-indexed)
#  Value = {"name", "density", "move", "color"}
#
#  ID 0 is always reserved for EMPTY cells in the grid.
#  So grid cell value 0 = empty, 1 = first element, etc.
# ============================================================
ELEMENTS: dict = {}


# ============================================================
# ELEMENT CLASS
# ============================================================
class Element:
    """
    Defines a material type and registers it in ELEMENTS.

    Parameters
    ----------
    name    : str   — display name
    density : float — relative weight (heavier sinks below lighter)
    move    : int   — movement class (see header)
    color   : tuple — (R, G, B) 0-255
    """

    def __init__(self, name: str, density: float, move: int, color: tuple, viscosity: float = 0.0):
        self.name    = name
        self.density = density
        self.viscosity = viscosity
        self.move    = move
        self.color   = color
        

        eid = len(ELEMENTS)
        ELEMENTS[eid] = {
            "name":    self.name,
            "density": self.density,
            "viscosity": self.viscosity,
            "move":    self.move,
            "color":   self.color,
        }


# ============================================================
# ELEMENTS  —  register materials here
# ============================================================
Sand     = Element("Sand",     1.6, 1, SAND)
Water    = Element("Water",    1.0, 2, WATER,     viscosity=0.05)
Stone    = Element("Stone",    2.5, 0, STONE)
Lava     = Element("Lava",     2.2, 2, LAVA,      viscosity=0.75)
Obsidian = Element("Obsidian", 3.5, 0, OBSIDIAN) # Changed move 0 -> 1 to prevent floating
Fire     = Element("Fire",     0.0, 4, FIRE)
Steam    = Element("Steam",    0.0, 3, STEAM)
Smoke    = Element("Smoke",    0.0, 3, SMOKE)
# ============================================================
# SHADERS
# ============================================================

# Vertex shader — draws a fullscreen quad covering clip space [-1, 1]
VERT = """
#version 430
in  vec2 in_pos;
out vec2 uv;
void main() {
    uv = in_pos * 0.5 + 0.5;   // remap [-1,1] → [0,1]
    uv.y = 1.0 - uv.y;         // flip Y: numpy row 0 = top, GL UV 0 = bottom
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

# Fragment shader — samples the grid texture and outputs the pixel color
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
# UI HELPERS
# ============================================================
def _font(size: int):
    """Load a monospace font, falling back gracefully across platforms."""
    for path in [
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except:
            pass
    return ImageFont.load_default()

FONT  = _font(13)
FONTS = _font(11)

# UI overlay colors (R, G, B, A)
UI_BG     = ( 10,  12,  20, 210)
UI_BORDER = ( 70,  80, 110, 255)
TEXT_DIM  = (140, 150, 170, 255)
TEXT_HI   = (220, 230, 255, 255)


# ============================================================
# SIMULATION
# ============================================================
class Sim(mglw.WindowConfig):
    """
    Main simulation window.

    Render pipeline each frame:
        1. _update_sim()      — advance the cellular automaton
        2. _upload_texture()  — push grid colors to the GPU texture
        3. vao.render()       — draw the fullscreen quad

    Mouse:
        Left-click / drag  → place selected element
        Right-click        → erase

    Keys:
        1 = Sand   2 = Water  3 = Stone  4 = Lava
        5 = Obsidian  6 = Fire  7 = Steam  8 = Smoke
        R = clear grid
    """

    gl_version  = (4, 3)
    title       = "Pixel Sim"
    window_size = (WINDOW_WIDTH, WINDOW_HEIGHT)

    # ── Init ──────────────────────────────────────────────────
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Grid — stores element IDs (0 = empty)
        self.grid        = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.uint8)
        self.active      = np.ones( (GRID_HEIGHT, GRID_WIDTH), dtype=bool)
        self.active_next = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=bool)

        # Per-cell lifetime counter — used by fire, steam, smoke
        self.metadata = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.uint8)

        self.moved = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=bool)


        # Brush state
        self.selected = 1   # element ID to paint (1 = Sand)
        self.brush_r  = 2   # paint radius in grid cells

        # Color lookup table: index = element ID → [R, G, B]
        lut_size = len(ELEMENTS) + 1
        self.color_lut = np.zeros((lut_size, 3), dtype=np.uint8)
        self.color_lut[0] = EMPTY
        for eid, data in ELEMENTS.items():
            self.color_lut[eid + 1] = data["color"]

        # Cache element IDs — looked up once, used everywhere
        self._ids = {data["name"]: eid for eid, data in ELEMENTS.items()}

        # Compile shaders
        self.prog = self.ctx.program(vertex_shader=VERT, fragment_shader=FRAG)

        # Fullscreen quad VAO
        quad = np.array([-1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0], dtype=np.float32)
        self.vbo = self.ctx.buffer(quad.tobytes())
        self.vao = self.ctx.vertex_array(self.prog, [(self.vbo, "2f", "in_pos")])

        # Grid texture — NEAREST filtering for crisp pixel edges
        self.texture = self.ctx.texture(size=(GRID_WIDTH, GRID_HEIGHT), components=3)
        self.texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.texture.use(location=0)
        self.prog["grid_tex"] = 0

        # Seed one sand cell so something is visible on startup
        cx, cy = GRID_WIDTH // 2, GRID_HEIGHT // 4
        self.grid[cy, cx] = 1


    # ── Helpers ───────────────────────────────────────────────
    def _window_to_grid(self, wx: float, wy: float) -> tuple[int, int]:
        """Convert window pixel coords → grid (col, row). Returns (-1,-1) if OOB."""
        gx = int(wx / WINDOW_WIDTH  * GRID_WIDTH)
        gy = int(wy / WINDOW_HEIGHT * GRID_HEIGHT)
        if 0 <= gx < GRID_WIDTH and 0 <= gy < GRID_HEIGHT:
            return gx, gy
        return -1, -1

    def _can_displace(self, mover_eid: int, target_cell: int) -> bool:
        """Return True if mover is denser than the particle in target_cell."""
        if target_cell == 0:
            return True
        return ELEMENTS[mover_eid]["density"] > ELEMENTS[target_cell - 1]["density"]

    def _move(self, r1: int, c1: int, r2: int, c2: int):
        a = self.grid[r1, c1]
        b = self.grid[r2, c2]
        self.grid[r1, c1] = b
        self.grid[r2, c2] = a

        # mark destination as moved
        self.moved[r2, c2] = True

        for dr in range(-1, 2):
            for dc in range(-1, 2):
                for nr, nc in [(r2+dr, c2+dc), (r1+dr, c1+dc)]:
                    if 0 <= nr < GRID_HEIGHT and 0 <= nc < GRID_WIDTH:
                        self.active_next[nr, nc] = True


    def _paint(self, wx: float, wy: float, eid: int):
        """
        Paint element eid at the grid cell under window coords (wx, wy).
        Writes to self.active directly so cells are processed this frame.
        Also initialises metadata for elements that need a lifetime.
        """
        gx, gy = self._window_to_grid(wx, wy)
        if gx == -1:
            return
        r = self.brush_r
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT:
                    self.grid[ny, nx]   = eid
                    self.active[ny, nx] = True
                    # Initialise lifetime — fire/steam/smoke die instantly without this
                    if eid > 0:
                        name = ELEMENTS[eid - 1]["name"]
                        if name == "Fire":
                            self.metadata[ny, nx] = np.random.randint(80, 160)
                        elif name == "Steam":
                            self.metadata[ny, nx] = np.random.randint(150, 255)
                        elif name == "Smoke":
                            self.metadata[ny, nx] = np.random.randint(40, 100)
                        else:
                            self.metadata[ny, nx] = 0

    def _upload_texture(self):
        """Map grid IDs → RGB colors via LUT and push to GPU texture."""
        frame = self.color_lut[self.grid]   # (H, W, 3) uint8 — no Python loop
        self.texture.write(frame.tobytes())

    # ── Reactions ─────────────────────────────────────────────
    def _react(self, row: int, col: int, obsidian_id: int, steam_id: int, fire_id: int, water_id: int):
        cell = self.grid[row, col]
        if cell == 0:
            return
            
        # 1. THE FALLING CHECK
        # If the cell below is empty, the lava is "in flight." 
        # We don't allow it to solidify in mid-air.
        if row < GRID_HEIGHT - 1:
            if self.grid[row + 1, col] == 0:
                return

        # 2. COOLDOWN
        if self.metadata[row, col] > 0:
            self.metadata[row, col] -= 1
            self.active_next[row, col] = True
            return

        # 3. NEIGHBOR CHECK
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = row + dr, col + dc
            if not (0 <= nr < GRID_HEIGHT and 0 <= nc < GRID_WIDTH):
                continue
            neighbour = self.grid[nr, nc]
            if neighbour == 0:
                continue
            n_name = ELEMENTS[neighbour - 1]["name"]

            if n_name == "Water":
                # Only solidify if we are supported by something (not air)
                # This creates the "Crust" on top of water or "Cap" over lava
                self.grid[row, col] = obsidian_id + 1
                self.grid[nr,  nc]  = 0 

                # Steam burst logic
                burst_targets = []
                for br in range(-2, 3):
                    for bc in range(-2, 3):
                        sr, sc = nr + br, nc + bc
                        if (0 <= sr < GRID_HEIGHT and 0 <= sc < GRID_WIDTH
                                and self.grid[sr, sc] == 0):
                            burst_targets.append((sr, sc))
                
                np.random.shuffle(burst_targets)
                for sr, sc in burst_targets[:6]:
                    self.grid[sr, sc]        = steam_id + 1
                    self.metadata[sr, sc]    = np.random.randint(150, 255)
                    self.active_next[sr, sc] = True

                self.active_next[row, col] = True
                return

            if n_name == "Sand":
                if np.random.random() < 0.02:
                    self.grid[nr, nc]        = fire_id + 1
                    self.metadata[nr, nc]    = np.random.randint(60, 140)
                    self.metadata[row, col]  = np.random.randint(10, 30)
                    self.active_next[nr, nc] = True

    # ── Simulation step ───────────────────────────────────────
    def _update_sim(self):
        self.active_next[:] = False
        self.moved[:] = False

        fire_id     = self._ids["Fire"]
        smoke_id    = self._ids["Smoke"]
        steam_id    = self._ids["Steam"]
        water_id    = self._ids["Water"]
        obsidian_id = self._ids["Obsidian"]
        lava_id     = self._ids["Lava"]

        rows, cols = np.where(self.active)
        indices    = np.arange(len(rows))
        np.random.shuffle(indices)

        for i in indices:
            row, col = int(rows[i]), int(cols[i])
            cell = self.grid[row, col]
            
            if cell == 0 or self.moved[row, col]:
                continue
                
            if row >= GRID_HEIGHT - 1:
                continue

            eid   = cell - 1
            move  = ELEMENTS[eid]["move"]
            below = row + 1
            above = row - 1

            # --- LAVA REACTION ---
            if cell == (lava_id + 1):
                self._react(row, col, obsidian_id, steam_id, fire_id, water_id)
                cell = self.grid[row, col]
                if cell == 0: continue
                eid = cell - 1
                move = ELEMENTS[eid]["move"]

            # --- POWDER PHYSICS (Sand, Stone, Obsidian) ---
            if move == 1:
                if self._can_displace(eid, self.grid[below, col]):
                    self._move(row, col, below, col)
                else:
                    dirs = [-1, 1]
                    np.random.shuffle(dirs)
                    for dx in dirs:
                        nc = col + dx
                        if 0 <= nc < GRID_WIDTH and self._can_displace(eid, self.grid[below, nc]):
                            self._move(row, col, below, nc)
                            break

            # --- LIQUID PHYSICS (Water, Lava) ---
            elif move == 2:
                if np.random.random() < ELEMENTS[eid]["viscosity"]:
                    self.active_next[row, col] = True
                    continue
                moved_liquid = False
                if self._can_displace(eid, self.grid[below, col]):
                    self._move(row, col, below, col)
                    moved_liquid = True
                else:
                    dirs = [-1, 1]
                    np.random.shuffle(dirs)
                    for dx in dirs:
                        nc = col + dx
                        if 0 <= nc < GRID_WIDTH and self._can_displace(eid, self.grid[below, nc]):
                            self._move(row, col, below, nc)
                            moved_liquid = True
                            break
                if not moved_liquid:
                    dirs = [-1, 1]
                    np.random.shuffle(dirs)
                    for dx in dirs:
                        nc = col + dx
                        if 0 <= nc < GRID_WIDTH and self.grid[row, nc] == 0:
                            self._move(row, col, row, nc)
                            moved_liquid = True
                            break

            # --- GAS PHYSICS (Steam, Smoke) ---
            elif move == 3:
                if self.metadata[row, col] > 0:
                    if np.random.random() < 0.005:
                        self.metadata[row, col] -= 1
                else:
                    self.grid[row, col] = 0
                    self.active_next[row, col] = True
                    continue

                moved_gas = False
                if above >= 0 and self.grid[above, col] == 0:
                    self.metadata[above, col] = self.metadata[row, col]
                    self.metadata[row, col] = 0
                    self._move(row, col, above, col)
                    moved_gas = True

            # --- FIRE PHYSICS ---
            elif move == 4:
                if self.metadata[row, col] > 0:
                    self.metadata[row, col] -= 1
                    self.active_next[row, col] = True
                else:
                    self.grid[row, col] = 0
                    self.active_next[row, col] = True
                    continue

        self.active, self.active_next = self.active_next, self.active

    # ── Frame loop ────────────────────────────────────────────
    def on_render(self, time: float, frame_time: float):
        for _ in range(2):   # ticks per frame — increase for faster sim
            self._update_sim()
        self._upload_texture()
        self.ctx.clear()
        self.vao.render(moderngl.TRIANGLE_STRIP)

    # ── Input ─────────────────────────────────────────────────
    def on_mouse_press_event(self, x: float, y: float, button: int):
        if button == 1:    self._paint(x, y, self.selected)   # left  → paint
        elif button == 2:  self._paint(x, y, 0)               # right → erase

    def on_mouse_drag_event(self, x: float, y: float, dx: float, dy: float):
        buttons = self.wnd.mouse_states
        if buttons.left:   self._paint(x, y, self.selected)
        elif buttons.right: self._paint(x, y, 0)

    def on_key_event(self, key, action, modifiers):
        if action == self.wnd.keys.ACTION_PRESS:
            if key == self.wnd.keys.NUMBER_1: self.selected = 1  # Sand
            if key == self.wnd.keys.NUMBER_2: self.selected = 2  # Water
            if key == self.wnd.keys.NUMBER_3: self.selected = 3  # Stone
            if key == self.wnd.keys.NUMBER_4: self.selected = 4  # Lava
            if key == self.wnd.keys.NUMBER_5: self.selected = 5  # Obsidian
            if key == self.wnd.keys.NUMBER_6: self.selected = 6  # Fire
            if key == self.wnd.keys.NUMBER_7: self.selected = 7  # Steam
            if key == self.wnd.keys.NUMBER_8: self.selected = 8  # Smoke
            if key == self.wnd.keys.R:        self.grid[:] = 0   # Clear


# ============================================================
# MAIN
# ============================================================
def displayElements():
    print("─" * 52)
    print("  REGISTERED ELEMENTS")
    print("─" * 52)
    for eid, data in ELEMENTS.items():
        print(f"  [{eid + 1}] {data['name']:<12} density={data['density']}  move={data['move']}")
    print("─" * 52)


if __name__ == "__main__":
    displayElements()
    mglw.run_window_config(Sim)