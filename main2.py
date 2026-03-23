"""
The "To-Do" List for your Sim

    Buoyancy & Layering (The Obsidian Fix): * Right now, Obsidian sinks because its density (3.5) is higher than Lava (2.2).

        To create a "crust" layer, we must check if a cell is "supported." If Obsidian is touching Lava or Water, we can give it a "Surface Tension" flag or simply lower its density dynamically so it floats on the interface.

    Gas Diffusion & Stretching:

        Gases currently "stack" at the top because they only check for empty cells.

        To make them "stretch," we need to implement Horizontal Displacement. If a gas particle is at the top (row 0), it should aggressively move left or right if there is space, simulating gas expanding to fill its container.

    The Pressure System:

        We add a pressure array. Liquids in a vertical column increase the pressure of the cells below them.

        If a liquid cell has high pressure and an empty neighbor, it "squirts" into that neighbor, even if it has to move upward or sideways (Equalization).

    Temperature (Heat Transfer):

        We add a temp array. Lava has high heat; Water is a heat sink.

        Heat should "diffuse" to neighbors. When a Water cell's temp exceeds a threshold, it turns into Steam. When Lava's temp drops, it turns into Obsidian.
"""

import moderngl
import moderngl_window as mglw
import numpy as np
from PIL import ImageFont

# ============================================================
# WINDOW / GRID CONSTANTS
# ============================================================
WINDOW_WIDTH  = 1280
WINDOW_HEIGHT = 800
GRID_WIDTH  = 320
GRID_HEIGHT = 200

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

ELEMENTS: dict = {}

class Element:
    def __init__(self, name: str, density: float, move: int, color: tuple, viscosity: float = 0.0):
        self.name = name
        self.density = density
        self.viscosity = viscosity
        self.move = move
        self.color = color
        
        eid = len(ELEMENTS)
        ELEMENTS[eid] = {
            "name":      self.name,
            "density":   self.density,
            "viscosity": self.viscosity,
            "move":      self.move,
            "color":     self.color,
        }

# --- REGISTERED MATERIALS ---
# Note: Obsidian move changed to 1 (Powder) so it falls if unsupported
Sand     = Element("Sand",     1.6, 1, SAND)
Water    = Element("Water",    1.0, 2, WATER,    viscosity=0.05)
Stone    = Element("Stone",    2.5, 0, STONE)
Lava     = Element("Lava",     2.2, 2, LAVA,     viscosity=0.75)
Obsidian = Element("Obsidian", 3.5, 1, OBSIDIAN) 
Fire     = Element("Fire",     0.0, 4, FIRE)
Steam    = Element("Steam",    0.0, 3, STEAM)
Smoke    = Element("Smoke",    0.0, 3, SMOKE)

# ============================================================
# SHADERS
# ============================================================
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

class Sim(mglw.WindowConfig):
    gl_version  = (4, 3)
    title       = "Pixel Sim - Enhanced Physics"
    window_size = (WINDOW_WIDTH, WINDOW_HEIGHT)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.uint8)
        self.active = np.ones((GRID_HEIGHT, GRID_WIDTH), dtype=bool)
        self.active_next = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=bool)
        self.metadata = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.uint8)
        self.moved = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=bool)

        self.selected = 1
        self.brush_r  = 2

        lut_size = len(ELEMENTS) + 1
        self.color_lut = np.zeros((lut_size, 3), dtype=np.uint8)
        self.color_lut[0] = EMPTY
        for eid, data in ELEMENTS.items():
            self.color_lut[eid + 1] = data["color"]

        self._ids = {data["name"]: eid for eid, data in ELEMENTS.items()}
        self.prog = self.ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
        quad = np.array([-1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0], dtype=np.float32)
        self.vbo = self.ctx.buffer(quad.tobytes())
        self.vao = self.ctx.vertex_array(self.prog, [(self.vbo, "2f", "in_pos")])
        self.texture = self.ctx.texture(size=(GRID_WIDTH, GRID_HEIGHT), components=3)
        self.texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.texture.use(location=0)
        self.prog["grid_tex"] = 0

    def _window_to_grid(self, wx: float, wy: float) -> tuple[int, int]:
        gx = int(wx / WINDOW_WIDTH  * GRID_WIDTH)
        gy = int(wy / WINDOW_HEIGHT * GRID_HEIGHT)
        return (gx, gy) if 0 <= gx < GRID_WIDTH and 0 <= gy < GRID_HEIGHT else (-1, -1)

    def _can_displace(self, mover_eid: int, target_cell: int) -> bool:
        if target_cell == 0: return True
        return ELEMENTS[mover_eid]["density"] > ELEMENTS[target_cell - 1]["density"]

    def _move(self, r1: int, c1: int, r2: int, c2: int):
        self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
        self.moved[r2, c2] = True
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                for nr, nc in [(r2+dr, c2+dc), (r1+dr, c1+dc)]:
                    if 0 <= nr < GRID_HEIGHT and 0 <= nc < GRID_WIDTH:
                        self.active_next[nr, nc] = True

    def _move_gas(self, r1: int, c1: int, r2: int, c2: int):
        self.metadata[r2, c2], self.metadata[r1, c1] = self.metadata[r1, c1], 0
        self._move(r1, c1, r2, c2)

    def _paint(self, wx: float, wy: float, eid: int):
        gx, gy = self._window_to_grid(wx, wy)
        if gx == -1: return
        r = self.brush_r
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT:
                    self.grid[ny, nx] = eid
                    self.active[ny, nx] = True
                    if eid > 0:
                        name = ELEMENTS[eid - 1]["name"]
                        if name == "Fire": self.metadata[ny, nx] = np.random.randint(80, 160)
                        elif name == "Steam": self.metadata[ny, nx] = np.random.randint(150, 255)
                        elif name == "Smoke": self.metadata[ny, nx] = np.random.randint(40, 100)
                        else: self.metadata[ny, nx] = 0

    def _react(self, row: int, col: int, obs_id: int, stm_id: int, fire_id: int, wat_id: int):
        # FIX 1: Prevent mid-air reaction if lava is currently falling
        if row < GRID_HEIGHT - 1 and self.grid[row + 1, col] == 0:
            return

        # Cooldown/Lifetime decay
        if self.metadata[row, col] > 0:
            self.metadata[row, col] -= 1
            self.active_next[row, col] = True
            return

        # Check neighbors for reaction (Water + Lava)
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = row + dr, col + dc
            if not (0 <= nr < GRID_HEIGHT and 0 <= nc < GRID_WIDTH): continue
            neighbor = self.grid[nr, nc]
            if neighbor == 0: continue
            
            n_name = ELEMENTS[neighbor - 1]["name"]
            if n_name == "Water":
                self.grid[row, col] = obs_id + 1 # Turn Lava to Obsidian
                self.grid[nr, nc] = 0 # Consume Water
                
                # Steam burst logic
                for _ in range(6):
                    sr, sc = nr + np.random.randint(-2, 3), nc + np.random.randint(-2, 3)
                    if 0 <= sr < GRID_HEIGHT and 0 <= sc < GRID_WIDTH and self.grid[sr, sc] == 0:
                        self.grid[sr, sc] = stm_id + 1
                        self.metadata[sr, sc] = np.random.randint(150, 255)
                        self.active_next[sr, sc] = True
                self.active_next[row, col] = True
                return

    def _update_sim(self):
        self.active_next[:] = False
        self.moved[:] = False
        
        ids = self._ids
        rows, cols = np.where(self.active)
        indices = np.arange(len(rows))
        np.random.shuffle(indices)

        for i in indices:
            r, c = int(rows[i]), int(cols[i])
            cell = self.grid[r, c]
            if cell == 0 or self.moved[r, c]: continue
            if r >= GRID_HEIGHT - 1: continue

            eid = cell - 1
            move = ELEMENTS[eid]["move"]
            below, above = r + 1, r - 1

            # LAVA REACTION LOGIC
            if cell == (ids["Lava"] + 1):
                self._react(r, c, ids["Obsidian"], ids["Steam"], ids["Fire"], ids["Water"])
                cell = self.grid[r, c]
                if cell == 0: continue
                eid = cell - 1
                move = ELEMENTS[eid]["move"]

            # POWDER PHYSICS (Sand, Obsidian)
            if move == 1:
                if self._can_displace(eid, self.grid[below, c]):
                    self._move(r, c, below, c)
                else:
                    for dx in np.random.permutation([-1, 1]):
                        nc = c + dx
                        if 0 <= nc < GRID_WIDTH and self._can_displace(eid, self.grid[below, nc]):
                            self._move(r, c, below, nc)
                            break

            # LIQUID PHYSICS (Water, Lava)
            elif move == 2:
                if np.random.random() < ELEMENTS[eid]["viscosity"]:
                    self.active_next[r, c] = True
                    continue
                
                # Try falling
                if self._can_displace(eid, self.grid[below, c]):
                    self._move(r, c, below, c)
                else:
                    # FIX 2: Better Liquid Leveling (checking diagonal then horizontal)
                    moved_liquid = False
                    for dx in np.random.permutation([-1, 1]):
                        nc = c + dx
                        if 0 <= nc < GRID_WIDTH and self._can_displace(eid, self.grid[below, nc]):
                            self._move(r, c, below, nc)
                            moved_liquid = True
                            break
                    if not moved_liquid:
                        for dx in np.random.permutation([-1, 1]):
                            nc = c + dx
                            if 0 <= nc < GRID_WIDTH and self.grid[r, nc] == 0:
                                self._move(r, c, r, nc)
                                break

            # GAS PHYSICS (Steam, Smoke)
            elif move == 3:
                if self.metadata[r, c] > 0:
                    if np.random.random() < 0.01: self.metadata[r, c] -= 1
                else:
                    self.grid[r, c] = 0
                    continue
                
                moved_gas = False
                if above >= 0 and self.grid[above, c] == 0:
                    self._move_gas(r, c, above, c)
                    moved_gas = True
                if not moved_gas and above >= 0:
                    for dx in np.random.permutation([-1, 1]):
                        nc = c + dx
                        if 0 <= nc < GRID_WIDTH and self.grid[above, nc] == 0:
                            self._move_gas(r, c, above, nc)
                            moved_gas = True
                            break
                self.active_next[r, c] = True

        self.active, self.active_next = self.active_next, self.active

    def on_render(self, time: float, frame_time: float):
        for _ in range(2): self._update_sim()
        self.texture.write(self.color_lut[self.grid].tobytes())
        self.ctx.clear()
        self.vao.render(moderngl.TRIANGLE_STRIP)

    def on_mouse_drag_event(self, x, y, dx, dy):
        if self.wnd.mouse_states.left: self._paint(x, y, self.selected)
        elif self.wnd.mouse_states.right: self._paint(x, y, 0)

    def on_key_event(self, key, action, modifiers):
        if action == self.wnd.keys.ACTION_PRESS:
            keys = self.wnd.keys
            mapping = {keys.NUMBER_1: 1, keys.NUMBER_2: 2, keys.NUMBER_3: 3, 
                       keys.NUMBER_4: 4, keys.NUMBER_5: 5, keys.NUMBER_6: 6,
                       keys.NUMBER_7: 7, keys.NUMBER_8: 8}
            if key in mapping: self.selected = mapping[key]
            if key == keys.R: self.grid[:] = 0

if __name__ == "__main__":
    mglw.run_window_config(Sim)