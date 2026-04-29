"""
demo.py — Pygame field renderer for robot table-tennis-ball collector.

Renders:
  • Field with green surface and grid
  • Borders inferred from 4 corners (left = big goal gap ~1/4 width,
    right = small goal gap ~1/8 width)
  • Corner markers
  • Cross obstacle (with orientation)
  • Balls (normal = white, VIP = orange-gold)
  • Robot (with orientation arrow)

Run:  python demo.py
"""

import math
import sys

import pygame

# ---------------------------------------------------------------------------
# Local imports — ensure the other files are on the path
# ---------------------------------------------------------------------------
import os
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "model"))
sys.path.insert(0, MODEL_DIR)

from ball import Ball
from corner import Corner
from cross import Cross
from robot import Robot
from state import FieldState

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
C_BG            = (15,  20,  25)       # near-black background
C_FIELD         = (34,  85,  45)       # dark grass green
C_FIELD_ALT     = (31,  79,  41)       # slightly darker for checkerboard
C_GRID          = (38,  95,  50)       # subtle grid lines
C_BORDER        = (220, 215, 180)      # off-white border lines
C_CORNER        = (255, 220,  60)      # yellow corner markers
C_BALL          = (240, 240, 240)      # normal ball
C_BALL_VIP      = (255, 185,  30)      # VIP ball (gold)
C_BALL_OUTLINE  = ( 80,  80,  80)
C_VIP_GLOW      = (255, 200,  60)
C_ROBOT_BODY    = ( 30, 140, 220)      # blue robot body
C_ROBOT_ARROW   = (255, 255, 255)
C_ROBOT_OUTLINE = ( 10,  80, 160)
C_CROSS         = (200,  55,  55)      # red cross obstacle
C_CROSS_OUTLINE = (120,  20,  20)
C_GOAL_BIG      = ( 50, 200, 120)      # big goal highlight (left)
C_GOAL_SMALL    = (200, 180,  40)      # small goal highlight (right)
C_LABEL         = (200, 210, 200)
C_PANEL_BG      = ( 22,  30,  38)

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

# 
PANEL_W             = 260
WINDOW_W, WINDOW_H = 1503 + PANEL_W, 1093.5
FIELD_MARGIN        = 60            # pixels from window edge to field corners

FIELD_X0 = FIELD_MARGIN
FIELD_Y0 = FIELD_MARGIN
FIELD_X1 = WINDOW_W - PANEL_W - FIELD_MARGIN
FIELD_Y1 = WINDOW_H - FIELD_MARGIN

# Field IRL is 167; 121.5. Scaled to pixels with the window ratio it is 1383; 973.5
FIELD_W  = FIELD_X1 - FIELD_X0
FIELD_H  = FIELD_Y1 - FIELD_Y0

# Goal sizes relative to field height (Big goal is 16 cm, small is 8.5)
BIG_GOAL_RATIO   = 0.131
SMALL_GOAL_RATIO = 0.07

FPS = 60

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def lerp(a, b, t):
    return a + (b - a) * t


def draw_dashed_line(surf, colour, p1, p2, dash=8, gap=5, width=1):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length  = math.hypot(dx, dy)
    if length == 0:
        return
    ux, uy  = dx / length, dy / length
    pos     = 0
    drawing = True
    while pos < length:
        seg = dash if drawing else gap
        end = min(pos + seg, length)
        if drawing:
            x0 = p1[0] + ux * pos
            y0 = p1[1] + uy * pos
            x1 = p1[0] + ux * end
            y1 = p1[1] + uy * end
            pygame.draw.line(surf, colour, (int(x0), int(y0)), (int(x1), int(y1)), width)
        pos     += seg
        drawing  = not drawing


def draw_arrow(surf, colour, origin, angle_deg, length, tip_size=10, width=3):
    """Draw a filled arrow from origin in direction angle_deg (degrees)."""
    rad  = math.radians(angle_deg)
    ex   = origin[0] + math.cos(rad) * length
    ey   = origin[1] - math.sin(rad) * length   # pygame y-axis is flipped
    pygame.draw.line(surf, colour, origin, (int(ex), int(ey)), width)
    # arrowhead
    left_rad  = rad + math.radians(150)
    right_rad = rad - math.radians(150)
    lx = ex + math.cos(left_rad)  * tip_size
    ly = ey - math.sin(left_rad)  * tip_size
    rx = ex + math.cos(right_rad) * tip_size
    ry = ey - math.sin(right_rad) * tip_size
    pygame.draw.polygon(surf, colour, [(int(ex), int(ey)), (int(lx), int(ly)), (int(rx), int(ry))])


def draw_cross_shape(surf, pos, size, angle_deg, colour, outline, width=14):
    """Draw a + shaped cross centred at pos, rotated by angle_deg."""
    cx, cy = pos
    for arm_offset in [0, 90]:
        rad = math.radians(angle_deg + arm_offset)
        dx, dy = math.cos(rad) * size, math.sin(rad) * size
        p1 = (cx - dx, cy - dy)
        p2 = (cx + dx, cy + dy)
        pygame.draw.line(surf, outline, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), width + 4)
        pygame.draw.line(surf, colour,  (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), width)


# ---------------------------------------------------------------------------
# Drawing sub-routines
# ---------------------------------------------------------------------------

def draw_field_surface(surf, corners: list[Corner]):
    """Fill the field polygon defined by the four corners."""
    pts = [c.position for c in corners]
    pygame.draw.polygon(surf, C_FIELD, pts)


def draw_borders(surf, corners: list[Corner]):
    tl, tr, br, bl = [c.position for c in corners]
    lw = 3

    # Top and bottom — full lines
    pygame.draw.line(surf, C_BORDER, tl, tr, lw)
    pygame.draw.line(surf, C_BORDER, bl, br, lw)

    # Left border — big goal gap
    left_len = math.hypot(bl[0]-tl[0], bl[1]-tl[1])
    half_gap = left_len * BIG_GOAL_RATIO / 2
    mid_t = 0.5 - (half_gap / left_len)
    mid_b = 0.5 + (half_gap / left_len)
    gap_top = (int(lerp(tl[0], bl[0], mid_t)), int(lerp(tl[1], bl[1], mid_t)))
    gap_bot = (int(lerp(tl[0], bl[0], mid_b)), int(lerp(tl[1], bl[1], mid_b)))
    pygame.draw.line(surf, C_BORDER, tl, gap_top, lw)
    pygame.draw.line(surf, C_BORDER, gap_bot, bl, lw)
    pygame.draw.line(surf, C_GOAL_BIG, gap_top, gap_bot, 5)

    # Right border — small goal gap
    right_len = math.hypot(br[0]-tr[0], br[1]-tr[1])
    half_gap = right_len * SMALL_GOAL_RATIO / 2
    mid_t = 0.5 - (half_gap / right_len)
    mid_b = 0.5 + (half_gap / right_len)
    gap_top = (int(lerp(tr[0], br[0], mid_t)), int(lerp(tr[1], br[1], mid_t)))
    gap_bot = (int(lerp(tr[0], br[0], mid_b)), int(lerp(tr[1], br[1], mid_b)))
    pygame.draw.line(surf, C_BORDER, tr, gap_top, lw)
    pygame.draw.line(surf, C_BORDER, gap_bot, br, lw)
    pygame.draw.line(surf, C_GOAL_SMALL, gap_top, gap_bot, 5)


def draw_corners(surf, corners: list[Corner]):
    for c in corners:
        x, y = c.position
        size = 10
        pygame.draw.line(surf, C_CORNER, (x - size, y), (x + size, y), 3)
        pygame.draw.line(surf, C_CORNER, (x, y - size), (x, y + size), 3)
        pygame.draw.circle(surf, C_CORNER, (x, y), 5)


def draw_balls(surf, balls: list[Ball], corners: list[Corner]):
    for ball in balls:
        x, y = field_to_screen(ball.position, corners)
        r    = 8
        if ball.is_vip:
            # glow ring
            for i in range(3, 0, -1):
                alpha_surf = pygame.Surface((r * 2 + i * 6, r * 2 + i * 6), pygame.SRCALPHA)
                alpha      = 60 - i * 15
                pygame.draw.circle(alpha_surf, (*C_VIP_GLOW, alpha),
                                   (r + i * 3, r + i * 3), r + i * 3)
                surf.blit(alpha_surf, (x - r - i * 3, y - r - i * 3))
            pygame.draw.circle(surf, C_BALL_VIP,     (x, y), r)
            pygame.draw.circle(surf, (255, 230, 120), (x, y), r, 2)
        else:
            pygame.draw.circle(surf, C_BALL,         (x, y), r)
            pygame.draw.circle(surf, C_BALL_OUTLINE,  (x, y), r, 1)


def draw_cross(surf, cross: Cross, corners: list[Corner]):
    # TODO: get proper size of cross (vibe sized)
    draw_cross_shape(surf, field_to_screen(cross.position, corners), 40*1.42, cross.orientation,
                     C_CROSS, C_CROSS_OUTLINE, width=14)
    pygame.draw.circle(surf, C_CROSS_OUTLINE, field_to_screen(cross.position, corners), 6)
    pygame.draw.circle(surf, C_CROSS,         field_to_screen(cross.position, corners), 4)


def draw_robot(surf, robot: Robot):
    x, y = robot.position
    r    = 16
    # Body
    pygame.draw.circle(surf, C_ROBOT_OUTLINE, (x, y), r + 2)
    pygame.draw.circle(surf, C_ROBOT_BODY,    (x, y), r)
    # Orientation arrow (pygame y is flipped → negate angle)
    draw_arrow(surf, C_ROBOT_ARROW, (x, y), robot.orientation, r + 10, tip_size=7, width=2)

def field_to_screen(pos: tuple[int, int], corners: list[Corner]) -> tuple[int, int]:
    tl, tr, br, bl = [c.position for c in corners]
    x = int(lerp(tl[0], tr[0], pos[0] / FIELD_W))
    y = int(lerp(bl[1], tl[1], pos[1] / FIELD_H))  # y flipped: 0 = bottom
    return (x, y)

# ---------------------------------------------------------------------------
# Side panel
# ---------------------------------------------------------------------------

def draw_panel(surf, font_sm, font_md, font_lg,
               robot: Robot, balls: list[Ball], cross: Cross,
               corners: list[Corner]):
    px = WINDOW_W - PANEL_W + 15
    pw = PANEL_W - FIELD_MARGIN // 2
    panel_rect = pygame.Rect(WINDOW_W - PANEL_W, 0, PANEL_W, WINDOW_H)
    pygame.draw.rect(surf, C_PANEL_BG, panel_rect)
    pygame.draw.line(surf, C_BORDER, (WINDOW_W - PANEL_W, 0), (WINDOW_W - PANEL_W, WINDOW_H), 1)

    y = 30
    def heading(text):
        nonlocal y
        label = font_md.render(text, True, C_CORNER)
        surf.blit(label, (px, y))
        y += label.get_height() + 4
        pygame.draw.line(surf, C_GRID, (px, y), (px + pw - 10, y), 1)
        y += 8

    def row(key, val):
        nonlocal y
        k = font_sm.render(key, True, (140, 155, 145))
        v = font_sm.render(str(val), True, C_LABEL)
        surf.blit(k, (px, y))
        surf.blit(v, (px + 100, y))
        y += k.get_height() + 4

    # Robot
    heading("ROBOT")
    row("Position",    robot.position)
    row("Orientation", f"{robot.orientation:.1f}°")
    y += 8

    # Cross
    heading("CROSS")
    row("Position",    cross.position)
    row("Orientation", f"{cross.orientation:.1f}°")
    y += 8

    # Balls
    heading(f"BALLS  ({len(balls)})")
    for i, b in enumerate(balls):
        tag = "VIP" if b.is_vip else f"#{i}"
        row(tag, b.position)
    y += 8

    # Corners
    heading("CORNERS")
    labels = ["TL", "TR", "BR", "BL"]
    for lbl, c in zip(labels, corners):
        row(lbl, c.position)
    y += 16

    # Goals info
    heading("GOALS")
    #row("Left (big)")
    #row("Right (small)")
    y += 16

    # Legend
    heading("LEGEND")
    items = [
        (C_CORNER,     "Corner"),
        (C_BALL,       "Ball"),
        (C_BALL_VIP,   "VIP Ball"),
        (C_ROBOT_BODY, "Robot"),
        (C_CROSS,      "Cross"),
        (C_GOAL_BIG,   "Big Goal"),
        (C_GOAL_SMALL, "Small Goal"),
    ]
    for colour, label in items:
        pygame.draw.circle(surf, colour, (px + 6, y + 7), 5)
        txt = font_sm.render(label, True, C_LABEL)
        surf.blit(txt, (px + 16, y))
        y += txt.get_height() + 4


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_gui(state: FieldState):
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Robot Field — Demo Renderer")
    clock  = pygame.time.Clock()

    font_sm = pygame.font.SysFont("monospace", 13)
    font_md = pygame.font.SysFont("monospace", 15, bold=True)
    font_lg = pygame.font.SysFont("monospace", 20, bold=True)

    # Animation state
    t = 0.0

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        t += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        with state.lock:
            robot = state.robot
            balls = list(state.balls)
            cross = state.cross
            corners = list(state.corners)

        # ------------------------------------------------------------------
        # Draw
        # ------------------------------------------------------------------
        screen.fill(C_BG)

        draw_field_surface(screen, corners)
        draw_borders(screen, corners)
        draw_corners(screen, corners)
        draw_cross(screen, cross, corners)
        draw_balls(screen, balls, corners)
        draw_robot(screen, robot)
        draw_panel(screen, font_sm, font_md, font_lg, robot, balls, cross, corners)

        # Title
        title = font_lg.render("Controller GUI", True, C_LABEL)
        screen.blit(title, (FIELD_X0, 14))

        # Controls hint
        hint = font_sm.render("Closing this window will exit the controller", True, (80, 95, 85))
        screen.blit(hint, (FIELD_X0, WINDOW_H - 22))

        pygame.display.flip()

    pygame.quit()
    sys.exit()

def get_test_field_state():
    state = FieldState()
    state.corners = [
        Corner((FIELD_X0, FIELD_Y0)),   # top-left
        Corner((FIELD_X1, FIELD_Y0)),   # top-right
        Corner((FIELD_X1, FIELD_Y1)),   # bottom-right
        Corner((FIELD_X0, FIELD_Y1)),   # bottom-left
    ]
    state.balls = [
        Ball((FIELD_X0 + 120, FIELD_Y0 + 90),  is_vip=False),
        Ball((FIELD_X0 + 280, FIELD_Y0 + 200), is_vip=False),
        Ball((FIELD_X0 + 420, FIELD_Y0 + 310), is_vip=True),   # VIP ball
        Ball((FIELD_X0 + 180, FIELD_Y0 + 380), is_vip=False),
        Ball((FIELD_X0 + 510, FIELD_Y0 + 120), is_vip=False),
        Ball((FIELD_X0 + 60,  FIELD_Y0 + 480), is_vip=False),
    ]
    state.cross  = Cross(position=(FIELD_X0 + FIELD_W // 2, FIELD_Y0 + FIELD_H // 2),
                   orientation=25.0)
    state.robot  = Robot(position=(FIELD_X0 + 100, FIELD_Y0 + FIELD_H // 2),
                   orientation=35.0)
    return state