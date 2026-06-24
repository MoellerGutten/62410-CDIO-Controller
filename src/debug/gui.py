import math
import sys
import cv2
import numpy as np
import pygame
from src.lib.cross_waypoints import get_cross_waypoints
from src.autonomous_mode.cross_avoidance_helpers import calculate_shortest_waypoint_path
from src.autonomous_mode.start_autonomous_session import _select_next_ball
from src.lib.constants import CROSS_APPROACH_POINTS_HORIZONTAL_OFFSET, CROSS_APPROACH_POINTS_VERTICAL_OFFSET, CROSS_AVOIDANCE_WAYPOINTS_OFFSET, CROSS_FINAL_APPROACH_HORIZONTAL_OFFSET, CROSS_FINAL_APPROACH_VERTICAL_OFFSET, ARENA_WIDTH_CM, ARENA_HEIGHT_CM
from src.model.ball import Ball
from src.model.corner import Corner
from src.model.cross import Cross
from src.model.robot import Robot
from src.model.arena_state import ArenaState
from math import floor
from time import time
from src.lib.cross_approach_points import get_cross_approach_points
from src.state.arena_tracker import ArenaTracker
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
C_CROSS_WAYPOINT= ( 222, 32, 29 )
C_FINAL_POINTS  = (245, 66, 176)

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

# 
PANEL_W             = 260
WINDOW_W, WINDOW_H = (1503 * 0.8 + PANEL_W), (1093.5 * 0.8)
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

#_start_time = 0 #

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


def draw_cross_bounding_box(surf, pos, size, angle_deg, colour, width=10):
    """
    Draw a cross shape rotated according to angle_deg.
    size is the overall bounding box size.
    width is the thickness of the cross arms.
    """
    cx, cy = pos
    rad = math.radians(-angle_deg)

    half_size = size * 3/4
    arm_thickness = width / 2

    # Cross is made from 2 rectangles:
    # 1) horizontal bar
    # 2) vertical bar
    rectangles = [
        # Horizontal bar: long in x, thin in y
        [
            (-half_size, -arm_thickness),
            (half_size, -arm_thickness),
            (half_size, arm_thickness),
            (-half_size, arm_thickness),
        ],
        # Vertical bar: thin in x, long in y
        [
            (-arm_thickness, -half_size),
            (arm_thickness, -half_size),
            (arm_thickness, half_size),
            (-arm_thickness, half_size),
        ],
    ]

    cos_a = math.cos(rad)
    sin_a = math.sin(rad)

    def rotate_point(x, y):
        rx = x * cos_a - y * sin_a
        ry = x * sin_a + y * cos_a
        return (cx + rx, cy + ry)

    for rect in rectangles:
        points = [rotate_point(x, y) for x, y in rect]
        pygame.draw.polygon(surf, colour, points)

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


def draw_balls(surf, balls: list[Ball], corners: list[Corner], state: ArenaState):
    radius = 8

    for ball in balls:
        x, y = field_to_screen(ball.position, corners)

        # draw vip ball
        if ball.is_vip:
            for glow_step in range(3, 0, -1):
                glow_radius = radius + glow_step * 3
                glow_size = glow_radius * 2

                glow_surface = pygame.Surface(
                    (glow_size, glow_size),
                    pygame.SRCALPHA,
                )

                pygame.draw.circle(
                    glow_surface,
                    (*C_VIP_GLOW, 60 - glow_step * 15),
                    (glow_radius, glow_radius),
                    glow_radius,
                )

                surf.blit(glow_surface, (x - glow_radius, y - glow_radius))

            pygame.draw.circle(surf, C_BALL_VIP, (x, y), radius)
            pygame.draw.circle(surf, (255, 230, 120), (x, y), radius, 2)
            continue

        # make ball blue if corner, red if edge, green if cross, brown if waypoint, white otherwise
        if ball.is_corner_ball():
            color = (30, 140, 220)
        elif ball.is_edge_ball()[0]:
            color = (200, 55, 55)
        elif ball.is_within_cross_zone(state.cross):
            color = (50, 168, 56)
        elif not ball.is_within_cross_zone(state.cross) and ball.is_within_waypoint_zone(state):
            color = (125, 67, 26)
        else:
            color = C_BALL

        pygame.draw.circle(surf, color, (x, y), radius)
        pygame.draw.circle(surf, C_BALL_OUTLINE, (x, y), radius, 1)


def draw_cross(surf, cross: Cross, corners: list[Corner]):
    draw_cross_bounding_box(surf, field_to_screen(cross.position, corners), cross.side_length*5, cross.orientation, C_CROSS)
    pygame.draw.circle(surf, C_CROSS_OUTLINE, field_to_screen(cross.position, corners), 6)
    pygame.draw.circle(surf, C_CROSS,         field_to_screen(cross.position, corners), 4)

def draw_waypoints(surf, cross: Cross, corners: list[Corner]):
    waypoints = get_cross_waypoints(cross)
    for point in waypoints:
        pygame.draw.circle(surf, C_CROSS_WAYPOINT, field_to_screen(point, corners), 4)

def draw_route_lines(surf, state: ArenaState, corners: list[Corner]):
    if state.robot is not None:
        robot_pos = field_to_screen(state.robot.position, corners)
    else:
        return
    try:
        if state.target_point is not None:
            waypoints = calculate_shortest_waypoint_path(state, state.target_point)
            waypoints.append(state.target_point)
        else:
            return
    except:
        return    

     # Build full path: robot -> waypoint 0 -> waypoint 1 -> ...
    path_points = [robot_pos] + [field_to_screen(p, corners) for p in waypoints]

    # Create a font for labels (system font, size 20)
    font = pygame.font.SysFont(None, 20)

    # Draw line from robot to first waypoint (if there's at least 1 waypoint)
    if len(path_points) >= 2:
        pygame.draw.line(surf, (0, 255, 0), path_points[0], path_points[1], 3)

    # Draw the rest of the path (waypoint 0 -> 1 -> 2 -> ...)
    if len(path_points) > 2:
        # path_points[1:] are the waypoints; draw lines between them
        waypoint_path = path_points[1:]
        pygame.draw.lines(surf, (0, 255, 0), False, waypoint_path, 3)

    # Draw points as circles and label them
    for i, pt in enumerate(path_points[1:]):
        # Draw point as a circle
        color = (0, 255, 255)  # robot in yellow, waypoints in cyan
        pygame.draw.circle(surf, color, pt, 6)

        # Draw label next to the point
        label_text = str(i)
        label_surf = font.render(label_text, True, (255, 255, 255))
        label_pos = (pt[0] + 10, pt[1] - 10)
        surf.blit(label_surf, label_pos)
    

def draw_robot(surf, robot: Robot, corners):
    x, y = field_to_screen(robot.position, corners)
    r    = 16
    # Body
    pygame.draw.circle(surf, C_ROBOT_OUTLINE, (x, y), r + 2)
    pygame.draw.circle(surf, C_ROBOT_BODY,    (x, y), r)
    # Orientation arrow (pygame y is flipped → negate angle)
    draw_arrow(surf, C_ROBOT_ARROW, (x, y), robot.orientation, r + 10, tip_size=7, width=2)

def field_to_screen(pos: tuple[float, float], corners: list[Corner]) -> tuple[int, int]:
    tl, tr, br, bl = [c.position for c in corners]
    x = int(lerp(tl[0], tr[0], pos[0] / ARENA_WIDTH_CM))
    y = int(lerp(bl[1], tl[1], pos[1] / ARENA_HEIGHT_CM))  # y flipped: 0 = bottom
    return (x, y)

def frame_to_surface(frame_bgr, target_size):
    small = cv2.resize(frame_bgr, target_size, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    return pygame.image.frombuffer(rgb.tobytes(), target_size, "RGB")

# ---------------------------------------------------------------------------
# Side panel
# ---------------------------------------------------------------------------

def draw_panel(surf, font_sm, font_md, font_lg,
               robot: Robot, balls: list[Ball], cross: Cross,
               corners: list[Corner], estimated_ball_count: int, 
               estimated_balls_in_robot: int, estimated_balls_delivered: int,
               all_balls_delivered: bool, state: ArenaState):
    #global _start_time#

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
    if robot is not None:
        heading("ROBOT")
        row("Position",    (floor(robot.position[0]), floor(robot.position[1])))
        row("Orientation", f"{robot.orientation:.1f}°")
        y += 8

    # Cross
    if cross is not None:
        heading("CROSS")
        row("Position",    (floor(cross.position[0]), floor(cross.position[1])))
        row("Orientation", f"{cross.orientation:.1f}°")
        y += 8

    # Balls
    heading(f"BALLS  ({len(balls)})")
    for i, b in enumerate(balls):
        tag = "VIP" if b.is_vip else f"#{i}"
        row(tag, (floor(b.position[0]), floor(b.position[1])))
    y += 8

    # Corners
    heading("CORNERS")
    labels = ["TL", "TR", "BR", "BL"]
    for lbl, c in zip(labels, corners):
        row(lbl, (floor(c.position[0]), floor(c.position[1])))
    y += 16

    # estimated ball count
    heading("Ball estimates")
    row("In arena", estimated_ball_count)
    row("In robot", estimated_balls_in_robot)
    row("Delivered", estimated_balls_delivered)
    y += 16

    # time passed
    heading("Time")
    if state.start_time is None:
        elapsed = 0
    elif state.finish_time is not None:
        elapsed = int(state.finish_time - state.start_time)
    else:
        elapsed = int(time() - state.start_time)

    row("Passed", f"{elapsed // 60:02d}:{elapsed % 60:02d}")

    return y

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_gui(state: ArenaState):
    #global _start_time

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("")
    clock  = pygame.time.Clock()
    #_start_time = time()

    tracker = ArenaTracker()

    font_sm = pygame.font.SysFont("monospace", 13)
    font_md = pygame.font.SysFont("monospace", 15, bold=True)
    font_lg = pygame.font.SysFont("monospace", 20, bold=True)

    # video feed config
    VIDEO_W = PANEL_W - FIELD_MARGIN // 2          # match panel content width
    VIDEO_H = int(VIDEO_W * 3 / 4)                 # 4:3
    VIDEO_X = WINDOW_W - PANEL_W + 15
    _last_frame_id = -1
    _video_surf = None

    t = 0.0

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        t += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                try:
                    path = tracker.dump_frame()
                    print(f"Saved {path}")
                except RuntimeError as e:
                    print(f"Can't dump yet: {e}")

        with state.lock:
            robot = state.robot
            balls = list(state.balls)
            cross = state.cross
            corners = list(state.corners)
            estimated_ball_count = state.estimated_ball_count
            estimated_balls_in_robot = state.estimated_balls_in_robot
            estimated_balls_delivered = state.estimated_balls_delivered

        # ------------------------------------------------------------------
        # Draw
        # ------------------------------------------------------------------
        screen.fill(C_BG)

        draw_field_surface(screen, corners)
        draw_borders(screen, corners)
        draw_corners(screen, corners)
        if cross is not None:
            draw_cross(screen, cross, corners)
            draw_waypoints(screen, cross, corners)
        draw_route_lines(screen, state, corners)
        draw_balls(screen, balls, corners, state)
        if robot is not None:
            draw_robot(screen, robot, corners)
        panel_end_y = draw_panel(screen, font_sm, font_md, font_lg, robot, balls, cross, corners,
                                  estimated_ball_count, estimated_balls_in_robot,
                                  estimated_balls_delivered, state.all_balls_delivered, state)

        for point in get_cross_approach_points(cross, CROSS_APPROACH_POINTS_HORIZONTAL_OFFSET, CROSS_APPROACH_POINTS_VERTICAL_OFFSET):
            pygame.draw.circle(screen, C_BALL_VIP, field_to_screen(point, corners), 5)
        for point in get_cross_approach_points(cross, CROSS_FINAL_APPROACH_HORIZONTAL_OFFSET, CROSS_FINAL_APPROACH_VERTICAL_OFFSET):
            pygame.draw.circle(screen, C_FINAL_POINTS, field_to_screen(point, corners), 5)

        # Video feed
        VIDEO_Y = panel_end_y + 10
        raw_frame, frame_id = tracker.get_latest_frame_with_id()
        if raw_frame is not None and frame_id != _last_frame_id:
            _video_surf = frame_to_surface(raw_frame, (VIDEO_W, VIDEO_H))
            _last_frame_id = frame_id
        if _video_surf is not None:
            screen.blit(_video_surf, (VIDEO_X, VIDEO_Y))
            pygame.draw.rect(screen, C_BORDER, (VIDEO_X, VIDEO_Y, VIDEO_W, VIDEO_H), 2)

        pygame.display.flip()

    pygame.quit()
    sys.exit()

def get_test_field_state():
    state = ArenaState()
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
    state.cross  = Cross(position=(500, 500), orientation=0, bounding_box=[(),(),(),()])
    state.robot  = Robot(position=(FIELD_X0 + 100, FIELD_Y0 + FIELD_H // 2),
                   orientation=35.0)
    return state