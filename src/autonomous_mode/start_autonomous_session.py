from src.autonomous_mode.deliver_balls import deliver_balls
from src.lib.connection import RobotConnection
from src.model.arena_state import ArenaState
from src.debug.log import get_logger
from src.autonomous_mode.movement_helpers import _start_ball_intake, _turn_toward_point, adjust_heading
from src.autonomous_mode.state_helpers import _await_robot, drive_and_collect_ball, has_vip_balls, update_ball_count_estimate
from time import time

def start_autonomous_session(state: ArenaState) -> None:
    connection = RobotConnection()
    _start_ball_intake(connection)
    update_ball_count_estimate(state)
    last_ball_count_update = time()

    while True:
        # update ball count estimate every 10 seconds
        if time() - 10 >= last_ball_count_update:
            update_ball_count_estimate(state)
            last_ball_count_update = time()

        robot = _await_robot(state, connection)
        if robot is None:
            get_logger().warning("Robot not detected after nudge — retrying main loop")
            continue

        start_collecting_vip_balls(state, connection)

        # start_collecting_normal_balls(state, connection)
        
        break


def start_collecting_vip_balls(state: ArenaState, connection: RobotConnection) -> None:
    while True:
        robot = _await_robot(state, connection)
        if robot is None:
            get_logger().warning("Robot not detected after nudge — retrying main loop")
            continue

        if not has_vip_balls(state):
            if update_ball_count_estimate(state) > 3: continue
            deliver_balls(state, connection)
            break

        get_logger().debug("Finding VIP ball to collect")
        vip = robot.get_nearest_vip_ball(state.balls)
        if vip is None: continue
        ball_point = vip.position

        _turn_toward_point(state, connection, ball_point)

        drive_and_collect_ball(robot, ball_point, connection, state)

        get_logger().debug("End of loop\n")


def start_collecting_normal_balls(state: ArenaState, connection: RobotConnection) -> None:
    while True:
        robot = _await_robot(state, connection)
        if robot is None:
            get_logger().warning("Robot not detected after nudge — retrying main loop")
            continue

        if not state.balls:
            if update_ball_count_estimate(state) > 0: continue
            deliver_balls(state, connection)
            break

        get_logger().debug("Finding ball to collect")
        nearest = robot.get_nearest_ball(state.balls)
        if nearest is None:continue
        ball_point = nearest.position

        _turn_toward_point(state, connection, ball_point)

        drive_and_collect_ball(robot, ball_point, connection, state)

        get_logger().debug("End of loop\n")


