from src.autonomous_mode.deliver_balls import deliver_balls
from src.lib.connection import RobotConnection
from src.model.arena_state import ArenaState
from src.debug.log import get_logger
from src.autonomous_mode.movement_helpers import _start_ball_intake, _turn_toward_point, _stop_ball_intake, drive_and_collect_ball
from src.autonomous_mode.state_helpers import await_robot, has_vip_balls, update_ball_count_estimate
from time import time
from protocol import Instruction, InstructionType, CommandName, Arguments, Message

_last_ball_count_update_time = 0 # time at which the estimated ball count was last updated

COLLECT_BALLS_PER_DELIVERY = 4

def start_autonomous_session(state: ArenaState) -> None:
    global _last_ball_count_update

    connection = RobotConnection()
    _start_ball_intake(connection)
    update_ball_count_estimate(state)
    _last_ball_count_update = time()

    collect_vip_balls(state, connection)
    collect_normal_balls(state, connection)
    _stop_ball_intake(connection)


def collect_vip_balls(state: ArenaState, connection: RobotConnection) -> None:
    logger = get_logger("collect_vip_balls")
    while True:
        _tick(state)

        robot = await_robot(state, connection)

        if not has_vip_balls(state):
            deliver_balls(state, connection)
            break

        logger.debug("Finding VIP ball to collect")
        vip = robot.get_nearest_vip_ball(state.balls)
        if vip is None: continue
        ball_point = vip.position
        print(vip.is_edge_ball())

        _turn_toward_point(state, connection, ball_point)

        #drive_and_collect_ball(robot, ball_point, connection, state)

        logger.debug("End of loop\n")


def collect_normal_balls(state: ArenaState, connection: RobotConnection) -> None:
    logger = get_logger("collect_normal_balls")
    total_normal_balls = update_ball_count_estimate(state)
    normal_balls_delivered = 0

    while True:
        _tick(state)
        balls_in_robot = total_normal_balls - state.estimated_ball_count - normal_balls_delivered

        if state.estimated_ball_count == 0 and balls_in_robot == 0:
            # done
            _stop_ball_intake(connection)
            _send_win_message(connection)
            logger.debug("All balls delivered, stopping.")
            break

        robot = await_robot(state, connection)

        if balls_in_robot >= COLLECT_BALLS_PER_DELIVERY or state.estimated_ball_count == 0:
            balls_in_arena_before_delivery = total_normal_balls - normal_balls_delivered
            logger.debug(f"Balls in robot: {balls_in_robot}, estimated ball count: {state.estimated_ball_count}. Commencing delivery")
            # COLLECT_BALLS_PER_DELIVERY balls in robot, deliver
            deliver_balls(state, connection)
            balls_in_arena_after_delivery = update_ball_count_estimate(state)
            normal_balls_delivered += balls_in_arena_before_delivery - balls_in_arena_after_delivery
            balls_in_robot = 0
            logger.debug(f"balls_in_robot: {balls_in_robot}, total_normal_balls: {total_normal_balls}, normal_balls_delivered: {normal_balls_delivered}, estimated_ball_count: {state.estimated_ball_count}, balls_in_arena_before_delivery: {balls_in_arena_before_delivery}, balls_in_arena_after_delivery: {balls_in_arena_after_delivery}")
            continue

        logger.debug("Finding ball to collect")
        nearest = robot.get_nearest_ball(state.balls)
        if nearest is None: continue
        ball_point = nearest.position

        _turn_toward_point(state, connection, ball_point)

        drive_and_collect_ball(robot, ball_point, connection, state)

        logger.debug("End of loop\n")

def _tick(state: ArenaState):
    global _last_ball_count_update

    if time() - 10 >= _last_ball_count_update:
        update_ball_count_estimate(state)
        _last_ball_count_update = time()

def _send_win_message(connection: RobotConnection):
    inst = Instruction(
        name=CommandName.TALK,
        type=InstructionType.COMMAND,
        args=Arguments(talk="I did it"),
    )
    connection.send_message(Message(instruction=inst))
