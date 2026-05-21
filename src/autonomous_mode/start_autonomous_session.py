from src.autonomous_mode.deliver_balls import deliver_balls
from src.state.state_manager import update_state
from protocol import CommandName, Arguments, Instruction, InstructionType, Message
from src.lib.connection import RobotConnection 
from logging import Logger
from src.model.arena_state import ArenaState
from time import sleep

def start_autonomous_session(state: ArenaState, logger: Logger) -> None:
    connection = RobotConnection()

    inst = Instruction(
        name=CommandName.BALL_IN,
        type=InstructionType.COMMAND,
        args=Arguments(seconds=500, speed=100),
    )
    message = Message(instruction=inst)
    connection.send_message(message)

    # Get an initial snapshot before the loop
    update_state(state, logger)

    ball = state.balls[0] if len(state.balls) > 0 else None

    while True:

        # Check initial ball count (to avoid duplicate work) and if no balls
        # take 'x' amount of pictures to make sure no balls are left
        if ball is None:
            has_balls = False
            for _ in range(3):
                update_state(state, logger)
                has_balls = True if len(state.balls) > 0 else False
                if has_balls:
                    break
            # If no balls are left, drive to goal and bust
            if not has_balls:
                # deliver_balls(state, connection, logger)
                
                # for now just stop instead of attempting delivery
                inst = Instruction(
                    name=CommandName.PANIC,
                    type=InstructionType.COMMAND,
                    args=Arguments(),
                )
                connection.send_message(Message(instruction=inst))
                break

        if ball is None:
            update_state(state, logger)
            ball = state.balls[0] if len(state.balls) > 0 else None   # refresh target after each scan
            continue

        if state.robot is None:
            update_state(state, logger)
            continue

        while ball is not None and state.robot is not None and not state.robot.is_facing_point(ball.position, 3.0):
            angle_to_point = state.robot.angle_to_point(ball.position)
            turn_ms = max(100, min(300, int(abs(angle_to_point) * 10))) / 1000
            turn_speed = max(10, min(20, int(abs(angle_to_point) * 0.4)))

            if angle_to_point > 0:
                inst = Instruction(
                    name=CommandName.TANK_RIGHT,
                    type=InstructionType.COMMAND,
                    args=Arguments(seconds=turn_ms, lspeed=turn_speed, rspeed=-turn_speed),
                )
            else:
                inst = Instruction(
                    name=CommandName.TANK_LEFT,
                    type=InstructionType.COMMAND,
                    args=Arguments(seconds=turn_ms, lspeed=-turn_speed, rspeed=turn_speed),
                )

            connection.send_message(Message(instruction=inst))
            sleep(turn_ms / 1000 + 0.05)  # vent til robotten er færdig + lille buffer
            update_state(state, logger)
            ball = state.balls[0] if state.balls else None
            if ball is None:
                break

        if ball is not None:
            distance = state.robot.distance_to_point(ball.position)
            fwd_ms = max(100, min(500, int(distance * 5))) / 1000
            fwd_speed = max(10, min(50, int(distance)))
            print(f"distance: {distance}, fwd_speed: {fwd_speed}, sleep for: {fwd_ms / 1000} seconds")
            inst = Instruction(
                name=CommandName.FORWARD,
                type=InstructionType.COMMAND,
                args=Arguments(seconds=fwd_ms,speed=fwd_speed),
            )
            connection.send_message(Message(instruction=inst))
            sleep(fwd_ms / 1000 + 0.05)
        update_state(state, logger)
        ball = state.balls[0] if len(state.balls) > 0 else None   # refresh target after each scan
