from src.autonomous_mode.deliver_balls import deliver_balls
from src.state.state_manager import update_state
from protocol import CommandName, Arguments, Instruction, InstructionType, Message
from src.lib.connection import RobotConnection 
from logging import Logger
from src.model.state import FieldState

def start_autonomous_session(state: FieldState, logger: Logger) -> None:
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
                deliver_balls(state, connection, logger)

        if ball is None:
            update_state(state, logger)
            ball = state.balls[0] if len(state.balls) > 0 else None   # refresh target after each scan
            continue
        while not state.robot.is_facing_point(ball.position, 5.0):
            angle_to_point = state.robot.angle_to_point(ball.position)
            if (angle_to_point > 0):
                inst = Instruction(
                    name=CommandName.TANK_RIGHT,
                    type=InstructionType.COMMAND,
                    args=Arguments(seconds=1,lspeed=10,rspeed=-10),
                )
                connection.send_message(Message(instruction=inst))
            else:
                inst = Instruction(
                    name=CommandName.TANK_LEFT,
                    type=InstructionType.COMMAND,
                    args=Arguments(seconds=1,lspeed=-10,rspeed=10),
                )
                connection.send_message(Message(instruction=inst))
            update_state(state, logger)
            ball = state.balls[0] if len(state.balls) > 0 else None   # refresh target after each scan

        
        inst = Instruction(
            name=CommandName.FORWARD,
            type=InstructionType.COMMAND,
            args=Arguments(seconds=1,speed=50),
        )
        connection.send_message(Message(instruction=inst))
        update_state(state, logger)
        ball = state.balls[0] if len(state.balls) > 0 else None   # refresh target after each scan
