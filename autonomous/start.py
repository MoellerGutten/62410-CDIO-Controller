from stateManager import update_state
from protocol import CommandName, Arguments, Instruction, InstructionType, Message, serialize_message
from connection import connect


def start_autonomous_session(state, logger):
    print("autonomous")

    socket = connect()

    inst = Instruction(
        name=CommandName.BALL_IN,
        type=InstructionType.COMMAND,
        args=Arguments(seconds=500, speed=100),
    )
    msg = Message(instruction=inst)
    socket.sendall(serialize_message(msg).encode("utf-8"))

    # Get an initial snapshot before the loop
    update_state(state, logger)

    ball = state.balls[0]

    while not state.robot.is_facing_point(ball.position, 5.0):
        inst = Instruction(
            name=CommandName.TANK_RIGHT,
            type=InstructionType.COMMAND,
            args=Arguments(),
        )
        socket.sendall(serialize_message(Message(instruction=inst)).encode("utf-8"))
        update_state(state, logger)
        ball = state.balls[0]   # refresh target after each scan

    while not state.robot.distance_to_point(ball.position) > 5:
        inst = Instruction(
            name=CommandName.FORWARD,
            type=InstructionType.COMMAND,
            args=Arguments(),
        )
        socket.sendall(serialize_message(Message(instruction=inst)).encode("utf-8"))
        update_state(state, logger)
        ball = state.balls[0]   # refresh target after each scan