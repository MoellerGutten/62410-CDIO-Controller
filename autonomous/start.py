from stateManager import update_state
from protocol import CommandName, Arguments, Instruction, InstructionType, Message, serialize_message
from connection import connect
from stateManager import update_state

def start_autonomous_session(state,  logger):
    print("autonomous")
    update_state(state, logger)

    socket = connect()

    inst = Instruction(
            name=CommandName.BALL_IN,
            type=InstructionType.COMMAND,
            args=Arguments(seconds=500, speed=100),
        )
    msg = Message(instruction=inst)
    s = serialize_message(msg)
    socket.sendall(s.encode("utf-8"))

    ball = state.balls[0]

    while not state.robot.is_facing_point(ball.position, 5.0):
        inst = Instruction(
            name=CommandName.TANK_RIGHT,
            type=InstructionType.COMMAND,
            args=Arguments(),
        )
        msg = Message(instruction=inst)
        s = serialize_message(msg)
        socket.sendall(s.encode("utf-8"))
        update_state(state, logger)

    while not state.robot.distance_to_point(ball.position) > 5:
        inst = Instruction(
            name=CommandName.FORWARD,
            type=InstructionType.COMMAND,
            args=Arguments(),
        )
        msg = Message(instruction=inst)
        s = serialize_message(msg)
        socket.sendall(s.encode("utf-8"))
        update_state(state, logger)
