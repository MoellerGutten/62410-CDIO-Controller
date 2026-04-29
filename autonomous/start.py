from stateManager import update_state
from protocol import CommandName, Arguments, Instruction, InstructionType, Message, serialize_message
from connection import connect

def start_autonomous_session(state,  logger):
    print("autonomous")
    update_state(state, logger)

    socket = connect()

    ball = state.balls[0]
    while not state.robot.is_facing_point(ball.position, 5.0):
        inst = Instruction(
            name=CommandName.TURN_RIGHT,
            type=InstructionType.COMMAND,
            args=Arguments(),
        )
        msg = Message(instruction=inst)
        s = serialize_message(msg)
        socket.sendall(s.encode("utf-8"))

    while not state.robot.distance_to_point(ball.position) > 5:
        inst = Instruction(
            name=CommandName.FORWARD,
            type=InstructionType.COMMAND,
            args=Arguments(),
        )
        msg = Message(instruction=inst)
        s = serialize_message(msg)
        socket.sendall(s.encode("utf-8"))
