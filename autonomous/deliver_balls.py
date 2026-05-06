from protocol import CommandName, Arguments, Instruction, InstructionType, Message, serialize_message
from debug.gui import FIELD_H
from model.state import FieldState
from stateManager import _get_tracker, update_state
from connection import connect 


def deliver_balls(state: FieldState, logger):
    socket = connect()
    tracker = _get_tracker()
    result = tracker.scan()
    # Goal point is middle point of goal a
    goalx = sum(p.x for p in result.goal_a) / len(result.goal_a) if result.goal_a else None
    goaly = sum(p.y for p in result.goal_a) / len(result.goal_a) if result.goal_a else None
    goal = [goalx, goaly]
    print("Target goal coords: " + str(goal))
    print("Intermediate target point (robot.x, FIELD_H/2): " + str([state.robot.position[0], FIELD_H/2]))
    # TODO: add support for shortest angle (driving backward (ask Merian))
    # Sequence should be (excluding cross problem):
    # Aim for y = arena_h/2, i.e. straight up or down from robot (turn first ofc)
    # Drive to goal
    # Eject balls

    # Turning toward point (robot.x, arena_h/2)
    target_point = [state.robot.position[0], FIELD_H/2]
    while not state.robot.is_facing_point(target_point, 5.0):
        angle_to_point = state.robot.angle_to_point(target_point)
        if (angle_to_point > 0):
            inst = Instruction(
                name=CommandName.TANK_RIGHT,
                type=InstructionType.COMMAND,
                args=Arguments(seconds=1,lspeed=-10,rspeed=10),
            )
            s = serialize_message(Message(instruction=inst))
            socket.sendall(s.encode("utf-8"))
        else:
            inst = Instruction(
                name=CommandName.TANK_LEFT,
                type=InstructionType.COMMAND,
                args=Arguments(seconds=1,lspeed=10,rspeed=-10),
            )
            s = serialize_message(Message(instruction=inst))
            socket.sendall(s.encode("utf-8"))
        # Update state for next turn
        update_state(state, logger)

    # Drives toward target point
    while state.robot.distance_to_point(target_point) > 1:
        inst = Instruction(
        name=CommandName.FORWARD,
        type=InstructionType.COMMAND,
        args=Arguments(seconds=1,speed=50),
        )
        s = serialize_message(Message(instruction=inst))
        socket.sendall(s.encode("utf-8"))
        print(s)
        # Update state for next turn
        update_state(state, logger)
    
    # Turning toward goal
    while not state.robot.is_facing_point(goal, 5.0):
        angle_to_point = state.robot.angle_to_point(target_point)
        if (angle_to_point > 0):
            inst = Instruction(
                name=CommandName.TANK_RIGHT,
                type=InstructionType.COMMAND,
                args=Arguments(seconds=1,lspeed=-10,rspeed=10),
            )
            s = serialize_message(Message(instruction=inst))
            socket.sendall(s.encode("utf-8"))
        else:
            inst = Instruction(
                name=CommandName.TANK_LEFT,
                type=InstructionType.COMMAND,
                args=Arguments(seconds=1,lspeed=10,rspeed=-10),
            )
            s = serialize_message(Message(instruction=inst))
            socket.sendall(s.encode("utf-8"))
        # Update state for next turn
        update_state(state, logger)

    # Drives toward goal
    while state.robot.distance_to_point(goal) > 1:
        inst = Instruction(
        name=CommandName.FORWARD,
        type=InstructionType.COMMAND,
        args=Arguments(seconds=1,speed=50),
        )
        s = serialize_message(Message(instruction=inst))
        socket.sendall(s.encode("utf-8"))
        print(s)
        # Update state for next turn
        update_state(state, logger)

    # Bust
    inst = Instruction(
        name=CommandName.EJECT,
        type=InstructionType.SEQUENCE,
        args=Arguments(speed=100),
    )
    s = serialize_message(Message(instruction=inst))
    socket.sendall(s.encode("utf-8"))
    print(s)