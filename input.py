from protocol import CommandName, SequenceName, InstructionType, Message, Arguments, Instruction

COMMAND_MAP = {
    # commands
    "fwd":  ("command", CommandName.FORWARD),
    "bwd":  ("command", CommandName.BACKWARD),
    "tl":   ("command", CommandName.TANK_LEFT),
    "tr":   ("command", CommandName.TANK_RIGHT),
    "bin":  ("command", CommandName.BALL_IN),
    "bout": ("command", CommandName.BALL_OUT),
    "boff": ("command", CommandName.BALL_OFF),
    "t":    ("command", CommandName.TALK),
    # sequences
    "bust": ("sequence", SequenceName.EJECT),
}

DEFAULT_ARG_OVERRIDE = {
    "fwd_slow": { "speed": 10 },
    "back_slow": { "speed": -10 },
}

def build_message_from_short_command(name, kwargs):
    """Build a Message from a short command name and user kwargs."""
    if name not in COMMAND_MAP:
        allowed = ", ".join(sorted(COMMAND_MAP))
        raise ValueError(f"Unknown command name {name}. Allowed: {allowed}")

    kind, enum_name = COMMAND_MAP[name]

    args = {}
    if name in DEFAULT_ARG_OVERRIDE:
        args.update(DEFAULT_ARG_OVERRIDE[name])
    args.update(kwargs)

    # Map user keys to protocol fields
    field_kw = {}
    for k in ["rspeed", "lspeed", "speed", "rotations", "position", "seconds",
              "target_angle", "brake", "block", "talk"]:
        if k in args:
            field_kw[k] = args[k]

    instruction_type = (
        InstructionType.COMMAND
        if kind == "command"
        else InstructionType.SEQUENCE
    )
    inst = Instruction(
        name=enum_name,
        type=instruction_type,
        args=Arguments(**field_kw),
    )
    return Message(instruction=inst)

def parse_input(line):
    """Parse "NAME key=value key2=value" into (name, kwargs)."""
    parts = line.strip().split()
    if not parts:
        raise ValueError("empty command")

    name = parts[0]

    kwargs = {}
    for p in parts[1:]:
        if "=" not in p:
            raise ValueError(f"Bad argument syntax: {p}; expected 'key=value'")
        k, v = p.split("=", 1)
        if v in ("True", "true"):
            v = True
        elif v in ("False", "false"):
            v = False
        else:
            try:
                if "." in v:
                    v = float(v)
                else:
                    v = int(v)
            except ValueError:
                pass  # leave as string
        kwargs[k] = v

    return name, kwargs