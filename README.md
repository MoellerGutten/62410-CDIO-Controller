# 62410-CDIO-Controller

### Configuration
To be able to connect to the robot, fill out `controller.config` in the project root like this

```
EV3_HOST=<ip>
EV3_PORT=<port>
```

### Commands
To run the controller, run `python -m main` (use `python3` instead of `python` if your system requires). Run with the flag `--it` to run the controller in interactive mode. Run with the flag `--gui` to use the debug gui.

Note that logs generated during controller execution will be persisted in `./logs`.

To calibrate the arena tracker, run `python -m src.state.arena_tracker` (use `python3` instead of `python` if your system requires).

### Unit tests
To run the unit tests in this repository, run `python3 -m unittest discover -s tests`. To run one single test file, run `python3 -m tests.<module>` (note that `<module>` refers to the name of the Python-file WITHOUT the `.py` extension).
