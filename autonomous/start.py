from model import state
from stateManager import update_state

def start_autonomous_session(state, logger):
    print("autonomous")
    update_state(state, logger)
    pass