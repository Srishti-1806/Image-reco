"""Block 8 (edge side): act on the server's decision. Replace the stubs with GPIO / relay / PACS calls."""
import logging

log = logging.getLogger("dispatcher")


def open_gate(person_id: str):
    log.info("GATE OPEN for %s", person_id)      # e.g. Jetson.GPIO relay pulse


def local_alarm():
    log.warning("LOCAL ALARM")                   # e.g. buzzer / strobe


def handle(resp: dict | None):
    if resp is None:
        return
    if resp["action"] == "ACCESS_GRANTED":
        open_gate(resp["person_id"])
    elif resp["action"] == "ALERT":
        local_alarm()
