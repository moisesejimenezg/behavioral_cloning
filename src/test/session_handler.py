import pytest

from src.session_handler import SessionHandler

def test_reading_file():
    session_handler = SessionHandler("data/driving_log.csv")
    assert(len(session_handler.get_data()) == 3)
    assert(len(session_handler.get_left_images()) == 10)
    assert(len(session_handler.get_right_images()) == 10)
    assert(len(session_handler.get_measurements()) == 10)
