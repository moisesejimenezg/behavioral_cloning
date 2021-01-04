from src.model import Model
from src.session_handler import SessionHandler
import numpy as np
from os.path import exists

path = "data/driving_log.csv"

session_handler = SessionHandler(path)

model = Model()
if exists('model.h5'):
    model.load_model()
model.fit_model(
    np.concatenate(
        (
            session_handler.get_center_images(),
            session_handler.get_left_images(),
            session_handler.get_right_images(),
        )
    ),
    np.concatenate(
        (
            session_handler.get_measurements(),
            session_handler.get_measurements() + 0.2,
            session_handler.get_measurements() - 0.2,
        )
    ),
)
