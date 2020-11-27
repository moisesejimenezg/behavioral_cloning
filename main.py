from src.model import Model
from src.session_handler import SessionHandler

path = "data/driving_log.csv"

session_handler = SessionHandler(path)

model = Model()
model.fit_model(session_handler.get_center_images(), session_handler.get_measurements())
