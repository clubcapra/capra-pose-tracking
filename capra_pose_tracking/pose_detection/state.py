from .constants import POSE_DICT

class SystemState:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.state = 1
            cls._instance.focus_body_id = None
            cls._instance.focus_body_bbox = None
        return cls._instance

    def set_state(self, new_state):
        self.state = new_state
        
    def set_focus_body_id(self, body_id):
        self.focus_body_id = body_id
        
    def set_focus_body_bbox(self, bbox):
        self.focus_body_bbox = bbox
    