from __future__ import annotations
from .constants import POSE_DICT
from zed_interfaces.msg import BoundingBox2Di
from rclpy.impl.rcutils_logger import RcutilsLogger

class SystemState:
    _instance:SystemState = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.state = 1
            cls._instance.focus_body_id = None
            cls._instance.focus_body_bbox = None
            cls._instance._logger = None
        return cls._instance

    @property
    def logger(self) -> RcutilsLogger:
        return self._logger
    
    @logger.setter
    def logger(self, logger:RcutilsLogger):
        self._logger = logger

    def set_state(self, new_state):
        if self.state == new_state:
            return
        self.state = new_state
        self.logger.info(f"State changed to {new_state}")
        
    def set_focus_body_id(self, body_id):
        if self.focus_body_id == body_id:
            return
        self.focus_body_id = body_id
        self.logger.info(f"Focus id changed to {body_id}")
        
    def set_focus_body_bbox(self, bbox: BoundingBox2Di):
        if self.focus_body_bbox == bbox:
            return
        self.focus_body_bbox = bbox
        self.logger.info(f"Focus bbox changed to {bbox}")
    