import numpy as np
import pandas as pd
from ..tools import data
from ..constants import FACE_KEYPOINTS, POSE_DICT
from ..tools.person import Person
from ..state import SystemState
from zed_interfaces.msg import ObjectsStamped

class PoseDetector():
    def __init__(self, model) -> None:
        self.model = model
        self.persons = {}
        self.system_state = SystemState()
        self.confidence_threshold = 0.8

    def clean_persons(self, bodies: list):
        id=[]
        ids = [body.label_id for body in bodies]
        return {
            k: v 
            for k, v in self.persons.items() if k in ids
        }

    def clear_persons_except(self, id: int):
        return {k: v for k, v in self.persons.items() if k==id}

    def infere(self, body):
        keypoints = data.getKeypointsOfInterestFromBodyData(body.skeleton_2d.keypoints)
        predictions = self.model.call(keypoints)
        max_idx = np.argmax(predictions)
    
        return max_idx, predictions[0][max_idx]
    
    def get_body_data_from_id(self, bodies, id):
        for body in bodies:
            if(id == body.label_id): return body
        return None
        
    def detect(self, bodies: ObjectsStamped):
        self.persons = self.clean_persons(bodies)
        
        if self.system_state.state != POSE_DICT["T-POSE"]:
            if self.system_state.focus_body_id in self.persons:

                person = self.persons[self.system_state.focus_body_id]
                body = ObjectsStamped
                self.get_body_data_from_id(body, self.system_state.focus_body_id)
                self.system_state.set_focus_body_bbox(body.bounding_box_2d)

                if not any(np.isnan(body.skeleton_2d.keypoints[id]).any() for id in FACE_KEYPOINTS):

                    prediction, confidence = self.infere(body)

                    if confidence > self.confidence_threshold:

                            focused_id = person.add_pose(prediction)

                            if focused_id > -1:
                                self.system_state.set_state(person.pose)
                                if person.pose not in [0,1]:
                                    self.system_state.set_focus_body_id(focused_id)                                        
                                else:
                                    self.system_state.set_focus_body_id(None)

                                person.add_pose(0)
                
            else:
                self.system_state.set_state(POSE_DICT["T-POSE"])
                self.system_state.set_focus_body_id(None)
            
        else:
            for body in bodies:

                if body.label_id not in self.persons:
                    self.persons[body.label_id] = Person(body.label_id)

                person = self.persons[body.label_id]

                if not any(pd.isnull(body.skeleton_2d.keypoints[int(id)]) for id in FACE_KEYPOINTS):

                    prediction, confidence = self.infere(body)

                    if confidence > self.confidence_threshold:

                        focused_id = person.add_pose(prediction)

                        if focused_id > -1:
                            self.system_state.set_state(person.pose)
                            if person.pose != 1:
                                self.system_state.set_focus_body_id(focused_id)
                            person.add_pose(0)
                    
                else:
                    self.persons[body.label_id].add_pose(POSE_DICT["NO POSE"])
        
        return self.persons