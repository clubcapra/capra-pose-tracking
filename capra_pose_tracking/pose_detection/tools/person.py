from time import time

class Person():
    def __init__(self, id):
        self.pose = 0
        self.start_time = None
        self.focus = False
        self.id = id

    def add_pose(self, pose_id):
        # If person already has an associated pose
        if self.pose != 0:
            # If the new pose is different from pose at previous frame
            if self.pose != pose_id:
                # We cancel the tracking and reset the time
                self.start_time = None
                self.pose = 0
            # If the new pose is the same as pose from previous frame
            else:
                # Calculate how long the pose has been being detected for
                elapsed_time = time() - self.start_time
                # If time is greater than threshold
                if elapsed_time > 4:
                    if pose_id not in [0, 1]:
                        # We set the focus on the person
                        self.focus = True
                        # And return the ID for main()
                    else:
                        self.focus = False
                        
                    return self.id
        # If person does not have an associated pose
        elif pose_id != 0:
            # Take not of start time of pose
            self.start_time = time()
            # Remember the pose
            self.pose = pose_id

        return -1

# class Person():
#     def __init__(self, id):
#         self.small_stack = np.full(10, None)
#         self.pose = None
#         self.start_time = None
#         self.big_stack = None
#         self.interest = False
#         self.focus = False
#         self.id = id
#         self.system_state = SystemState()

#     def add_pose(self, pose_id):
#         if self.interest:
#             self.big_stack = shift(self.big_stack, 1, cval=pose_id)
#             self.check_big_stack()
#         else:
#             self.small_stack = shift(self.small_stack, 1, cval=pose_id)
#             self.check_small_stack()

#     def check_big_stack(self):
#         if np.all(self.big_stack == self.big_stack[0] and self.big_stack[0] == POSE_DICT.fromkeys("NO POSE")):
#             self.focus = True
#             self.system_state.set_state(POSE_DICT.get(self.big_stack[0]))
#             self.big_stack = None

    
#     def check_small_stack(self):
#         if np.all(self.small_stack == self.small_stack[0] and self.small_stack[0] == POSE_DICT.fromkeys("NO POSE")):
#             self.big_stack = np.full(100, None)
#             self.big_stack = shift(self.big_stack, 10, cval=self.small_stack[0])
#             self.interest = True

#     def get_pose(self):
#         if self.interest:
#             return self.big_stack[0]
        
#         return self.small_stack[0]
    
#     def get_id(self):
#         return self.id
    
# class Person():
#     def __init__(self, id):
#         self.small_stack = np.full(10, None)
#         self.pose = None
#         self.start_time = None
#         self.big_stack = None
#         self.interest = False
#         self.focus = False
#         self.id = id
#         self.system_state = SystemState()

#     def add_pose(self, pose_id):
#         if self.interest:
#             self.big_stack = shift(self.big_stack, 1, cval=pose_id)
#             self.check_big_stack()
#         else:
#             self.small_stack = shift(self.small_stack, 1, cval=pose_id)
#             self.check_small_stack()

#     def check_big_stack(self):
#         if np.all(self.big_stack == self.big_stack[0] and self.big_stack[0] == POSE_DICT.fromkeys("NO POSE")):
#             self.focus = True
#             self.system_state.set_state(POSE_DICT.get(self.big_stack[0]))
#             self.big_stack = None

    
#     def check_small_stack(self):
#         if np.all(self.small_stack == self.small_stack[0] and self.small_stack[0] == POSE_DICT.fromkeys("NO POSE")):
#             self.big_stack = np.full(100, None)
#             self.big_stack = shift(self.big_stack, 10, cval=self.small_stack[0])
#             self.interest = True

#     def get_pose(self):
#         if self.interest:
#             return self.big_stack[0]
        
#         return self.small_stack[0]
    
#     def get_id(self):
#         return self.id
    




