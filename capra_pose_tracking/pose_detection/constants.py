KEYPOINT_COUPLES = [[1, 2],
                    [2, 3],
                    [3, 4],
                    [1, 5],
                    [5, 6],
                    [6, 7]]

KEYPOINT_OF_INTEREST = [1, 2, 3, 4, 5, 6, 7]

FACE_KEYPOINTS = [0, 14, 15]

DATASET_PATH = "./dataset"

PREDICTION_OUTPUT_DICT = {
        0: "NO POSE",
        1: "T-POSE",
        2: "BUCKET",
        3: "SKYWARD" 
}

POSE_DICT = {
        "NO POSE": 0,
        "T-POSE": 1,
        "BUCKET": 2,
        "SKYWARD": 3
}

STATE_DICT = {
    POSE_DICT["T-POSE"]: 'IDLE',
    POSE_DICT["BUCKET"]: 'FOLLOW',
    POSE_DICT["SKYWARD"]: 'RETRACE'
}

MODEL_PATH = './trainings/training19/model/model19.keras'
CHECKSUM_PATH = './trainings/training19/model/weights_checksum19.txt'
