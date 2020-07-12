from face_detection import FaceDetection
from facial_landmarks_detection import FacialLandmarksDetection
from head_pose_estimation import HeadPoseEstimation
from gaze_estimation import GazeEstimation
import cv2
from argparse import ArgumentParser
from input_feeder import InputFeeder
from mouse_controller import MouseController
import time
import logging as logger

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-f", "--face_detection_model", required=True, type=str,
                        help="Path of Face Detection model xml file.") 
    parser.add_argument("-l", "--facial_landmarks_detection_model", required=True, type=str,
                        help="Path of Facial Landmarks Detection model xml file.")
    parser.add_argument("-hp", "--head_pose_estimation_model", required=True, type=str,
                        help="Path of Head Pose Estimation model xml file.")
    parser.add_argument("-g", "--gaze_estimation_model", required=True, type=str,
                        help="Path of Gaze Estimation model xml file.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path of video file or Enter cam for webcam feed")
    return parser

def run_inference(args):

    mouseController = MouseController("medium", "fast")
    loading_start_time = time.time()
    faceDetection = FaceDetection(model_name=args.face_detection_model)
    faceDetection.load_model()
    
    facialLandmarksDetection = FacialLandmarksDetection(args.facial_landmarks_detection_model)
    facialLandmarksDetection.load_model()

    headPoseEstimation = HeadPoseEstimation(args.head_pose_estimation_model)
    headPoseEstimation.load_model()

    gazeEstimation = GazeEstimation(args.gaze_estimation_model)
    gazeEstimation.load_model()
    total_loading_time = time.time() - loading_start_time
    logger.info("Total loading time for models: ", total_loading_time)

    feed = None
    if args.input.lower() == "cam":
        feed = InputFeeder(input_type="cam")
    else:
        feed = InputFeeder(input_type='video', input_file=args.input)
        
    feed.load_data()
    total_infer_time = 0
    for flag, batch in feed.next_batch():
        if not flag:
            break

        cv2.imshow("Output", cv2.resize(batch, (500, 500)))
        key = cv2.waitKey(60)

        if (key == 27):
            break

        # getting face
        face = faceDetection.predict(batch)
        total_infer_time += faceDetection.total_infer_time
        
        # getting eyes
        left_eye, right_eye = facialLandmarksDetection.predict(face)
        total_infer_time += facialLandmarksDetection.total_infer_time
        
        # getting head pose angles
        head_pose = headPoseEstimation.predict(face)
        total_infer_time += headPoseEstimation.total_infer_time
        
        # get mouse points
        mouse_coords = gazeEstimation.predict(left_eye, right_eye, head_pose)
        total_infer_time += gazeEstimation.total_infer_time
        
        mouseController.move(mouse_coords[0], mouse_coords[1])
    feed.close() 
    logger.info("total inference time: {}".format(total_infer_time)) 


def main():
    args = build_argparser().parse_args()

    run_inference(args)

if __name__ == "__main__":
    main()
