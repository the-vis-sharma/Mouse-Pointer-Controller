from face_detection import FaceDetection
from facial_landmarks_detection import FacialLandmarksDetection
from head_pose_estimation import HeadPoseEstimation
from gaze_estimation import GazeEstimation
import cv2
from argparse import ArgumentParser
from input_feeder import InputFeeder

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

    feed=InputFeeder(input_type='video', input_file=args.input)
    feed.load_data()
    for batch in feed.next_batch():
        cv2.imshow("Output", cv2.resize(batch, (500, 500)))
        key = cv2.waitKey(60)

        if (key == 27):
            break

        # getting face
        faceDetection = FaceDetection(model_name=args.face_detection_model)
        faceDetection.load_model()
        face = faceDetection.predict(batch)

        # getting eyes
        facialLandmarksDetection = FacialLandmarksDetection(args.facial_landmarks_detection_model)
        facialLandmarksDetection.load_model()
        left_eye, right_eye = facialLandmarksDetection.predict(face)
        
        # getting head pose angles
        headPoseEstimation = HeadPoseEstimation(args.head_pose_estimation_model)
        headPoseEstimation.load_model()
        head_pose = headPoseEstimation.predict(face)
        print("head pose angles: ", head_pose)

        # get mouse points
        gazeEstimation = GazeEstimation(args.gaze_estimation_model)
        gazeEstimation.load_model()
        mouse_coords = gazeEstimation.predict(left_eye, right_eye, head_pose)
        print("gaze  output: ", mouse_coords)
    feed.close()   


def main():
    args = build_argparser().parse_args()

    run_inference(args)

if __name__ == "__main__":
    main()
