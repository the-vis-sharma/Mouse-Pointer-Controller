from face_detection import FaceDetection
from facial_landmarks_detection import FacialLandmarksDetection
from head_pose_estimation import HeadPoseEstimation
from gaze_estimation import GazeEstimation
import cv2
from argparse import ArgumentParser

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
    # getting face
    faceDetection = FaceDetection(model_name=args.face_detection_model)
    faceDetection.load_model()
    img = cv2.imread(args.input)
    face, face_coords = faceDetection.predict(img)
    cv2.imshow("Face", face)
    cv2.waitKey(0)

    # getting eyes
    facialLandmarksDetection = FacialLandmarksDetection(args.facial_landmarks_detection_model)
    facialLandmarksDetection.load_model()
    left_eye, right_eye, left_eye_coords, right_eye_coords = facialLandmarksDetection.predict(face)
    cv2.imshow("left eye", left_eye)
    cv2.waitKey(0)
    cv2.imshow("right eye", right_eye)
    cv2.waitKey(0)

    # getting head pose angles
    headPoseEstimation = HeadPoseEstimation(args.head_pose_estimation_model)
    headPoseEstimation.load_model()
    angles = headPoseEstimation.predict(face)
    print("head pose angles: ", angles)

    # get mouse points
    gazeEstimation = GazeEstimation(args.gaze_estimation_model)
    gazeEstimation.load_model()
    gazeEstimation.predict(left_eye, right_eye, angles)


def main():
    args = build_argparser().parse_args()

    run_inference(args)

if __name__ == "__main__":
    main()
