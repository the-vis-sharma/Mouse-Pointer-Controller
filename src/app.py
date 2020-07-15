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
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Type of device like CPU, GPU, MYRIAD or FPGA. Default is CPU.")
    parser.add_argument("-ce", "--cpu_extension", required=False, type=str,
                        default=None, help="Path for CPU extension")
    parser.add_argument("-v", "--visualizers", required=False, nargs='+',
                        default=[], help="Enter one or more model symbol like f, l, hp and g.")
    return parser

def visualize_face(image, face_coords):
    # draw face boundary
    color = (255, 0, 0)
    width = 4
    pt1 = (face_coords[0], face_coords[1])
    pt2 = (face_coords[2], face_coords[3])
    cv2.rectangle(image, pt1, pt2, color, width)
    return image

def visualize_eyes(image, face, face_coords, left_eye_coords, right_eye_coords):
    # draw left eye and right eye boundary
    color = (255, 0, 0)
    width = 4

    left_eye_pt1 = (left_eye_coords[0], left_eye_coords[1])
    left_eye_pt2 = (left_eye_coords[2], left_eye_coords[3])
    cv2.rectangle(face, left_eye_pt1, left_eye_pt2, color, width)

    right_eye_pt1 = (right_eye_coords[0], right_eye_coords[1])
    right_eye_pt2 = (right_eye_coords[2], right_eye_coords[3])
    cv2.rectangle(face, right_eye_pt1, right_eye_pt2, color, width)

    image[face_coords[1]:face_coords[3], face_coords[0]:face_coords[2]] = face
    return image

def visualize_head_pose(image, head_pose):
    # display head pose info
    color = (255, 0, 0)
    width = 4
    fontScale = 1.5
    pt = (10, 40)
    msg = "Head Pose Output: Yaw: {:.2f}, Pitch: {:.2f}, Roll: {:.2f}".format(head_pose[0], head_pose[1], head_pose[2])
    cv2.putText(image, msg, pt, cv2.FONT_HERSHEY_COMPLEX, fontScale, color, width)
    return image

def visualize_gaze(image, x, y):
    # visualize gaze direction
    x, y = x * 100, y * 100
    y = -y
    pt = (10, 120)
    pt1 = (400, 100)
    pt2 = (int(400 + x), int(100 + y))
    width = 4
    fontScale = 1.5
    textColor = (0, 255, 0)
    lineColor = (255, 0, 255)
    msg = "Gaze Output:"
    cv2.putText(image, msg, pt, cv2.FONT_HERSHEY_COMPLEX, fontScale, textColor, width)
    cv2.arrowedLine(image, pt1, pt2, lineColor, width, tipLength = 0.5) # green
    return image

def run_inference(args):
    logger.debug("visualizers: ", args.visualizers)
    device = args.device
    cpu_ext = args.cpu_extension
    mouseController = MouseController("medium", "fast")
    loading_start_time = time.time()
    faceDetection = FaceDetection(model_name=args.face_detection_model, device=device, extensions=cpu_ext)
    faceDetection.load_model()
    
    facialLandmarksDetection = FacialLandmarksDetection(args.facial_landmarks_detection_model, device=device, extensions=cpu_ext)
    facialLandmarksDetection.load_model()

    headPoseEstimation = HeadPoseEstimation(args.head_pose_estimation_model, device=device, extensions=cpu_ext)
    headPoseEstimation.load_model()

    gazeEstimation = GazeEstimation(args.gaze_estimation_model, device=device, extensions=cpu_ext)
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

        # getting face
        face, face_coords = faceDetection.predict(batch)
        total_infer_time += faceDetection.total_infer_time
        if "f" in args.visualizers:
            batch = visualize_face(batch, face_coords)
        
        # getting eyes
        left_eye, right_eye, left_eye_coords, right_eye_coords = facialLandmarksDetection.predict(face)
        total_infer_time += facialLandmarksDetection.total_infer_time
        if "l" in args.visualizers:
            batch = visualize_eyes(batch, face, face_coords, left_eye_coords, right_eye_coords)
        
        # getting head pose angles
        head_pose = headPoseEstimation.predict(face)
        total_infer_time += headPoseEstimation.total_infer_time
        if "hp" in args.visualizers:
            batch = visualize_head_pose(batch, head_pose)
        
        # get mouse points
        mouse_coords, result = gazeEstimation.predict(left_eye, right_eye, head_pose)
        total_infer_time += gazeEstimation.total_infer_time
        if "g" in args.visualizers:
            batch = visualize_gaze(batch, mouse_coords[0], mouse_coords[1]) 
        mouseController.move(mouse_coords[0], mouse_coords[1])

        cv2.imshow("Output", cv2.resize(batch, (500, 500)))
        key = cv2.waitKey(60)

        if (key == 27):
            break
    feed.close() 
    logger.info("total inference time: {}".format(total_infer_time)) 


def main():
    args = build_argparser().parse_args()
    run_inference(args)

if __name__ == "__main__":
    main()
