import numpy as np
from itertools import product
import depthai as dai
from math import gcd
from pathlib import Path
from FPS import FPS, now
import cv2
import os, sys, re


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_YUNET_MODEL = str(SCRIPT_DIR / "models/face_detection_yunet_180x320_sh4.blob")

def find_isp_scale_params(size, is_height=True):
    """
    Find closest valid size close to 'size' and and the corresponding parameters to setIspScale()
    This function is useful to work around a bug in depthai where ImageManip is scrambling images that have an invalid size
    is_height : boolean that indicates if the value is the height or the width of the image
    Returns: valid size, (numerator, denominator)
    """
    # We want size >= 288
    if size < 288:
        size = 288
 
    
    # We are looking for the list on integers that are divisible by 16 and
    # that can be written like n/d where n <= 16 and d <= 63
    if is_height:
        reference = 1080 
        other = 1920
    else:
        reference = 1920 
        other = 1080
    size_candidates = {}
    for s in range(16,reference,16):
        f = gcd(reference, s)
        n = s//f
        d = reference//f
        if n <= 16 and d <= 63 and int(round(other * n / d) % 2 == 0):
            size_candidates[s] = (n, d)
            
    # What is the candidate size closer to 'size' ?
    min_dist = -1
    for s in size_candidates:
        dist = abs(size - s)
        if min_dist == -1:
            min_dist = dist
            candidate = s
        else:
            if dist > min_dist: break
            candidate = s
            min_dist = dist
    return candidate, size_candidates[candidate]


class YuNet:
    """
    YuNet Face Detector : https://github.com/opencv/opencv_zoo/tree/dev/models/face_detection_yunet
    Arguments:
    - model: path to Yunet blob
    - model_resolution: None or string "HxW" where H and W are the Yunet input resolution (Height, Width)
            If None, the resolution is inferred from the model path "face_detection_yunet_HxW.blob"
    - input_src: frame source, 
                    - "rgb" or None: OAK* internal color camera,
                    - "rgb_laconic": same as "rgb" but without sending the frames to the host,
                    - a file path of an image or a video,
                    - an integer (eg 0) for a webcam id,
    - conf_threshold: detection score threshold [0..1],
    - nms_threshold:  Non Maximal Suppression threshold [0..1],
    - internal_fps : when using the internal color camera as input source, set its FPS to this value (calling setFps()).
    - internal_frame_height : when using the internal color camera, set the frame height (calling setIspScale()).
                                The width is calculated accordingly to height and depends on value of 'crop'
    - stats : boolean, when True, display some statistics when exiting.   
    - trace: boolean, when True print some debug messages   
    """
    def __init__(self, 
                model = str(DEFAULT_YUNET_MODEL),
                model_resolution=None, 
                input_src=None,                 
                conf_threshold=0.6, 
                nms_threshold=0.3, 
                top_k = 50,
                internal_fps=50,
                internal_frame_height=640,
                stats=False,
                trace=False,
                ):

        self.model = model
        if not os.path.isfile(model):
            print(f"Model path '{model}' does not exist !!!")
            sys.exit()

        if model_resolution is None: model_resolution = model

        # Try to infer from the model path
        
        match = re.search(r'.*?(\d+)x(\d+).*', model)
        if not match:
            print(f"Impossible to infer the model input resolution from model name '{model}' does not exist !!!")      
            sys.exit()
        self.nn_input_w = int(match.group(2))
        self.nn_input_h = int(match.group(1))
        print(f"Model : {self.model} - Input resolution: {self.nn_input_h}x{self.nn_input_w}")

        self.internal_fps = internal_fps
        
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.stats = stats
        self.trace = trace

        self.min_sizes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
        self.steps = [8, 16, 32, 64]
        self.variance = [0.1, 0.2]

        # Generate priors
        self.prior_gen()

        self.device = dai.Device()

        if input_src is None or input_src == "rgb" or input_src == "rgb_laconic":
            self.input_type = "rgb" # OAK* internal color camera
            self.laconic = input_src == "rgb_laconic" # Camera frames are not sent to the host
            self.video_fps = self.internal_fps # Used when saving the output in a video file. Should be close to the real fps

            width, self.scale_nd = find_isp_scale_params(internal_frame_height * 1920 / 1080, is_height=False)
            self.img_h = int(round(1080 * self.scale_nd[0] / self.scale_nd[1]))
            self.img_w = int(round(1920 * self.scale_nd[0] / self.scale_nd[1]))
            print(f"Internal camera image size: {self.img_w} x {self.img_h}")

        elif input_src.endswith('.jpg') or input_src.endswith('.png') :
            self.input_type= "image"
            self.img = cv2.imread(input_src)
            self.video_fps = 25
            self.img_h, self.img_w = self.img.shape[:2]
        else:
            self.input_type = "video"
            if input_src.isdigit():
                input_type = "webcam"
                input_src = int(input_src)
            self.cap = cv2.VideoCapture(input_src)
            self.video_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            self.img_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.img_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print("Video FPS:", self.video_fps)
        

        # We want to keep aspect ratio of the input images
        # So we may need to pad the images before feeding them to the model
        # 'padded_size' is the size of the image once padded. 
        # Note that the padding when used is not applied on both sides (top and bottom, 
        # or left and right) but only on the side opposite to the origin (top or left). 
        # It makes calculations easier.
        self.iwnh_ihnw = self.img_w * self.nn_input_h / (self.img_h * self.nn_input_w) 
        if self.iwnh_ihnw >= 1: 
            self.padded_size = np.array((self.img_w, self.img_h * self.iwnh_ihnw)).astype(int)
        else:
            self.padded_size = np.array((self.img_w / self.iwnh_ihnw, self.img_h)).astype(int)
        print(f"Source image size: {self.img_w} x {self.img_h}")
        print(f"Padded image size: {self.padded_size[0]} x {self.padded_size[1]}")

        # Define and start pipeline
        usb_speed = self.device.getUsbSpeed()
        self.device.startPipeline(self.create_pipeline())
        print(f"Pipeline started - USB speed: {str(usb_speed).split('.')[-1]}")

        # Define data queues 
        if self.input_type == "rgb":
            if not self.laconic:
                self.q_video = self.device.getOutputQueue(name="cam_out", maxSize=1, blocking=False)
            if self.trace:
                self.q_manip_out = self.device.getOutputQueue(name="manip_out", maxSize=1, blocking=False)
        else:
            self.q_nn_in = self.device.getInputQueue(name="nn_in")
        self.q_nn_out = self.device.getOutputQueue(name="nn_out", maxSize=4, blocking=False)

        self.fps = FPS()   
        self.glob_rtrip_time = 0 
        self.glob_posprocessing_time = 0

    def create_pipeline(self):
        print("Creating pipeline...")
        # Start defining a pipeline
        pipeline = dai.Pipeline()
        pipeline.setOpenVINOVersion(version = dai.OpenVINO.Version.VERSION_2021_4)

        if self.input_type == "rgb":
            # ColorCamera
            print("Creating Color Camera...")
            cam = pipeline.createColorCamera()
            cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            cam.setBoardSocket(dai.CameraBoardSocket.RGB)
            cam.setInterleaved(False)
            cam.setIspScale(self.scale_nd[0], self.scale_nd[1])
            cam.setFps(self.internal_fps)
            cam.setPreviewSize(self.img_w, self.img_h)

            if not self.laconic:
                cam_out = pipeline.createXLinkOut()
                cam_out.setStreamName("cam_out")
                cam_out.input.setQueueSize(1)
                cam_out.input.setBlocking(False)
                cam.video.link(cam_out.input)

            # The frame is padded to have the same ratio width/height 
            # as the model input, and resized to the model input resolution
            print("Creating Image Manip node...")
            manip = pipeline.createImageManip()
            manip.setMaxOutputFrameSize(self.nn_input_w*self.nn_input_h*3)
            manip.inputImage.setQueueSize(1)
            manip.inputImage.setBlocking(False)
            points = [
                    [0, 0],
                    [self.padded_size[0], 0],
                    [self.padded_size[0], self.padded_size[1]],
                    [0, self.padded_size[1]]]
            point2fList = []
            for p in points:
                pt = dai.Point2f()
                pt.x, pt.y = p[0], p[1]
                point2fList.append(pt)
            manip.initialConfig.setWarpTransformFourPoints(point2fList, False)
            manip.initialConfig.setResize(self.nn_input_w, self.nn_input_h)

            cam.preview.link(manip.inputImage)

            # For debugging
            if self.trace:
                manip_out = pipeline.createXLinkOut()
                manip_out.setStreamName("manip_out")
                manip.out.link(manip_out.input)
        
        # Define YUNET model
        print("Creating YUNET Neural Network...")
        nn = pipeline.createNeuralNetwork()
        nn.setBlobPath(self.model)
        if self.input_type == "rgb":
            manip.out.link(nn.input)
        else:
            nn_in = pipeline.createXLinkIn()
            nn_in.setStreamName("nn_in")
            nn_in.out.link(nn.input)

        # YUNET output
        nn_out = pipeline.createXLinkOut()
        nn_out.setStreamName("nn_out")
        nn.out.link(nn_out.input)

        print("Pipeline created.")
        return pipeline  

    def prior_gen(self):
        w, h = self.nn_input_w, self.nn_input_h
        feature_map_2th = [int(int((h + 1) / 2) / 2),
                           int(int((w + 1) / 2) / 2)]
        feature_map_3th = [int(feature_map_2th[0] / 2),
                           int(feature_map_2th[1] / 2)]
        feature_map_4th = [int(feature_map_3th[0] / 2),
                           int(feature_map_3th[1] / 2)]
        feature_map_5th = [int(feature_map_4th[0] / 2),
                           int(feature_map_4th[1] / 2)]
        feature_map_6th = [int(feature_map_5th[0] / 2),
                           int(feature_map_5th[1] / 2)]

        feature_maps = [feature_map_3th, feature_map_4th,
                        feature_map_5th, feature_map_6th]

        priors = []
        for k, f in enumerate(feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])): # i->h, j->w
                for min_size in min_sizes:
                    s_kx = min_size / w
                    s_ky = min_size / h

                    cx = (j + 0.5) * self.steps[k] / w
                    cy = (i + 0.5) * self.steps[k] / h

                    priors.append([cx, cy, s_kx, s_ky])
        print("Priors length =", len(priors))
        self.priors = np.array(priors, dtype=np.float32)

    def decode(self, inference):
        # print(inference.getAllLayerNames())
        loc = np.array(inference.getLayerFp16("loc"), dtype=np.float32).reshape(-1, 14)
        conf = np.array(inference.getLayerFp16("conf"), dtype=np.float32).reshape(-1, 2)
        iou_scores = np.array(inference.getLayerFp16("iou"), dtype=np.float32)

        # get score
        cls_scores = conf[:, 1]
        # clamp
        idx = np.where(iou_scores < 0.)
        iou_scores[idx] = 0.
        idx = np.where(iou_scores > 1.)
        iou_scores[idx] = 1.
        scores = np.sqrt(cls_scores * iou_scores)
        scores = scores[:, np.newaxis]

        # get bboxes
        bboxes = np.hstack((
            (self.priors[:, 0:2] + loc[:, 0:2] * self.variance[0] * self.priors[:, 2:4]) * self.padded_size,
            (self.priors[:, 2:4] * np.exp(loc[:, 2:4] * self.variance)) * self.padded_size
        ))
        # (x_c, y_c, w, h) -> (x1, y1, w, h)
        bboxes[:, 0:2] -= bboxes[:, 2:4] / 2

        # get landmarks
        landmarks = np.hstack((
            (self.priors[:, 0:2] + loc[:,  4: 6] * self.variance[0] * self.priors[:, 2:4]) * self.padded_size,
            (self.priors[:, 0:2] + loc[:,  6: 8] * self.variance[0] * self.priors[:, 2:4]) * self.padded_size,
            (self.priors[:, 0:2] + loc[:,  8:10] * self.variance[0] * self.priors[:, 2:4]) * self.padded_size,
            (self.priors[:, 0:2] + loc[:, 10:12] * self.variance[0] * self.priors[:, 2:4]) * self.padded_size,
            (self.priors[:, 0:2] + loc[:, 12:14] * self.variance[0] * self.priors[:, 2:4]) * self.padded_size
        ))

        dets = np.hstack((bboxes, landmarks, scores))

        return dets

    def save_inference_to_npz(self, inference):
        loc = np.array(inference.getLayerFp16("loc"), dtype=np.float32).reshape(-1, 14)
        conf = np.array(inference.getLayerFp16("conf"), dtype=np.float32).reshape(-1, 2)
        iou = np.array(inference.getLayerFp16("iou"), dtype=np.float32)
        np.savez("models/build/yunet_output.npz", loc=loc, conf=conf, iou=iou, w=self.nn_input_w, h=self.nn_input_h)


    def postprocess(self, inference):
        # Decode
        dets = self.decode(inference)

        # NMS
        keep_idx = cv2.dnn.NMSBoxes(
            bboxes=dets[:, 0:4].tolist(),
            scores=dets[:, -1].tolist(),
            score_threshold=self.conf_threshold,
            nms_threshold=self.nms_threshold,
            top_k=self.top_k
        ) # box_num x class_num
        if len(keep_idx) > 0:
            dets = dets[keep_idx]
            # If opencv >= 4.5.4.58, NMSBoxes returns Nx1x15
            # Else, NMSBoxes returns 1x15
            if len(dets.shape) > 2:
                dets = np.squeeze(dets, axis=1)
            return dets # [:self.keep_top_k]
        else:
            return np.empty(shape=(0, 15))

    def next_frame(self):
        """
        Return:
        - frame: source input frame,
        - faces: detected faces as a 2D numpy arrays of dim (N, 15) with N = number of faces:
            - faces[:,0:4] represents the bounding box (x,y,width,height),
            - faces[:,4:14] represents the 5 facial landmarks coordinates (x,y),
            - faces[:,15] is the detection score.
        """
        self.fps.update()
        if self.input_type == "rgb":
            if self.laconic:
                frame = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
            else:
                # Read color frame from the device
                in_video = self.q_video.get()
                frame = in_video.getCvFrame()
        else:
            if self.input_type == "image":
                frame = self.img.copy()
            else:
                ok, frame = self.cap.read()
                if not ok:
                    return None, None   
            # Send color frame to the device
            # The frame is padded to have the same ratio width/height 
            # as the model input, and resized to the model input resolution
            padded = cv2.copyMakeBorder(frame,
                                        0,
                                        self.padded_size[1] - self.img_h,
                                        0,
                                        self.padded_size[0] - self.img_w,
                                        cv2.BORDER_CONSTANT)
            padded = cv2.resize(padded, (self.nn_input_w, self.nn_input_h), interpolation=cv2.INTER_AREA)
            if self.trace:
                cv2.imshow("NN input", padded)
            frame_nn = dai.ImgFrame()
            frame_nn.setTimestamp(now())
            frame_nn.setWidth(self.nn_input_w)
            frame_nn.setHeight(self.nn_input_h)
            frame_nn.setData(padded.transpose(2, 0, 1))
            self.q_nn_in.send(frame_nn)
            rtrip_time = now()

        # Get model inference
        inference = self.q_nn_out.get()
        _now = now()
        if self.input_type != "rgb":
            self.glob_rtrip_time += _now - rtrip_time
        faces = self.postprocess(inference)
        self.glob_posprocessing_time = now() - _now

        # For debugging
        if self.trace and self.input_type == "rgb":
            manip = self.q_manip_out.get()
            manip = manip.getCvFrame()
            cv2.imshow("NN input", manip)  

        return frame, faces

    def exit(self):
        self.device.close()
        # Print some stats
        if self.stats:
            nb_frames = self.fps.nb_frames()
            print(f"FPS : {self.fps.get_global():.1f} f/s (# frames = {nb_frames})")
            if self.input_type != "rgb":
                print(f"Round trip (send frame + get back inference result)     : {self.glob_rtrip_time/nb_frames*1000:.1f} ms")
            print(f"Post processing time (on the host)                      : {self.glob_posprocessing_time/nb_frames*1000:.1f} ms")