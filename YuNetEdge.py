import numpy as np
from itertools import product
import depthai as dai
from math import gcd
from pathlib import Path
from FPS import FPS
import cv2
import os, sys, re
from string import Template


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_YUNET_MODEL = str(SCRIPT_DIR / "models/face_detection_yunet_180x320_sh4.blob")
POSTPROC_MODEL_FORMAT = "models/postproc_yunet_top50_{H}x{W}_sh4.blob"
TEMPLATE_MANAGER_SCRIPT = str(SCRIPT_DIR / "template_manager_script.py")

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
    - model_postproc: None or path to the post processing model. If None, the path is determined
            from the resolution: POSTPROC_MODEL_FORMAT.format(res=model_resolution)
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
    - sync : boolean. If True, the video frame sent to the host is the frame on which inference was run
            You probably want sync=True, if you are doing face recognition. 
            Note that it lowers the FPS.
    - stats : boolean, when True, display some statistics when exiting.   
    - trace: boolean, when True print some debug messages   
    """
    def __init__(self, 
                model = str(DEFAULT_YUNET_MODEL),
                model_resolution=None, 
                model_postproc = None,
                input_src=None,                 
                conf_threshold=0.6, 
                nms_threshold=0.3, 
                internal_fps=50,
                internal_frame_height=640,
                sync=False,
                stats=False,
                trace=False):

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
        print(f"Model YuNet : {self.model} - Input resolution: {self.nn_input_h}x{self.nn_input_w}")

        if model_postproc is None:
            self.model_postproc = str(SCRIPT_DIR / POSTPROC_MODEL_FORMAT.format(H=self.nn_input_h, W=self.nn_input_w))
        else:
            self.model_postproc = model_postproc
        if not os.path.isfile(self.model_postproc):
            print(f"Model path '{self.model_postproc}' does not exist !!!")
            sys.exit()
        print(f"Model postprocessing : {self.model_postproc}")

        self.internal_fps = internal_fps
        
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.sync = sync
        print(f"Synchronization {'enabled' if self.sync else 'disabled'}")
        self.stats = stats
        self.trace = trace

        self.device = dai.Device()

        if input_src is None or input_src == "rgb" or input_src == "rgb_laconic":
            self.input_type = "rgb" # OAK* internal color camera
            self.laconic = input_src == "rgb_laconic" # Camera frames are not sent to the host
            self.video_fps = self.internal_fps # Used when saving the output in a video file. Should be close to the real fps

            width, self.scale_nd = find_isp_scale_params(internal_frame_height * 1920 / 1080, is_height=False)
            self.img_h = int(round(1080 * self.scale_nd[0] / self.scale_nd[1]))
            self.img_w = int(round(1920 * self.scale_nd[0] / self.scale_nd[1]))
            print(f"Internal camera image size: {self.img_w} x {self.img_h}")

        else:
            print("Invalid input source:", input_src)
            sys.exit()
        
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
                if self.sync:
                    self.q_video = self.device.getOutputQueue(name="cam_out", maxSize=4, blocking=True)
                else:
                    self.q_video = self.device.getOutputQueue(name="cam_out", maxSize=1, blocking=False)
            if self.trace:
                self.q_manip_out = self.device.getOutputQueue(name="manip_out", maxSize=1, blocking=False)
        if self.sync:
            self.q_nn_pp_out = self.device.getOutputQueue(name="nn_pp_out", maxSize=4, blocking=True)
        else:
            self.q_nn_pp_out = self.device.getOutputQueue(name="nn_pp_out", maxSize=1, blocking=False)

        self.fps = FPS()   

    def create_pipeline(self):
        print("Creating pipeline...")
        # Start defining a pipeline
        pipeline = dai.Pipeline()
        pipeline.setOpenVINOVersion(version = dai.OpenVINO.Version.VERSION_2021_4)

        # ColorCamera
        print("Creating Color Camera...")
        cam = pipeline.create(dai.node.ColorCamera)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam.setInterleaved(False)
        cam.setIspScale(self.scale_nd[0], self.scale_nd[1])
        cam.setFps(self.internal_fps)
        cam.setPreviewSize(self.img_w, self.img_h)

        if not self.laconic:
            cam_out = pipeline.create(dai.node.XLinkOut)
            cam_out.setStreamName("cam_out")

            if self.sync:
                print("Creating Script node...")
                manager = pipeline.create(dai.node.Script)
                manager.inputs['preview'].setQueueSize(1)
                manager.inputs['preview'].setBlocking(False)
                manager.setScript(self.build_manager_script())

                cam.preview.link(manager.inputs['preview'])
                manager.outputs['host'].link(cam_out.input)
            else:
                cam_out.input.setQueueSize(1)
                cam_out.input.setBlocking(True)
                cam.video.link(cam_out.input)

            


        # The frame is padded to have the same ratio width/height 
        # as the model input, and resized to the model input resolution
        print("Creating Image Manip node...")
        manip = pipeline.create(dai.node.ImageManip)
        manip.setMaxOutputFrameSize(self.nn_input_w*self.nn_input_h*3)
        manip.inputImage.setQueueSize(1)
        if self.sync:
            manip.inputImage.setBlocking(True)
        else:
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
        if self.laconic or not self.sync:
            cam.preview.link(manip.inputImage)
        else:
            manager.outputs['manip'].link(manip.inputImage)

        # For debugging
        if self.trace:
            manip_out = pipeline.create(dai.node.XLinkOut)
            manip_out.setStreamName("manip_out")
            manip.out.link(manip_out.input)
        
        # Define YUNET model
        print("Creating YUNET Neural Network...")
        nn = pipeline.create(dai.node.NeuralNetwork)
        nn.setBlobPath(self.model)
        if self.sync:
            nn.input.setBlocking(True)
            nn.input.setQueueSize(1)
        manip.out.link(nn.input)

        # Define postprocessing model
        print("Creating Post Processing Neural Network...")
        nn_pp = pipeline.create(dai.node.NeuralNetwork)
        nn_pp.setBlobPath(self.model_postproc)
        nn.out.link(nn_pp.input)
        

        # Post Processing output
        nn_pp_out = pipeline.create(dai.node.XLinkOut)
        nn_pp_out.setStreamName("nn_pp_out")
        nn_pp.out.link(nn_pp_out.input)

        print("Pipeline created.")
        return pipeline  

    def build_manager_script(self):
        '''
        The code of the scripting node 'manager' is built 
        from the content of the file template_manager_script.py 
        which is a python template
        '''
        # Read the template
        with open(TEMPLATE_MANAGER_SCRIPT, 'r') as file:
            template = Template(file.read())
        
        # Perform the substitution
        code = template.substitute(
                    _TRACE = "node.warn" if self.trace else "#",
        )
        # Remove comments and empty lines
        import re
        code = re.sub(r'"{3}.*?"{3}', '', code, flags=re.DOTALL)
        code = re.sub(r'#.*', '', code)
        code = re.sub('\n\s*\n', '\n', code)
        # For debugging
        if self.trace:
            with open("tmp_code.py", "w") as file:
                file.write(code)

        return code

    def postprocess(self, inference):
        faces = np.array(inference.getFirstLayerFp16(), dtype=np.float32).reshape(-1, 15)
        # Faces are sorted by score (higher score first)
        # Find the first face with score below threshold and discard from there
        for i in range(faces.shape[0]):
            conf = faces[i,-1]
            if conf < self.conf_threshold:
                break
        faces = faces[:i]

        faces[:,2:4] -= faces[:,0:2] # Replace (x2,y2) by (w,h)
        faces[:,:14] *= np.tile(self.padded_size, 7)
        return faces

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

        if self.laconic:
            frame = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        else:
            # Read color frame from the device
            in_video = self.q_video.get()
            frame = in_video.getCvFrame()

        # Get model inference
        inference = self.q_nn_pp_out.get()
        faces = self.postprocess(inference) 

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
        