# YuNet Face Detection with DepthAI


Running YuNet on [DepthAI](https://docs.luxonis.com/) hardware (OAK-1, OAK-D, ...). 

**YuNet** is a light-weight, fast and accurate face detection model, which achieves 0.834(easy), 0.824(medium), 0.708(hard) on the WIDER Face validation set.

This repository heavily rely on the work done by :
- OpenCV : the original ONNX model and the postprocessing code comes from the [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/dev/models/face_detection_yunet);
- PINTO : the models can be found [there](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/144_YuNet). The models are also present in the current repository and scripts for regerenerating or generating new models are also available.

![Demo](img/oscars_360x640.gif)

## Install


Install the python packages (depthai, opencv) with the following command:

```
python3 -m pip install -r requirements.txt
```

## Run

**Usage:**

```
> python3 demo.py -h
usage: demo.py [-h] [-e] [-myu MODEL_YUNET] [-mpp MODEL_POSTPROC] [-i INPUT]
               [-mr {360x640,270x480,180x320,90x160,120x160,144x256}]
               [-f INTERNAL_FPS]
               [--internal_frame_height INTERNAL_FRAME_HEIGHT] [-s] [-t]
               [-o OUTPUT]

optional arguments:
  -h, --help            show this help message and exit
  -e, --edge            Use Edge mode (postprocessing runs on the device)

Tracker arguments:
  -myu MODEL_YUNET, --model_yunet MODEL_YUNET
                        Path to Yunet blob
  -mpp MODEL_POSTPROC, --model_postproc MODEL_POSTPROC
                        Path to Post Processing model blob (only in edge node)
  -i INPUT, --input INPUT
                        Path to video or image file to use as input (if not
                        specified, use OAK color camera)
  -mr {360x640,270x480,180x320,90x160,120x160,144x256}, --model_resolution {360x640,270x480,180x320,90x160,120x160,144x256}
                        Select YuNet model input resolution (default=180x320)
  -f INTERNAL_FPS, --internal_fps INTERNAL_FPS
                        Fps of internal color camera. Too high value lower NN
                        fps
  --internal_frame_height INTERNAL_FRAME_HEIGHT
                        Internal color camera frame height in pixels
  -s, --sync            Synchronize video frame and Yunet inference (only in
                        Edge mode)
  -t, --trace           Print some debug messages

Renderer arguments:
  -o OUTPUT, --output OUTPUT
                        Path to output video file
```

**Examples :**

- To run in Host mode (postprocessing done on the host) using default internal color camera as input and default model (180x320):

    ```python3 demo.py```

- To run in Edge mode (postprocessing done on the device) using default internal color camera as input and default model (180x320):

    ```python3 demo.py -e```

- To run in Edge mode (postprocessing done on the device) using default internal color camera as input and default model (180x320) with video frame/ detections synchronization:

    ```python3 demo.py -e -s```

Synchronization means here that the video frame you get is the one on which inference was run. Synchronization is important in applications where the face detection is an early step in a more complex pipeline with other neural networks exploiting the detection result (e.g. face recognition, age-gender estimation,...). Synchronization has a small impact on FPS.

- To use another model resolution:
    ```python3 demo.py -mr 270x480```

- To use a file (video or image) as input (only in Host mode):

    ```python3 demo.py -i filename```

- When using the internal camera, to change its FPS to 70 : 

    ```python3 demo.py --internal_fps 70```

    Note: by default, the default internal camera FPS is fixed. It may be too slow or too fast for the chosen model. **So please, don't hesitate to play with this parameter to find the optimal value.** If you observe that your FPS is well below the default value, you should lower the FPS with this option until the set FPS is just above the observed FPS.

|Keypress in OpenCV window|Function|
|-|-|
|*Esc*|Exit|
|*space*|Pause|
|b|Show/hide bounding boxes|
|l|Show/hide landmarks|
|s|Show/hide scores|
|f|Show/hide FPS|

## Models:
There are 2 models:
* The YuNet model, that does all the face detection job;
* The Post Processing model, that takes the output of the YuNet model and processes it (mainly it performs non-maximum suppression (NMS) on the detection boxes). This model is used only in Edge mode. In Host mode the post processing is done on the host's CPU.

The 2 models go in pairs: both are generated for a given common YuNet model input resolution (heightxwidth).

### 1) Yunet model
The [source ONNX model](https://github.com/opencv/opencv_zoo/blob/dev/models/face_detection_yunet/face_detection_yunet.onnx) comes from the OpenCv zoo.
To run the model on MYRIAD, the model needs to be converted in OpenVINO IR format, then compiled into a blob file. This is done with the [openvino2tensorflow tool from PINTO](https://github.com/PINTO0309/openvino2tensorflow).

The blob files does not allow dynamic input resolution. Therefore, we need to generate one blob file for any desired resolution. 
A few models with a 16/9 resolution are available in the models directory of this repository (16/9 is the ratio of the internal color camera).

To generate a blob for another resolution (or for another number of shaves), follow this procedure:
1) Install the docker version of openvino2tensorflow (install docker if necessary):
```docker pull pinto0309/openvino2tensorflow:latest```
2) After having cloned this repository:
```
> cd models
> ./docker_openvino2tensorflow.sh # Run the docker container
> cd workdir/build
> ./build_yunet_blob.sh -h
./build_yunet_blob.sh -h
Usage:
Generate a new YuNet blob with a specified model input resolution and number of shaves

Usage: ./build_yunet_blob.sh -r HxW [-s nb_shaves]

HxW: height x width, example 120x160
nb_shaves must be between 1 and 13. If not specified, default=4
```
So to generate the blob for a 360x640 resolution using 6 shaves of the MyriadX, run :
```
> ./build_yunet_blob.sh -r 360x640 -s 6 
```

### 2) Post Processing model
The post processing model : 
- takes the 3 outputs of the Yunet model (loc:Nx14, conf:Nx2 and iou:Nx1, with N depending on the Yunet input resolution, e.g. N=3210 for 180x360),
- arrange a bit the datas before applying to them the Non Maximum Suppression algorithm. NMS outputs the `top_k` better detections having a score above `score_thresh`. `top_k` and `score_thresh` are user-defined parameters chosen when generating the model. `top_k` corresponds the number max of faces, you want to detect. The higher, the slower. For instance, in an authentification application where the user stands in front of the camera, `top_k = 1` is probably enough. The default value in the script below is `top_k = 50`. The final outputs of the model are: 
  - `dets` of shape top_k x 15, where:
    - dets[:,0:4] represent the bounding boxes (x,y,width,height),
    - dets[:,4:14] represent the 5 facial landmarks coordinates (x,y),
    - dets[:,15] is the detection score.
  - `dets@shape` that corresponds to the number of valid detections. So in `dets`, only the first `dets@shape` entries are valid.

To generate a Post Processing model, some python packages are needed: torch, onnx, onnx-simplifier, onnx_graphsurgeon. They can be installed with the following command:
```
python3 -m pip install -r models/build/requirements.txt
```

Then use the script `models/build/generate_postproc_onnx.py` to generate the ONNX model :
```
> cd models/build
> python3 generate_postproc_onnx.py -h
usage: generate_postproc_onnx.py [-h] -W W -H H [-top_k TOP_K]
                                 [-score_thresh SCORE_THRESH] [-no_simp]

optional arguments:
  -h, --help            show this help message and exit
  -W W                  yunet model input width
  -H H                  yunet model input height
  -top_k TOP_K          max number of detections (default=50)
  -score_thresh SCORE_THRESH
                        NMS score threshold
  -no_simp              do not run simplifier


# Example:
> python3 generate_postproc_onnx.py -H 180 -W 320
# The command above generates 'postproc_yunet_top50_th60_180x320.onnx' in the 'models/build' directory (_th60_ means score_threshold=0.6)
```

Finally convert the ONNX model into OpenVINO IR format then in a blob file, using a similar method to the Yunet convertion :
```
> cd models
> ./docker_openvino2tensorflow.sh # Run the docker container
> cd workdir/build
> ./build_postproc_blob.sh -h
Generate a blob from an ONNX model with a specified number of shaves and cmx (nb cmx = nb shaves)

Usage: ./build_postproc_blob.sh [-m model_onnx] [-s nb_shaves]

model_onnx: ONNX file
nb_shaves must be between 1 and 13 (default=4)

# Example:
> ./build_postproc_blob.sh -m postproc_yunet_top50_th60_180x320.onnx
# The command above generates 'postproc_yunet_top50_th60_360x640_sh4.blob' in the 'models' directory
```


# Credits
* [OpenCV Zoo](https://github.com/opencv/opencv_zoo)
* Katsuya Hyodo a.k.a [Pinto](https://github.com/PINTO0309), the Wizard of Model Conversion !