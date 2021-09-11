# YuNet Face Detection with DepthAI

Running YuNet on [DepthAI](https://docs.luxonis.com/) hardware (OAK-1, OAK-D, ...). 

**YuNet** is a light-weight, fast and accurate face detection model, which achieves 0.834(easy), 0.824(medium), 0.708(hard) on the WIDER Face validation set.

This repository heavily rely on the work done by :
- OpenCV : the original ONNX model and the postprocessing code comes from the [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/dev/models/face_detection_yunet);
- PINTO : the models can be found [there](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/144_YuNet)

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
usage: demo.py [-h] [-i INPUT] [-mr {360x640,90x160,180x320,120_160}]
               [-f INTERNAL_FPS]
               [--internal_frame_height INTERNAL_FRAME_HEIGHT] [-t]
               [-o OUTPUT]

optional arguments:
  -h, --help            show this help message and exit

Tracker arguments:
  -i INPUT, --input INPUT
                        Path to video or image file to use as input (if not
                        specified, use OAK color camera)
  -mr {360x640,90x160,180x320,120_160}, --model_resolution {360x640,90x160,180x320,120_160}
                        Select YuNet model input resolution (default=180x320)
  -f INTERNAL_FPS, --internal_fps INTERNAL_FPS
                        Fps of internal color camera. Too high value lower NN
                        fps
  --internal_frame_height INTERNAL_FRAME_HEIGHT
                        Internal color camera frame height in pixels
  -t, --trace           Print some debug messages

Renderer arguments:
  -o OUTPUT, --output OUTPUT
                        Path to output video file

```

**Examples :**

- To use default internal color camera as input with the default model ():

    ```python3 demo.py```

- To use another model resolution:
    ```python3 demo.py -mr 360x640```

- To use a file (video or image) as input :

    ```python3 demo.py -i filename```

## Models:

The [source ONNX model](https://github.com/opencv/opencv_zoo/blob/dev/models/face_detection_yunet/face_detection_yunet.onnx) comes from the OpenCv zoo.
To run the model on MYRIAD, the model needs to be converted in OpenVINO IR format, then compiled into a blob file. This is done with the [openvino2tensorflow tool from PINTO](https://github.com/PINTO0309/openvino2tensorflow).

The blob files does not allow dynamic input resolution. Therefore, we need to generate one blob file for any desired resolution. 
A few models with a 16/9 resolution are available in the models directory of this repository (16/9 is the ratio of the internal color camera).

To generate a blob for another resolution (or for another number of shaves), follow this procedure:
1) Install the docker version of openvino2tensorflow (install docker if necessary):
```docker pull pinto0309/tflite2tensorflow:latest```
2) After having cloned this repository:
```
> cd models
> ./docker_openvino2tensorflow.sh # Run the docker container
> cd workdir/build
> ./build.sh -h
./build.sh -h
Usage:
Generate a new YuNet blob with a specified model input resolution and number of shaves

Usage: ./build.sh -r HxW [-s nb_shaves]

HxW: height x width, example 120x160
nb_shaves must be between 1 and 13. If not specified, default=4
```
So to generate the blob for a 360x640 resolution using 6 shaves of the MyriadX, run :
```
> ./build.sh -r 360x640 -s 6 
```

# Credits
* [OpenCV Zoo](https://github.com/opencv/opencv_zoo)
* Katsuya Hyodo a.k.a [Pinto](https://github.com/PINTO0309), the Wizard of Model Conversion !