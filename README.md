# Face Detection, Recognition and Expression Recognition

The repo contains the code for running the All-in-one face detection/recognition + expression recognition

## Installation

Requires several libraries to get the demo running.

1. Caffe + PyCaffe [Installation](http://caffe.berkeleyvision.org/installation.html)
2. Tensorflow for Face expression [Installation](https://www.tensorflow.org/install/pip)
3. OpenCV
4. Currently the code supports NVIDIA GPUs. For CPU runs we may need to modify the code.
5. Requires Python 2.7


## Usage

1. Download the model files from [here](https://drive.google.com/file/d/1Y3KXEX3BsuZYtgQUXCYjC_4WD-irdYsP/view?usp=sharing)
2. Untar the contents and place the content in the models directory
3. Place the images that you want to be identified in the data directory. Then run 

``` Python
 python utils/generate_train_features.py 
```

4. Next, run
``` Python
 python demo.py 
```

A window will open and will start detecting the faces

5. Experimental feature.
For detecting the facial expressions, modify utils/MTCNN.py and change EXPRESSION_DETECTION_ENABLED = True and restart the demo.py

TODO: Expression recognition is buggy and the trained model is inaccurate. Need a better model for improved accuracy. 


 

## Contributing
For major changes, please open an issue first to discuss what you would like to change.
