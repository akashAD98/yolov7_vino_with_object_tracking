# yolov7_vino_with_deepsort object_tracking
it has support for openvino converted model of yolov7-int.xml ,yolov7x,

installation steps

first install
1.install openvino
follow this steps to install openvino on windows

https://github.com/openvinotoolkit/openvino_notebooks/wiki/Windows`

2.install deepsort dependenices
```
pip install -r requirements.txt
```

To run code
```
python main_face.py

```

To change extractor go inside this file & use which extractor you want for tracking

```
https://github.com/akashAD98/yolov7_vino_with_object_tracking/blob/main/deep_sort/deep/extractor.py
```


### download object tracking person-reid weight from here

 weights & tracking code from this
 https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/407-person-tracking-webcam
 
 weight file:
https://drive.google.com/drive/folders/16Wi8LhdikcEhKubTr3YvtPdWF2YIFSCA?usp=share_link


### Citation
```bibtex
@article{openvino-notebbok,
https://github.com/openvinotoolkit/openvino_notebooks
}
```

```
```
