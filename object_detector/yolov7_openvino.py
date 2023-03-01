from openvino.runtime import Core
from typing import List, Tuple, Dict
from utils.general import scale_coords, non_max_suppression
from openvino.runtime import Model
import numpy as np
import torch
from PIL import Image
from utils.datasets import letterbox
from pathlib import Path

core = Core()
# read converted model
model = core.read_model(r"D:\openVINO\working_code_deep_sort\ori_mam_sup\DeepSORT_Object-master\ir_weights\napparel_v1_withbgtrt_IRnewint8.xml")
# load model on CPU device
model = core.compile_model(model, 'CPU')

def preprocess_image(img0: np.ndarray):
    """
    Preprocess image according to YOLOv7 input requirements. 
    Takes image in np.array format, resizes it to specific size using letterbox resize, converts color space from BGR (default in OpenCV) to RGB and changes data layout from HWC to CHW.
    
    Parameters:
      img0 (np.ndarray): image for preprocessing
    Returns:
      img (np.ndarray): image after preprocessing
      img0 (np.ndarray): original image
    """
    # resize
    img = letterbox(img0, auto=False)[0]
    
    # Convert
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    return img, img0


def prepare_input_tensor(image: np.ndarray):
    """
    Converts preprocessed image to tensor format according to YOLOv7 input requirements. 
    Takes image in np.array format with unit8 data in [0, 255] range and converts it to torch.Tensor object with float data in [0, 1] range
    
    Parameters:
      image (np.ndarray): image for conversion to tensor
    Returns:
      input_tensor (torch.Tensor): float tensor ready to use for YOLOv7 inference
    """
    input_tensor = image.astype(np.float32)  # uint8 to fp16/32
    input_tensor /= 255.0  # 0 - 255 to 0.0 - 1.0
    
    if input_tensor.ndim == 3:
        input_tensor = np.expand_dims(input_tensor, 0)
    return input_tensor


# label names for visualization
NAMES = ['Tie','Boots','Shirt','Jeans','Suit','Dhoti','Long dress/gown','Hoodie','Pants','Sherwani','Turban','Sunglasses & goggles','Spectacles & glasses','Hand bag','Necklace','Hat','Scarf','Backpack','Cap','Sneakers','Wrist watch','Salwar Suit','Earring','Anklet','Bangle','Barefoot','Bracelet','Chappals','Nose ring','Ring','Shorts','Denim jacket','Kurta\kurti','Leggins','Mask','Pyjama','Sweater','Taqiyah','Heels','Waist Coat','Night Gown','Long skirt','Short skirt','Headphones','Belt','Duffle bags','Earphones','Smart watch','Suitcase','Swimwear','Sling Bag']

# colors for visualization
COLORS = {name: [np.random.randint(0, 255) for _ in range(3)]
          for i, name in enumerate(NAMES)}


def detect( image_path: Path, conf_thres: float = 0.60, iou_thres: float = 0.50, classes: List[int] = None, agnostic_nms: bool = False):
    """
    OpenVINO YOLOv7 model inference function. Reads image, preprocess it, runs model inference and postprocess results using NMS.
    Parameters:
        model (Model): OpenVINO compiled model.
        image_path (Path): input image path.
        conf_thres (float, *optional*, 0.25): minimal accpeted confidence for object filtering
        iou_thres (float, *optional*, 0.45): minimal overlap score for remloving objects duplicates in NMS
        classes (List[int], *optional*, None): labels for prediction filtering, if not provided all predicted labels will be used
        agnostic_nms (bool, *optiona*, False): apply class agnostinc NMS approach or not
    Returns:
       pred (List): list of detections with (n,6) shape, where n - number of detected boxes in format [x1, y1, x2, y2, score, label] 
       orig_img (np.ndarray): image before preprocessing, can be used for results visualization
       inpjut_shape (Tuple[int]): shape of model input tensor, can be used for output rescaling
    """
    output_blob = model.output(0)
    #img = np.array(Image.open(image_path))
    preprocessed_img, orig_img = preprocess_image(image_path)
    input_tensor = prepare_input_tensor(preprocessed_img)
    predictions = torch.from_numpy(model(input_tensor)[output_blob])
    pred = non_max_suppression(predictions, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
    boxes = []
    labels = []
    scores = []
    pred=pred[0]
    if len(pred):
        # Rescale boxes from input size to original image size
        pred[:, :4] = scale_coords(input_tensor.shape[2:], pred[:, :4], orig_img.shape).round()
        for *xyxy, conf, cls in reversed(pred):
            boxes.append(xyxy)
            labels.append(cls)
            scores.append(conf)
            print('score',conf)
    if len(boxes) == 0:
        boxes = np.array([]).reshape(0, 4)
        scores = np.array([])
        labels = np.array([])

    return np.array(boxes), np.array(scores), np.array(labels)