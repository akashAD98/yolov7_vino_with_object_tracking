import numpy as np
import torch
from PIL import Image
from .deep.extractor import Extractorv3
from .deep.extractor import  Extractorv3_try_ad
from .deep.extractor import  ModelExtractor
from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.preprocessing import non_max_suppression
from .sort.detection import Detection
from .sort.tracker import Tracker

__all__ = ['DeepSortFace']


class DeepSortFace(object):
    def __init__(self, model_path, max_dist=0.2, min_confidence=0.3, nms_max_overlap=1.0, max_iou_distance=0.7,
                 max_age=70, n_init=3, nn_budget=100, use_cuda=True):
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap

        # ************************* face embedding *************************
        

        ## reid model
        #reid_model_path=r'D:\openVINO\working_code_deep_sort\DeepSORT_Object-master\deep_sort\deep\person-reidentification-retail-0287.xml'
        self.extractor = ModelExtractor(use_cuda=use_cuda)

        #print("model loaded",self.extractor)
        #print('Extractor is loading -->>',self.extractor)
        
        #self.extractor = Extractorv3(use_cuda=use_cuda)
        #self.extractor =Extractorv3_try_ad(use_cuda=use_cuda)
        # ******************************************************************
        max_cosine_distance = max_dist
        nn_budget = 100
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)

        # tracker maintain a list contains(self.tracks) for each Track object
        self.tracker = Tracker(metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

    def update(self, bbox_xywh, confidences,classes, ori_img):
        # bbox_xywh (#obj,4), [xc,yc, w, h]     bounding box for each person
        # conf (#obj,1)

        self.height, self.width = ori_img.shape[:2]

        # get appearance feature with neural network (Deep) *********************************************************
        features = self._get_features(bbox_xywh, ori_img)

        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)  # # [cx,cy,w,h] -> [x1,y1,w,h]   top left

        #  generate detections class object for each person *********************************************************
        # filter object with less confidence
        # each Detection obj maintain the location(bbox_tlwh), confidence(conf), and appearance feature
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(confidences) if
                      conf > self.min_confidence]

        # run on non-maximum supression (useless) *******************************************************************
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = non_max_suppression(boxes, self.nms_max_overlap, scores)  # Here, nms_max_overlap is 1
        detections = [detections[i] for i in indices]

        # update tracker ********************************************************************************************
        self.tracker.predict()  # predict based on t-1 info
        # for first frame, this function do nothing

        # detections is the measurement results as time T
        self.tracker.update(detections, classes, confidences)
        
        # output bbox identities ************************************************************************************
        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue

            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            
            track_id = track.track_id
            class_id = track.class_id
            conf = track.conf
            outputs.append(np.array([x1, y1, x2, y2, track_id, class_id, conf]))
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs

    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """

    
class DeepSortFaceSingle(object):
    def __init__(self, model_path, max_dist=0.2, min_confidence=0.3, nms_max_overlap=1.0, max_iou_distance=0.7,
                 max_age=70, n_init=3, nn_budget=100, use_cuda=True):
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap
        self.trackers = {}
        self.extractors = {}

        # create a tracker and re-identification model for each class
        for i in range(51):
            metric = NearestNeighborDistanceMetric("cosine", max_dist, nn_budget)
            tracker = Tracker(metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)
            self.trackers[i] = tracker
            self.extractors[i] = ModelExtractor(use_cuda=use_cuda)

    def update(self, bbox_xywh, confidences, classes, ori_img):
        self.height, self.width = ori_img.shape[:2]
        outputs = []


        #classes= []
        #for i in range(1, len(classes) + 1):
        #classes=['']
        for i in range(51):
            tracker = self.trackers[i]
            extractor = self.extractors[i]

            # filter detections by class
            class_mask = (classes == i)
            class_bbox_xywh = bbox_xywh[class_mask]
            class_confidences = confidences[class_mask]

            if len(class_bbox_xywh) == 0:
                continue

            # get appearance feature with neural network
            features = self._get_features(class_bbox_xywh, ori_img)

            bbox_tlwh = self._xywh_to_tlwh(class_bbox_xywh)

            # generate detections for this class
            detections = [Detection(bbox_tlwh[j], conf, features[j]) for j, conf in enumerate(class_confidences) if
                          conf > self.min_confidence]

            # run non-maximum suppression
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = non_max_suppression(boxes, self.nms_max_overlap, scores)
            detections = [detections[j] for j in indices]

            # update tracker
            tracker.predict()
            tracker.update(detections)

            # output bbox identities
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                box = track.to_tlwh()
                x1, y1, x2, y2 = self._tlwh_to_xyxy(box)

                track_id = track.track_id
                class_id = i
                conf = track.confidence
                outputs.append(np.array([x1, y1, x2, y2, track_id, class_id, conf]))

        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs


    
    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh      # xc, yc, w, h
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x + w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y + h), self.height - 1)
        return x1, y1, x2, y2

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h

    def _get_features(self, bbox_xywh, ori_img):
        """
        :param bbox_xywh:
        :param ori_img: cv2 array (h,w,3)
        :return:
        """
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            #data = Image .fromarray(im)
            im_crops.append(im)
        if im_crops:
           # features = self.extractor.get_features(im_crops)
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features





