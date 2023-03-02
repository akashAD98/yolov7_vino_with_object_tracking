from .inception_resnet_v1 import InceptionResnetV1
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
import onnxruntime
from openvino.runtime import Core



from openvino.inference_engine import IECore
import torch.nn.functional as F

class Extractor(object):
    def __init__(self, img_size=160, use_cuda=True):
        self.net = InceptionResnetV1(pretrained='vggface2')
        self.img_size = img_size    # 160

        self.use_cuda = use_cuda
        if use_cuda:
            self.net = self.net.cuda()

    def get_features(self, im_crops):
        # ori_img, cv2 array
        faces_im = []
        for face in im_crops:
            face = cv2.resize(face, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)  # (h, w, 3)
            face = face.transpose(2, 0, 1)                  # (3, H, W)
            face = torch.from_numpy(np.float32(face))
            face = (face - 127.5) / 128.0
            faces_im.append(face)
        faces_im = torch.stack(faces_im)

        return faces_im

    def __call__(self, im_crops):
        self.net.eval()

        img_batch = self.get_features(im_crops)

        with torch.no_grad():

            if self.use_cuda:
                img_batch = img_batch.cuda()
            embedding = self.net(img_batch)
        print(type(embedding),len(embedding))
        return embedding.cpu().numpy()

from PIL import Image
import torchvision.transforms as transforms
import onnxruntime

class Extractorv2(object):
    def __init__(self, img_size=160, use_cuda=True):
        self.resize = transforms.Resize([img_size, img_size])
        self.onnx_session = onnxruntime.InferenceSession(r"C:\Users\admin\Desktop\Reflexion\ADevops\Model_Optimization\DS\Reflexion-Face-Detection-Tracking\face_detector_1mb\facenet4.onnx",providers=['CPUExecutionProvider'])
    
    def to_numpy(self,tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


    def get_features(self, im_crops):
        # ori_img, cv2 array
        faces_im = []
        for face in im_crops:
            face = self.resize(face)
            img_ycbcr = face.convert('YCbCr')
            to_tensor = transforms.ToTensor()
            img_y = to_tensor(img_ycbcr)
            img_y.unsqueeze_(0)
            img_y=self.to_numpy(img_y)
            faces_im.append(img_y)
        #aces_im = torch.stack(faces_im)
        faces_im = np.vstack(faces_im)

        return faces_im

    def __call__(self, im_crops):

        img_batch = self.get_features(im_crops)
        embeddings=[]
        
        #for i in img_batch:
        ort_inputs = {self.onnx_session.get_inputs()[0].name: img_batch}
        embedding = self.onnx_session.run(None, ort_inputs)
        #embeddings.append(embedding[0][0])
        
        return embedding[0]

class Extractorv3(object):
    def __init__(self, img_size=160, use_cuda=True):
        model_path=r"./face_detector_1mb/mobilenet_v2_fp32.xml"
        self.img_size = img_size
        core = Core()
        self.model = core.read_model(model=model_path)



        # self.input_layer = self.model.input(0)
        # self.input_shape = self.input_layer.shape
        # self.height = self.input_shape[2]
        # self.width = self.input_shape[3]
        # print("v3")
        # print('input_layer',self.input_layer)
        # print('self.input_shape',self.input_shape)
        # print('self.height',self.height)
        # print('self.height',self.width)

        self.compiled_model_ir = core.compile_model(model=self.model, device_name="CPU")
        self.output_layer_ir = self.compiled_model_ir.output(0)

    
    def to_numpy(self,tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


    def get_features(self, im_crops):
        faces_im = []
        c=0
        for face in im_crops:
            c=c+1
            print('face_count_loaded_count',c)
            face = cv2.resize(face, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)  # (h, w, 3)
            face = face.transpose(2, 0, 1) 
            print("face shape",face.shape)                 # (3, H, W)
            face = torch.from_numpy(np.float32(face))
            face = (face - 127.5) / 128.0
            faces_im.append(face)
        faces_im = torch.stack(faces_im)

        return faces_im

    def __call__(self, im_crops):

        img_batch = self.get_features(im_crops)
        output = self.compiled_model_ir([img_batch])[self.output_layer_ir]

        return output

# old 128
class Extractorv3_try_ad(object):
    def __init__(self, batchsize=1, img_size=160,use_cuda=False):
        model_path=r'D:\openVINO\working_code_deep_sort\DeepSORT_Object-master\deep_sort\deep\person-reidentification-retail-0287.xml'
        #r"./face_detector_1mb/mobilenet_v2_fp32.xml"
        self.img_size = img_size
        core = Core()
        
        self.model = Model(model_path, -1)
       # self.model = core.read_model(model=model_path,-1)
        #self.model = self.model.reshape([1,3,256,256])
        

        self.input_layer = self.model.input(0)
        self.input_shape = self.input_layer.shape
        self.height = self.input_shape[2]
        self.width = self.input_shape[3]
        print("person-detection-0202")
        print('input_layer',self.input_layer)
        print('self.input_shape',self.input_shape)
        print('self.height',self.height)
        print('self.height',self.width)

        # print("######    ")
        # self.model_resized = self.model.reshape([1,3,256,256])
        # print("resized model")
        #print('self.input_shape',self.input_shape)


        for layer in self.model.inputs:
            input_shape = layer.partial_shape
            input_shape[0] = batchsize
            self.model.reshape({layer: input_shape})
            print("## input shape",layer.partial_shape)
            print('inp shape batchsize', input_shape[0])
        # self.compiled_model = ie_core.compile_model(model=self.model, device_name="CPU")
        # self.output_layer = self.compiled_model.output(0)
        self.compiled_model_ir = core.compile_model(model=self.model, device_name="CPU")
        self.output_layer_ir = self.compiled_model_ir.output(0)
    
    def to_numpy(self,tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


    def get_features(self, im_crops):
        faces_im = []
        for face in im_crops:
            face = cv2.resize(face, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)  # (h, w, 3)
            face = face.transpose(2, 0, 1)                  # (3, H, W)
            face = torch.from_numpy(np.float32(face))
            face = (face - 127.5) / 128.0
            faces_im.append(face)
        faces_im = torch.stack(faces_im)

        return faces_im

    def __call__(self, im_crops):

        img_batch = self.get_features(im_crops)
        output = self.compiled_model_ir([img_batch])[self.output_layer_ir]

        return output


from openvino.runtime import Core, PartialShape
ie_core = Core()
class ModelExtractor:
    """
    This class represents a OpenVINO model object.

    """
    def __init__(self,img_size=256, batchsize=-1, device="AUTO",use_cuda=False):

        ie_core = Core()
        """
        Initialize the model object
        
        Parameters
        ----------
        model_path: path of inference model
        batchsize: batch size of input data
        device: device used to run inference
        """
        model_path=r'D:\openVINO\working_code_deep_sort\ORI_MA\DeepSORT_Object-master\model_weight\person-reidentification-retail-0277\FP32\person-reidentification-retail-0277.xml'
        #'D:\openVINO\working_code_deep_sort\ORI_MA\DeepSORT_Object-master\model_weight\fp16\person-reidentification-retail-0287.xml'
        #r'D:\openVINO\working_code_deep_sort\DeepSORT_Object-master\deep_sort\deep\person-reidentification-retail-0287.xml'

#D:\openVINO\working_code_deep_sort\ORI_MA\DeepSORT_Object-master\model_weight\person-reidentification-retail-0277\FP32\person-reidentification-retail-0277.xml
    
        self.img_size = img_size
        
        self.model = ie_core.read_model(model=model_path)
        self.input_layer = self.model.input(0)
        self.input_shape = self.input_layer.shape
        self.height = self.input_shape[2]
        self.width = self.input_shape[3]
        print("Orignal_model_shape",self.input_layer.shape)


        new_shape = PartialShape([1, 3, 256,128])
        self.model.reshape({self.input_layer.any_name: new_shape})
        #self.height = self.input_shape[2]
        #self.width = self.input_shape[3]
        self.input_layer = self.model.input(0)
        self.input_shape = self.input_layer.shape
        self.height = self.input_shape[2]
        self.width = self.input_shape[3]

        print("~~~~ RESHAPED MODEL ~~~~")
        print("after resuze input shape",self.input_layer.shape)
        print("after resize height",self.input_shape[2])
        print("after resize width",self.input_shape[3])
  
        for layer in self.model.inputs:
            input_shape = layer.partial_shape
            input_shape[0] = batchsize
            self.model.reshape({layer: input_shape})

        self.compiled_model = ie_core.compile_model(model=self.model, device_name=device)
        # print(f"compiled_model input shape: "
        #     f"{self.compiled_model.input(index=0).shape}"
        # )
        self.output_layer = self.compiled_model.output(0)
        # print(f"compiled_model output shape: {self.output_layer.shape}")

        # # self.compiled_model = ie_core.compile_model(model=self.model, device_name="CPU")
        # self.output_layer_ir = self.compiled_model.output(0)
    


    def to_numpy(self,tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()



    def get_features_single(self, face):
        faces_im = []
        for f in face:
            f = cv2.resize(f, (self.width, self.height), interpolation=cv2.INTER_AREA)  # (h, w, 3)
            #assert f.shape == (self.height, self.width, 3), f"Input image shape {f.shape} does not match expected shape {(self.height, self.width, 3)}"
            f = f.transpose(2, 0, 1)                  # (3, H, W)
            f = np.float32(f) / 255    
            print('face shape',f.shape)               # convert to float and scale to [0, 1]
            f = (f - 0.5) / 0.5                      # normalize the image
            f = torch.from_numpy(f)
            faces_im.append(f)
        faces_im = torch.stack(faces_im)
        return faces_im

    def get_features(self, im_crops):
        features = []
        for face in im_crops:
            feature = self.get_features_single([face])
            features.append(feature)
        features = torch.cat(features, dim=0)
        return features


    def __call__(self, im_crops):

        img_batch = self.get_features(im_crops)
        output = self.compiled_model([img_batch])[self.output_layer]
   
        return output
