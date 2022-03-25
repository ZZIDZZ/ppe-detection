import tkinter as tk
from cv2 import VideoCapture,destroyAllWindows, cvtColor, CAP_DSHOW, COLOR_RGB2BGR, VideoWriter, VideoWriter_fourcc, imwrite
import PIL.Image, PIL.ImageTk, PIL.ImageDraw
from time import time
import numpy as np
from torch import tensor, from_numpy
import torch.backends.cudnn as cudnn
from models.common import DetectMultiBackend
from utils.general import (non_max_suppression, scale_coords,  xyxy2xywh)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from utils.augmentations import letterbox
from sys import exit
import logger
from importlib import reload

LARGEFONT = ("Verdana", 35)

VIDEO_PATH = 0

def returnCameraIndexes():
    index = 0
    arr = []
    i = 10
    while i > 0:
        cap = VideoCapture(index,CAP_DSHOW)
        if cap.read()[0]:
            arr.append(index)
            cap.release()
        index += 1
        i -= 1
    destroyAllWindows()
    return arr

class Menu(tk.Frame):
    def __init__(self, container):
        # TODO Menu: Rapikan GUI,  Add text input for api send specification
        super().__init__(container)
        print("Menu initialized")
        self.container = container
        self.frame = tk.Frame(master=container)
        self.frame.pack()
        self.ok=False
        self.vid = VideoCap()
        # self.frame = tk.Frame(container, width = self.vid.width, height = self.vid.height, bd=1)
        # self.frame.pack()

        # Create a canvas that can fit the above video source size
        self.canvas = tk.Canvas(self.frame, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()

        # Video control buttons
        videoInputs = returnCameraIndexes()
        # video input choice
        self.buttons = []
        for _, val in enumerate(videoInputs):
            self.buttons.append(tk.Button(self.frame, text=f"video({val})", command=lambda i=_:self.changeCamera(i)))
            self.buttons[_].pack( side = tk.LEFT )

        # quit button
        self.btn_quit=tk.Button(self.frame, text='QUIT', command=exit)
        self.btn_quit.pack(side=tk.LEFT)

        # Continue button
        self.btn_quit=tk.Button(self.frame, text='LANJUT', command=self.inference)
        self.btn_quit.pack(side=tk.LEFT)
        
        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay=10
        self.update()

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        if self.ok:
            self.vid.out.write(cvtColor(frame,COLOR_RGB2BGR))
        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tk.NW)
        self.after(self.delay,self.update)
    
    def inference(self):
        self.ok = False
        self.frame.destroy()
        destroyAllWindows()
        self.vid.__del__()
        self.container.switch_frame(InferenceView)
        
    def changeCamera(self, source):
        global VIDEO_PATH
        VIDEO_PATH = source
        self.ok = False
        destroyAllWindows()
        self.vid.__del__()
        self.vid = VideoCap()
        self.ok = True
        self.canvas.pack()

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self._frame = None
        self.switch_frame(Menu)
        try:
            self.tk.call("source", "tk-theme/azure.tcl")
            self.tk.call("set_theme", "light")
        except:
            pass

    def switch_frame(self, frame_class):
        new_frame = frame_class(self)
        if self._frame is not None:
            self._frame.pack_forget()
            self._frame.destroy()
        self._frame = new_frame
        self._frame.pack()

class VideoCap :
    def __init__(self):
        global VIDEO_PATH
        print("videocapture initialized")
        # Open the video source
        self.vid = VideoCapture(VIDEO_PATH)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", VIDEO_PATH)

        # 1. Video Type
        VIDEO_TYPE = {
            'avi': VideoWriter_fourcc(*'XVID'),
            #'mp4': VideoWriter_fourcc(*'H264'),
            'mp4': VideoWriter_fourcc(*'XVID'),
        }

        self.fourcc=VIDEO_TYPE['avi']

        # 2. Video Dimension
        STD_DIMENSIONS =  {
            '480p': (640, 480),
            '736p': (1280, 736),
            '864p': (1536, 864),
            '1080p': (1920, 1080),
            '4k': (3840, 2160),
        }
        res=STD_DIMENSIONS['736p']
        print('output',self.fourcc,res)
        self.out = VideoWriter('output'+'.'+'avi',self.fourcc,10,res)

        #set video sourec width and height
        self.vid.set(3,res[0])
        self.vid.set(4,res[1])

        # Get video source width and height
        self.width,self.height=res

    # To get frames
    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cvtColor(frame, COLOR_RGB2BGR))
            else:
                return (ret, None)
        else:
            return (False, None)

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
            self.out.release()
            destroyAllWindows()


class InferenceView(tk.Frame):
    global VIDEO_PATH
    def __init__(self, container):
        # also TODO InferenceView: rapikan GUI dan logger
        super().__init__(container)
        self.inferencer = Inferencer(0, "yolov5m.pt", "coco128.yaml")
        self.prev_frame_time = 0
        self.new_frame_time = 0
        self.frame = tk.Frame(master=container)
        self.frame.pack()
        self.ok=False
        self.video_source = VIDEO_PATH
        self.vid = VideoCap()
        # Create a canvas that can fit the above video source size
        self.canvas = tk.Canvas(self.frame, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()

        # quit button
        self.btn_quit=tk.Button(self.frame, text='QUIT', command=exit)
        self.btn_quit.pack(side=tk.LEFT)
        
        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay=10
        self.update()

        self.frame.mainloop()

    def update(self):

        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        if self.ok:
            self.vid.out.write(cvtColor(frame,COLOR_RGB2BGR))
        if ret:
            image = PIL.Image.fromarray(self.inferencer.inference(frame))

            self.new_frame_time = time()

            # Calculating the fps

            # fps will be number of frame processed in given time frame
            # since their will be most of time error of 0.001 second
            # we will be subtracting it to get more accurate result
            fps = 1/(self.new_frame_time-self.prev_frame_time)
            self.prev_frame_time = self.new_frame_time

            # converting the fps into integer
            fps = int(fps)

            # converting the fps to string so that we can display it on frame
            # by using putText function
            fps = str(fps)

            # write test to PIL img
            PIL.ImageDraw.Draw(image).text((0, 0), fps, fill =(0, 255, 0))

            self.photo = PIL.ImageTk.PhotoImage(image = image)
            self.canvas.create_image(0, 0, image = self.photo, anchor = tk.NW)
        self.frame.after(self.delay,self.update)

    def changeCamera(self, source):
        self.ok = False
        destroyAllWindows()
        self.video_source = source
        self.vid.__del__()
        self.vid = VideoCapture(source)
        self.ok = True

class Inferencer:
    def __init__(self, video_path, model_path, data_path, imgsz = (864,1536)):
        # TODO Inferencer: 
        # Create logger algorithm
        # Optimisasi Inferencer for more fpszzzzzzzz (if can) 
        self.video_path = video_path
        self.model_path = model_path
        self.data_path  = data_path
        self.imgsz = imgsz
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.max_det = 1000
        self.line_thickness = 3
        self.half = False
        self.hide_conf = False
        self.detected, self.prev_detected = {},{}
        self.time = round(time(), 2)
        # Open the video source
        self.vid = VideoCapture(video_path)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_path)

        # Load model
        self.device = select_device()
        self.model = DetectMultiBackend(self.model_path, device=self.device, data=self.data_path)
        self.stride, self.names, self.pt, = self.model.stride, self.model.names, self.model.pt
        self.model.warmup(imgsz=(1, 3, *imgsz), half=False)
        cudnn.benchmark = True

    # adapted from yolov5's detect.py, thanks yolov5
    # pls optimize me
    def inference(self, frame:np.ndarray):
        model = self.model
        names = self.names

        im = letterbox(frame, self.imgsz, stride=self.stride, auto=True)[0]
        im = im.transpose((2, 0, 1))[::-1]
        im = np.ascontiguousarray(im)

        # what is this im and im0
        im = from_numpy(im).to(self.device)
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        pred = model(im, augment=False, visualize=False)

        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, max_det=self.max_det)

        det = pred[0]
        # Process predictions
        gn = tensor(frame.shape)[[1, 0, 1, 0]]

        annotator = Annotator(frame, line_width=self.line_thickness, example=str(names))
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], frame.shape).round()

            # logger
            reload(logger)
            try:
                if round(time(), 2) - self.time > 0.2:
                    self.detected, self.prev_detected= logger.log(det,names, self.detected,self.prev_detected)
                    self.time =round(time(), 2)
            except Exception as e:
                print(e)
                pass

            # for *xyxy, conf, cls in reversed(det):
            #     xywh = (xyxy2xywh(tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            #     line = (cls, *xywh, conf)
            #     with open("logs/asdlog" + '.txt', 'a') as f:
            #         out = f"{(str(line[0]).rstrip(), cls)} " +"\n"
            #         f.write(out)
            #         # print(out)

            # Write results
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label = names[c] if self.hide_conf else f'{names[c]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors(c, True))
        if self.time - round(time(), 2) > 10:
            self.time = round(time(), 2)
        # Stream results
        return annotator.result()


if __name__ == "__main__":
    app = App()
    app.mainloop()