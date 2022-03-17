import tkinter as tk
import cv2
import PIL.Image, PIL.ImageTk
import time
import datetime as dt
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tkinter import ttk
from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadStreams, LoadImages
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from utils.augmentations import letterbox

LARGEFONT =("Verdana", 35)
def returnCameraIndexes():
    index = 0
    arr = []
    i = 10
    while i > 0:
        cap = cv2.VideoCapture(index,cv2.CAP_DSHOW)
        if cap.read()[0]:
            arr.append(index)
            cap.release()
        index += 1
        i -= 1
    cv2.destroyAllWindows()
    return arr

class App(tk.Tk):
    def __init__(self,  video_source=1,master=None):
        super().__init__(master)
        container = tk.Frame(self) 
        container.pack(side = "top", fill = "both", expand = True)
  
        container.grid_rowconfigure(0, weight = 1)
        container.grid_columnconfigure(0, weight = 1)
  
        # initializing frames to an empty array
        self.frames = {} 
  
        # iterating through a tuple consisting
        # of the different page layouts
        for F in (Menu, Inferencer):
  
            frame = F(container, self)
  
            # initializing frame of that object from
            # Menu, page1, page2 respectively with
            # for loop
            self.frames[F] = frame
  
            frame.grid(row = 0, column = 0, sticky ="nsew")
  
        self.show_frame(Menu)
        self.update()

        self.window.mainloop()

    def snapshot(self):
        # Get a frame from the video source
        ret,frame=self.vid.get_frame()
        if ret:
            cv2.imwrite("frame-"+time.strftime("%d-%m-%Y-%H-%M-%S")+".jpg",cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))

    def open_camera(self):
        self.ok = True
        self.timer.start()
        print("camera opened => Recording")

    def close_camera(self):
        self.ok = False
        self.timer.stop()
        print("camera closed => Not Recording")

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        if self.ok:
            self.vid.out.write(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tk.NW)
        self.window.after(self.delay,self.update)

    def changeCamera(self, source):
        self.ok = False
        cv2.destroyAllWindows()
        self.video_source = source
        self.vid.__del__()
        self.vid = VideoCapture(source)
        self.ok = True
    
    def inference(self):
        print("inferencing")
        cv2.destroyAllWindows()
    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

class Menu(tk.Frame):
    def __init__(self, controller, video_source = 1, master = None):
        super().__init__(master)
        label = ttk.Label(self, text ="Startpage", font = LARGEFONT)
        self.video_source = video_source
        self.ok=False
        self.vid = VideoCapture(self.video_source)
        self.canvas = tk.Canvas(self, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()
        # Create a canvas that can fit the above video source size
        # Button that lets the user take a snapshot
        self.btn_snapshot=tk.Button(self, text="Snapshot", command=self.snapshot)
        self.btn_snapshot.pack(side=tk.LEFT)

        videoInputs = returnCameraIndexes()
        self.buttons = []
        for _, val in enumerate(videoInputs):
            self.buttons.append(tk.Button(self, text=f"video({val})", command=lambda i=_:self.changeCamera(i)))
            self.buttons[_].pack( side = tk.LEFT )
        #video control buttons

        self.btn_start=tk.Button(self, text='START', command=self.open_camera)
        self.btn_start.pack(side=tk.LEFT)

        self.btn_stop=tk.Button(self, text='STOP', command=self.close_camera)
        self.btn_stop.pack(side=tk.LEFT)

        # quit button
        self.btn_quit=tk.Button(self, text='QUIT', command=quit)
        self.btn_quit.pack(side=tk.LEFT)

        self.btn_quit=tk.Button(self, text='LANJUT', command=lambda : controller.show_frame(Inferencer))
        self.btn_quit.pack(side=tk.LEFT)
        
        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay=10
        self.update()

        self.mainloop()

    def snapshot(self):
        # Get a frame from the video source
        ret,frame=self.vid.get_frame()
        if ret:
            cv2.imwrite("frame-"+time.strftime("%d-%m-%Y-%H-%M-%S")+".jpg",cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))

    def open_camera(self):
        self.ok = True
        self.timer.start()
        print("camera opened => Recording")

    def close_camera(self):
        self.ok = False
        self.timer.stop()
        print("camera closed => Not Recording")

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        if self.ok:
            self.vid.out.write(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tk.NW)
        self.window.after(self.delay,self.update)

    def changeCamera(self, source):
        self.ok = False
        cv2.destroyAllWindows()
        self.video_source = source
        self.vid.__del__()
        self.vid = VideoCapture(source)
        self.ok = True
    

class VideoCapture:
    def __init__(self, video_source=0):

        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Command Line Parser
        args=CommandLineParser().args

        # 1. Video Type
        VIDEO_TYPE = {
            'avi': cv2.VideoWriter_fourcc(*'XVID'),
            #'mp4': cv2.VideoWriter_fourcc(*'H264'),
            'mp4': cv2.VideoWriter_fourcc(*'XVID'),
        }
        self.fourcc=VIDEO_TYPE['avi']

        # 2. Video Dimension
        STD_DIMENSIONS =  {
            '480p': (640, 480),
            '720p': (1280, 720),
            '900p': (1600,900),
            '1080p': (1920, 1080),
            '4k': (3840, 2160),
        }
        res=STD_DIMENSIONS[args.res[0]]
        print(args.name,self.fourcc,res)
        self.out = cv2.VideoWriter(args.name[0]+'.'+args.type[0],self.fourcc,20,res)

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
                return (ret, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            else:
                return (ret, None)
        else:
            return (ret, None)

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
            self.out.release()
            cv2.destroyAllWindows()

class CommandLineParser:
    
    def __init__(self):

        # Create object of the Argument Parser
        parser=argparse.ArgumentParser(description='Script to record videos')

        # Create a group for requirement 
        # for now no required arguments 
        # required_arguments=parser.add_argument_group('Required command line arguments')
        # Only values is supporting for the tag --type. So nargs will be '1' to get
        parser.add_argument('--type', nargs=1, default=['avi'], type=str, help='Type of the video output: for now we have only AVI & MP4')
        # Only one values are going to accept for the tag --res. So nargs will be '1'
        parser.add_argument('--res', nargs=1, default=['720p'], type=str, help='Resolution of the video output: for now we have 480p, 720p, 1080p & 4k')
        # Only one values are going to accept for the tag --name. So nargs will be '1'
        parser.add_argument('--name', nargs=1, default=['output'], type=str, help='Enter Output video title/name')
        # Parse the arguments and get all the values in the form of namespace.
        # Here args is of namespace and values will be accessed through tag names
        self.args = parser.parse_args()

class Inferencer:
    conf_thres = 0.25
    iou_thres = 0.45
    max_det = 1000
    line_thickness = 3
    half = False
    hide_conf = False

    def __init__(self, video_path, model_path, imgsz = (640,640)):
        self.video_path = video_path
        self.model_path = model_path
        self.imgsz = imgsz
        # Open the video source
        self.vid = cv2.VideoCapture(video_path)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_path)

        # Command Line Parser
        args=CommandLineParser().args

        # 1. Video Type
        VIDEO_TYPE = {
            'avi': cv2.VideoWriter_fourcc(*'XVID'),
            #'mp4': cv2.VideoWriter_fourcc(*'H264'),
            'mp4': cv2.VideoWriter_fourcc(*'XVID'),
        }
        self.fourcc=VIDEO_TYPE['avi']

        # 2. Video Dimension
        STD_DIMENSIONS =  {
            '480p': (640, 480),
            '720p': (1280, 720),
            '900p': (1600,900),
            '1080p': (1920, 1080),
            '4k': (3840, 2160),
        }
        res=STD_DIMENSIONS[args.res[0]]
        print(args.name,self.fourcc,res)
        self.out = cv2.VideoWriter(args.name[0]+'.'+args.type[0],self.fourcc,20,res)

        #set video sourec width and height
        self.vid.set(3,res[0])
        self.vid.set(4,res[1])

        # Get video source width and height
        self.width,self.height=res

        # Load model
        self.device = select_device()
        self.model = DetectMultiBackend(self.model_path, device=self.device)
        self.stride, self.names, self.pt, = self.model.stride, self.model.names, self.model.pt
        self.model.warmup(imgsz=(1, 3, *imgsz), half=False)
        cudnn.benchmark = True

    def inference(self, frame:np.ndarray):
        model = self.model
        names = self.names
        

        im = letterbox(frame, self.img_size, stride=self.stride, auto=self.auto)[0]
        im = im.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)

        # what is this im and im0
        im = torch.from_numpy(im).to(self.device)
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
        gn = torch.tensor(frame.shape)[[1, 0, 1, 0]]

        annotator = Annotator(frame, line_width=self.line_thickness, example=str(names))
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], frame.shape).round()

            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh, conf)
                with open("asdlog" + '.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # # Write results
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label = names[c] if self.hide_conf else f'{names[c]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors(c, True))

        # Stream results
        return annotator.result()        


def main():
    # Create a window and pass it to the Application object
    App(1)

main()