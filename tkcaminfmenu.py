import tkinter as tk
import cv2
import PIL.Image, PIL.ImageTk, PIL.ImageDraw
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

class Menu(tk.Frame):
    def __init__(self, parent, controller, video_source=0):
        print("Menu initialized")
        tk.Frame.__init__(self,parent)
        label = ttk.Label(self, text ="Startpage", font = LARGEFONT)
        label.pack(pady=10,padx=10)
        self.video_source = video_source
        self.ok=False
        self.vid = VideoCapture(self.video_source)
        self.canvas = tk.Canvas(self, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()
        # Create a canvas that can fit the above video source size
        # Button that lets the user take a snapshot

        videoInputs = returnCameraIndexes()
        self.buttons = []
        for _, val in enumerate(videoInputs):
            self.buttons.append(tk.Button(self.canvas, text=f"video({val})", command=lambda i=_:self.changeCamera(i)))
            self.buttons[_].pack( side = tk.LEFT )
        #video control buttons

        # quit button
        self.btn_quit=tk.Button(self.canvas, text='QUIT', command=quit)
        self.btn_quit.pack(side=tk.LEFT)

        self.btn_quit=tk.Button(self.canvas, text='LANJUT', command=lambda : controller.show_frame(Inferencer))
        self.btn_quit.pack(side=tk.LEFT)
        
        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay=10
        self.update()

        self.mainloop()

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        if self.ok:
            self.vid.out.write(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(master = self.canvas,image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tk.NW)
        self.after(self.delay,self.update)
    
    def inference(self):
        return

    def changeCamera(self, source):
        self.ok = False
        cv2.destroyAllWindows()
        self.video_source = source
        self.vid.__del__()
        self.vid = VideoCapture(source)
        self.ok = True


class Inferencer(tk.Frame):
    def __init__(self, video_path, model_path, data_path, imgsz = (736,1280), tkFrame = tk.Frame()):
        print("Inferencer initialized")
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
        self.prev_frame_time = 0
        self.new_frame_time = 0
        # self.window = window
        # self.window.title(window_title)
        # self.video_source = video_source
        self.ok=False
        self.vid = VideoCapture(self.video_source)
        self.canvas = tk.Canvas(self.window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()
        
        # Open the video source
        self.vid = cv2.VideoCapture(video_path)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_path)

        # Load model
        self.device = select_device()
        self.model = DetectMultiBackend(self.model_path, device=self.device, data=self.data_path)
        self.stride, self.names, self.pt, = self.model.stride, self.model.names, self.model.pt
        self.model.warmup(imgsz=(1, 3, *imgsz), half=False)
        cudnn.benchmark = True

    # adapted from yolov5's detect.py, thanks yolov5
    def inference(self, frame:np.ndarray):
        model = self.model
        names = self.names

        im = letterbox(frame, self.imgsz, stride=self.stride, auto=True)[0]
        im = im.transpose((2, 0, 1))[::-1]
        im = np.ascontiguousarray(im)

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

            # # Write results
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label = names[c] if self.hide_conf else f'{names[c]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors(c, True))

        # Stream results
        return annotator.result()

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        if self.ok:
            self.vid.out.write(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
        if ret:
            image = PIL.Image.fromarray(self.inference(frame))
            self.new_frame_time = time.time()
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
        self.window.after(self.delay,self.update)


class App(tk.Tk):
    def __init__(self, video_source=1, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        container = tk.Frame(self) 
        container.pack(side = "top", fill = "both", expand = True)
        # self.inferencer = Inferencer(0, "best.pt", "ppe3.yaml")
        self.video_source = video_source
        # self.ok=False
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
        self.window.mainloop()

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


class VideoCapture:
    def __init__(self, video_source=0):
        print("videocapture initialized")
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Command Line Parser
        args=CommandLineParser().args
        
        #create videowriter

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
            '1080p': (1920, 1080),
            '4k': (3840, 2160),
        }
        res=STD_DIMENSIONS[args.res[0]]
        print(args.name,self.fourcc,res)
        self.out = cv2.VideoWriter(args.name[0]+'.'+args.type[0],self.fourcc,10,res)

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
        parser.add_argument('--res', nargs=1, default=['480p'], type=str, help='Resolution of the video output: for now we have 480p, 720p, 1080p & 4k')

        # Only one values are going to accept for the tag --name. So nargs will be '1'
        parser.add_argument('--name', nargs=1, default=['output'], type=str, help='Enter Output video title/name')

        # Parse the arguments and get all the values in the form of namespace.
        # Here args is of namespace and values will be accessed through tag names
        self.args = parser.parse_args()


if __name__ == "__main__":
    app = App()
    app.mainloop()