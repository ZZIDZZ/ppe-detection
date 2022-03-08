from statistics import mode
from turtle import st
import torch
import cv2
import threading
from pathlib import Path
import sys, os


import torch.backends.cudnn as cudnn

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadStreams, LoadImages
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync




import tkinter as tk
import tkinter.messagebox as tkmsg
from PIL import ImageGrab, Image, ImageTk

class MainWindow(tk.Frame):
    """Main window"""
    def __init__(self, master = None):
        super().__init__(master)
        self.pack()

        self.file_name_var = tk.StringVar()
        self.ean_code_var = tk.StringVar()

        self.create_widgets()


    def create_widgets(self):
        self.questioning_label = tk.Label(self,
                                        text = 'Bisa ga ya?',
                                        justify='center',
                                        font=('Arial', 20, 'bold'))
        self.canvas = VideoCanvas(master=self)
        self.btn_exit = tk.Button(self,
                                  text = "run",
                                  command = self.canvas.aurun)

        self.questioning_label.pack()
        self.canvas.pack()
        self.btn_exit.pack()

    def popup_error_message(self, errmsg):
        """For popup error message"""
        tkmsg.showerror(title="Wrong Input!",
                        message=errmsg)

class VideoCanvas(tk.Canvas):
    """canvas untuk menggambar ean13 barcode"""

    @property
    def imagearr(self):
        return self._imagearr

    @imagearr.setter # trigger ada perubahan nilai di property imagearr
    def imagearr(self, imagearr):
        print("trigger")
        self._imagearr = imagearr
        imgtk = ImageTk.PhotoImage(image=imagearr)
        self.create_image(0, 0, image=imgtk)
        self.pack()
        

    def __init__(self, master = None):
        super().__init__(master, bg='white', height=640, width=640)

        weights = "./last.pt"
        data = "./palo.yaml"
        dnn = False

        device = select_device()
        self.model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
        self.device = device
        self.webcam = False

    def aurun(self):
        url = "./palo.mp4"
        device = self.device
        conf_thres = 0.25
        iou_thres = 0.45
        classes = None
        max_det = self.max_det = 1000
        line_thickness = self.line_thickness = 3
        model = self.model
        webcam = self.webcam
        half = False
        hide_conf = self.hide_conf = False
        imgsz=(640, 640)

    
        stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # view_img = check_imshow()
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(url, img_size=imgsz, stride=stride, auto=pt)
            bs = len(dataset)  # batch_size
        else:
            dataset = LoadImages(url, img_size=imgsz, stride=stride, auto=pt)
            bs = 1  # batch_size

        # Run inference
        model.warmup(imgsz=(1 if pt else bs, 3, *imgsz), half=False)  # warmup
        dt, seen = [0.0, 0.0, 0.0], 0
        for path, im, im0s, vid_cap, s in dataset:
            t1 = time_sync()
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            visualize = False
            pred = model(im, augment=False, visualize=visualize)
            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, max_det=max_det)
            dt[2] += time_sync() - t3

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                # save_path = str(save_dir / p.name)  # im.jpg
                # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                # s += '%gx%g ' % im.shape[2:]  # print string
                # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                # imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # # Write results
                    for *xyxy, conf, cls in reversed(det):
                    #     if save_txt:  # Write to file
                    #         xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    #         line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    #         with open(txt_path + '.txt', 'a') as f:
                    #             f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        # if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = names[c] if hide_conf else f'{names[c]} {conf:.2f}'
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        # if save_crop:
                        #     save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                # Stream results
                im0 = annotator.result()
                blue,green,red = cv2.split(im0)
                img = cv2.merge((red,green,blue))

                # trigger
                self.imagearr = Image.fromarray(img)
                
                # trigger
                # cv2.imshow(str(p), im0)
                # cv2.waitKey(1)  # 1 millisecond

            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

            # Print results
            t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
            LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
            # if save_txt or save_img:
            #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            #     LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
            # if update:
            #     strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    

def main():
    m = MainWindow()
    m.master.title("UWOOGHHH")
    m.master.mainloop()

if __name__ == "__main__":
    main()