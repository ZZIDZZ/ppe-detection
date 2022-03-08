from tkinter import *
import cv2
from PIL import Image, ImageTk

class MainWindow:
    def __init__(self):
        self.window = Tk()
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

def show_frames(label,videoInput = 0):
    cap = cv2.VideoCapture(videoInput)
    _, frame = cap.read()
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)
    label.after(1, show_frames) 


def chooseVideo():
    screen = Tk()
    label =Label(screen)
    screen.geometry("640x640")
    frame = Frame(screen)
    frame.pack()
    videoInputs = returnCameraIndexes()
    buttons = []
    for i, val in enumerate(videoInputs):
        buttons.append(Button(frame, text=val, fg="blue",command=show_frames(label, i)))
        buttons[i].pack( side = LEFT )
    screen.mainloop()

chooseVideo()