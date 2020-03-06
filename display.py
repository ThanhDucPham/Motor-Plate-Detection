from tkinter import *
import tkinter.filedialog as fdialog
import cv2
from predict import predict_plate
from PIL import ImageTk, Image

a= Tk()
canv = Canvas(a, width=80, height=80, bg='white')
canv.grid(row=2, column=3)


def mfileopen() :
    file = fdialog.askopenfile(initialdir = "F:\HUST\Python\py-app\py\ImageProcessing\IP\GreenParking")
    img = ImageTk.PhotoImage(Image.open(file.name))
    canv.create_image(20, 20, anchor=NW, image=img)


def mpredict() :
    # predict_plate(mfileopen)
    file = mfileopen()
    # predict_plate(file.name)

button1 = Button(text = "chose image",  command = mfileopen)
button2 = Button(text = "predict",  command = mpredict)
a.mainloop()