import tkinter.font
from tkinter import *
from PIL import ImageTk, Image




class Main:
    def __init__(self, frame):
        self.frame = frame
        self.img1_dir = ImageTk.PhotoImage(Image.open(r"common/icon.PNG").resize((100, 100)), Image.ANTIALIAS)
        self.img1 = Label(self.frame, image=self.img1_dir, width=150, height=150)
        self.img1.place(x=100, y=30)
        self.label1 = Label(self.frame, text="MEDSCAN",
                              font=tkinter.font.Font(family="Helvetica", size=100, weight="bold"), fg="blue")
        self.label1.place(x=180, y=30)
        self.label2 = Label(self.frame,
                              text="Diagnose your  MRI, CITY SCAN with us!  we  use  modern machine learning algorithms to predict out the best possible disease. output from   model  is  not  100%   true,  it  all  depends  on  the  accuracy of model, data   on which   model is   train   on,  quality   of training data,   quality of input image, type of disease and other factors.",
                              font=tkinter.font.Font(family="Helvetica", size=20, weight="bold"), fg="black",
                              wraplength=900,
                              justify=LEFT)
        self.label2.place(x=50, y=200)


