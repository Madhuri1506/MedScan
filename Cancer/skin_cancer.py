import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tkinter.font
from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import cv2
import tensorflow as tf

class SkinCancer:
    def __init__(self, frame):
        self.frame = frame
        self.skin_cancer_model = tf.keras.models.load_model(r'Cancer/cancer1.h5')
        self.filename = [r"common/NAN.jpg"]
        self.dir = r"C:/"


        #Example Section
        self.width = 110
        self.height = 110
        self.img1_dir = ImageTk.PhotoImage(Image.open(r"Cancer/mal (1).jpg").resize((110, 110)), Image.ANTIALIAS)
        self.img1 = Label(self.frame, image=self.img1_dir, width=self.width, height=self.height)
        self.img2_dir = ImageTk.PhotoImage(Image.open(r"Cancer/mal (1).jpg").resize((110, 110)), Image.ANTIALIAS)
        self.img2 = Label(self.frame, image=self.img2_dir, width=self.width, height=self.height)
        self.img3_dir = ImageTk.PhotoImage(Image.open(r"Cancer/mal (1).jpg").resize((110, 110)), Image.ANTIALIAS)
        self.img3 = Label(self.frame, image=self.img3_dir, width=self.width, height=self.height)

        self.label1 = Label(self.frame, text="INPUT EXAMPLE", font=tkinter.font.Font(family="Helvetica", size=15, weight="bold"))
        self.img1_label = Label(self.frame, text="Example 1", font=tkinter.font.Font(family="Helvetica", size=12, weight="normal"))
        self.img2_label = Label(self.frame, text="Example 2", font=tkinter.font.Font(family="Helvetica", size=12, weight="normal"))
        self.img3_label = Label(self.frame, text="Example 3", font=tkinter.font.Font(family="Helvetica", size=12, weight="normal"))

        self.img1_tag = Label(self.frame, text="Malignant", font=tkinter.font.Font(family="Helvetica", size=9, weight="normal"))
        self.img2_tag = Label(self.frame, text="Benign", font=tkinter.font.Font(family="Helvetica", size=9, weight="normal"))
        self.img3_tag = Label(self.frame, text="Malignant", font=tkinter.font.Font(family="Helvetica", size=9, weight="normal"))

        self.label1.place(x=175, y=10)
        self.img1_label.place(x=40, y=50)
        self.img2_label.place(x=220, y=50)
        self.img3_label.place(x=400, y=50)
        self.img1.place(x=20, y=75)
        self.img2.place(x=200, y=75)
        self.img3.place(x=380, y=75)
        self.img1_tag.place(x=55, y=190)
        self.img2_tag.place(x=235, y=190)
        self.img3_tag.place(x=410, y=190)

        # About Section
        self.label6i = Label(self.frame, text="About Malignant and Benign",
                          font=tkinter.font.Font(family="Helvetica", size=15, weight="bold"))
        self.label6i.place(x=135, y=220)
        self.label6 = Label(self.frame, wraplength=500, justify=LEFT, text="""Malignant
Our bodies constantly produce new cells to replace old ones. Sometimes, DNA gets damaged in the process, so new cells develop abnormally. Instead of dying off, they continue to multiply faster than the immune system can handle, forming a tumor.
Cancer cells can break away from tumors and travel through the bloodstream or lymphatic system to other parts of the body.

Benign
Benign tumors aren’t cancerous. They won’t invade surrounding tissue or spread elsewhere.
Even so, they can cause serious problems when they grow near vital organs, press on a nerve, or restrict blood flow. Benign tumors usually respond well to treatment.
""")
        self.label6.place(x=5, y=250)


        # treatment Section
        self.label9i = Label(self.frame, text="Treatment", font=tkinter.font.Font(family="Helvetica", size=15, weight="bold"))
        self.label9 = Label(self.frame, wraplength=500, justify=LEFT, text="""Malignant
Treatment for cancerous tumors depends on many factors, such as where the primary tumor is located and whether it’s spread. A pathology report can reveal specific information about the tumor to help guide treatment, which may include:
surgery
radiation therapy
chemotherapy
targeted therapy
immunotherapy, also known as biological therapy

Benign
In many cases, benign tumors need no treatment. Doctors may simply use "watchful waiting" to make sure they cause no problems. But treatment may be needed if symptoms are a problem. Surgery is a common type of treatment for benign tumors. The goal is to remove the tumor without damaging surrounding tissues. Other types of treatment may include medication or radiation.
""")
        self.label9.place(x=5, y=461)
        self.label9i.place(x=185, y=435)

        # right Section
        self.label10 = Label(self.frame, text="MALIGNANT/BENIGN PREDICTION",
                          font=tkinter.font.Font(family="Helvetica", size=15, weight="bold"))
        self.label10.place(x=595, y=10)

        self.Button1 = Button(self.frame, text="Upload Image", command=self.getfile,
                           font=tkinter.font.Font(family="Helvetica", size=15, weight="bold"), width=13, height=1,
                           relief="solid", activebackground="white")
        self.Button1.place(x=580, y=270)
        self.Button1 = Button(self.frame, text="Upload Image", command=self.getfile,
                           font=tkinter.font.Font(family="Helvetica", size=15, weight="bold"), width=13, height=1,
                           relief="solid", activebackground="white")
        self.Button1.place(x=580, y=270)

        self.label13 = Label(self.frame, text="File name:", font=tkinter.font.Font(family="Helvetica", size=14, weight="bold"))
        self.label13.place(x=580, y=350)

        self.label14 = Label(self.frame, text="Prediction:", font=tkinter.font.Font(family="Helvetica", size=14, weight="bold"))
        self.label14.place(x=580, y=390)

        self.label15 = Label(self.frame, text="Accuracy:", font=tkinter.font.Font(family="Helvetica", size=14, weight="bold"))
        self.label15.place(x=580, y=430)

        self.img_nan = r"common/NAN.jpg"
        self.img5i = ImageTk.PhotoImage(Image.open(self.img_nan).resize((250, 200), Image.ANTIALIAS))
        self.img5label = Label(self.frame, image=self.img5i, height=200, width=250, relief="solid")
        self.img5label.place(x=635, y=50)

        self.label16 = Label(self.frame, text="Benign:",
                             font=tkinter.font.Font(family="Helvetica", size=14, weight="bold"))
        self.label16.place(x=580, y=485)

        self.label17 = Label(self.frame, text="Malignant:",
                             font=tkinter.font.Font(family="Helvetica", size=14, weight="bold"))
        self.label17.place(x=580, y=525)


        # model
        try:
            self.Button2 = Button(self.frame, text="Predict", command=lambda: self.predict(self.filename[0]),
                               font=tkinter.font.Font(family="Helvetica", size=15, weight="bold"), width=13, height=1,
                               relief="solid", activebackground="white")
            self.Button2.place(x=800, y=270)
        except:
            print("error")

        try:
            self.label_predict = Label(self.frame, text="NaN", font=tkinter.font.Font(family="Helvetica", size=14, weight="bold"))
            self.label11 = Label(self.frame, text=predictionlabel,
                              font=tkinter.font.Font(family="Helvetica", size=14, weight="bold"))
            self.label12 = Label(self.frame, text="NaN", font=tkinter.font.Font(family="Helvetica", size=14, weight="bold"))
            self.label12i = Label(self.frame, text="NaN", font=tkinter.font.Font(family="Helvetica", size=14, weight="bold"))
            self.label12ii = Label(self.frame, text="NaN", font=tkinter.font.Font(family="Helvetica", size=14, weight="bold"))
        except:
            self.label11 = Label(self.frame, text="NaN", font=tkinter.font.Font(family="Helvetica", size=14, weight="bold"))
            self.label12 = Label(self.frame, text="NaN", font=tkinter.font.Font(family="Helvetica", size=14, weight="bold"))
            self.label_predict = Label(self.frame, text="NaN", font=tkinter.font.Font(family="Helvetica", size=14, weight="bold"))
            self.label12i = Label(self.frame, text="NaN",
                                  font=tkinter.font.Font(family="Helvetica", size=14, weight="bold"))
            self.label12ii = Label(self.frame, text="NaN",
                                   font=tkinter.font.Font(family="Helvetica", size=14, weight="bold"))
        self.label_predict.place(x=690, y=350)
        self.label11.place(x=690, y=390)
        self.label12.place(x=690, y=430)
        self.label12i.place(x=690,y=485)
        self.label12ii.place(x=690,y=525)

        # border
        self.border = Label(self.frame, text="", bg="black", bd=1, height=745)
        self.border.place(x=510, y=0)

        self.img4 = ImageTk.PhotoImage(Image.open(r"Cancer/normal1.png").resize((20, 1)), Image.ANTIALIAS)
        self.border1 = Label(self.frame, image=self.img4, bg="black", bd=1, width=510, height=1)
        self.border1.place(x=0, y=210)
        self.img5 = ImageTk.PhotoImage(Image.open(r"Cancer/normal1.png").resize((20, 1)), Image.ANTIALIAS)
        self.border2 = Label(self.frame, image=self.img5, bg="black", bd=1, width=510, height=1)
        self.border2.place(x=512, y=475)



    def callback2(self):
        img5 = ImageTk.PhotoImage(Image.open(self.filename[0]).resize((250, 200), Image.ANTIALIAS))
        self.img5label.configure(image=img5)
        self.img5label.image = img5


    def getfile(self):
        try:
            temp = self.filename[0]
            self.filename[0] = filedialog.askopenfilename(initialdir=self.dir, title="Select A File", filetypes=(
                ("jpg file", "*.jpg"), ("png files", "*.png"), ("jpeg file", "*.jpeg")))
            self.dir = self.get_dir()
            if len(self.filename[0])!=0:
                self.label11.config(text="NaN")
                self.label12.config(text="NaN")
                self.label12i.config(text="NaN")
                self.label12ii.config(text="NaN")

            if len(self.filename[0])==0:
                self.filename[0] = temp
                self.dir = self.get_dir()

        except:
            self.filename[0] = r"common/NAN.jpg"
        self.callback2()
        self.callback1()


    def callback1(self):
        filename3 = self.filename[0][::-1]
        filename4 = ""
        filename5 = "/"
        filename4 = filename3.find(filename5)
        filename4 = filename3[0:filename4]
        filename4 = filename4[::-1]
        self.label_predict.config(text=filename4)


    def predictionlabel1(self):
        self.label11.config(text=predictionlabel)
        self.label12.config(text=str(predictionAccuracy) + "%")
        self.label12i.config(text=str(round(prediction[0]*100, 2))+"%")
        self.label12ii.config(text=str(round(prediction[1]*100, 2))+"%")


    def predict(self,img):
        global prediction
        global predictionlabel
        global predictionAccuracy
        image_array2 = cv2.imread(img)
        new_array2 = cv2.resize(image_array2, (200, 200))
        new_array2 = new_array2.reshape(-1, 200, 200, 3)
        new_array2 = [new_array2 / 255.0]
        prediction = self.skin_cancer_model.predict(new_array2)
        prediction = prediction[0]
        benign = prediction[0]
        malignant = prediction[1]
        if malignant >= benign:
            predictionlabel = "Malignant"
            predictionAccuracy = malignant * 100
        else:
            predictionlabel = "Benign"
            predictionAccuracy = benign * 100
        predictionAccuracy = round(predictionAccuracy, 2)
        self.predictionlabel1()

    def get_dir(self):
        if len(self.filename[0])!=0:
            filename3 = self.filename[0][::-1]
            filename4 = ""
            filename5 = "/"
            filename4 = filename3.find(filename5)
            filename4 = filename3[filename4:]
            filename4 = filename4[::-1]
            print(filename4)
            return filename4
        else:
            return 0





