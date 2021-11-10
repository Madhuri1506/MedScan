import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tkinter.font
from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import cv2
import tensorflow as tf

class BrainTumor:
    def __init__(self, frame):
        self.frame = frame
        self.brain_tumor_model = tf.keras.models.load_model(r'brain_tumor/tumrmodel.h5')
        self.filename = [r"common/NAN.jpg"]
        self.dir = r"C:/"


        #Example Section
        self.width = 110
        self.height = 110
        self.img1_dir = ImageTk.PhotoImage(Image.open(r"brain_tumor/Y251.JPG").resize((110, 110)), Image.ANTIALIAS)
        self.img1 = Label(self.frame, image=self.img1_dir, width=self.width, height=self.height)
        self.img2_dir = ImageTk.PhotoImage(Image.open(r"brain_tumor/18 no.jpg").resize((110, 110)), Image.ANTIALIAS)
        self.img2 = Label(self.frame, image=self.img2_dir, width=self.width, height=self.height)
        self.img3_dir = ImageTk.PhotoImage(Image.open(r"brain_tumor/Y258.JPG").resize((110, 110)), Image.ANTIALIAS)
        self.img3 = Label(self.frame, image=self.img3_dir, width=self.width, height=self.height)

        self.label1 = Label(self.frame, text="INPUT EXAMPLE", font=tkinter.font.Font(family="Helvetica", size=15, weight="bold"))
        self.img1_label = Label(self.frame, text="Example 1", font=tkinter.font.Font(family="Helvetica", size=12, weight="normal"))
        self.img2_label = Label(self.frame, text="Example 2", font=tkinter.font.Font(family="Helvetica", size=12, weight="normal"))
        self.img3_label = Label(self.frame, text="Example 3", font=tkinter.font.Font(family="Helvetica", size=12, weight="normal"))

        self.img1_tag = Label(self.frame, text="Tumor", font=tkinter.font.Font(family="Helvetica", size=9, weight="normal"))
        self.img2_tag = Label(self.frame, text="Normal", font=tkinter.font.Font(family="Helvetica", size=9, weight="normal"))
        self.img3_tag = Label(self.frame, text="Tumor", font=tkinter.font.Font(family="Helvetica", size=9, weight="normal"))

        self.label1.place(x=175, y=10)
        self.img1_label.place(x=40, y=50)
        self.img2_label.place(x=220, y=50)
        self.img3_label.place(x=400, y=50)
        self.img1.place(x=20, y=75)
        self.img2.place(x=200, y=75)
        self.img3.place(x=380, y=75)
        self.img1_tag.place(x=55, y=190)
        self.img2_tag.place(x=235, y=190)
        self.img3_tag.place(x=415, y=190)

        # About Section
        self.label6i = Label(self.frame, text="About Brain Tumor",
                          font=tkinter.font.Font(family="Helvetica", size=15, weight="bold"))
        self.label6i.place(x=150, y=220)
        self.label6 = Label(self.frame, wraplength=500, justify=LEFT,
                         text="""A cancerous or non-cancerous mass or growth of abnormal cells in the brain.Tumours can start in the brain, or cancer elsewhere in the body can spread to the brain.Symptoms include new or increasingly strong headaches, blurred vision, loss of balance, confusion and seizures. In some cases, there may be no symptoms.""")
        self.label6.place(x=5, y=250)

        # symptoms Section
        self.label7i = Label(self.frame, text="Symptoms", font=tkinter.font.Font(family="Helvetica", size=15, weight="bold"))

        self.label7 = Label(self.frame, wraplength=500, justify=LEFT, text="""Headache: can be acute or persistent
Muscular: difficulty walking, instability, muscle weakness, problems with coordination, weakness of one side of the body, or weakness of the arms and legs
Whole body: dizziness, fatigue, or vertigo
Gastrointestinal: nausea or vomiting
Sensory: pins and needles or reduced sensation of touch
Cognitive: inability to speak or understand language or mental confusion
Also common: blurred vision, difficulty speaking, personality change, seizures, or sleepiness
""")
        self.label7.place(x=6, y=360)
        self.label7i.place(x=185, y=330)

        # treatment Section
        self.label9i = Label(self.frame, text="Treatment", font=tkinter.font.Font(family="Helvetica", size=15, weight="bold"))
        self.label9 = Label(self.frame, wraplength=500, justify=LEFT, text="""Treatment depends on stage
Treatments include surgery, radiation and chemotherapy.
Medications: Chemotherapy
Surgery: Craniotomy
Medical procedure: Tomotherapy and Radiation therapy""")
        self.label9.place(x=5, y=526)
        self.label9i.place(x=185, y=500)

        # right Section
        self.label10 = Label(self.frame, text="BRAIN TUMOR PREDICTION",
                          font=tkinter.font.Font(family="Helvetica", size=15, weight="bold"))
        self.label10.place(x=625, y=10)

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

        self.label16 = Label(self.frame, text="Normal:",
                             font=tkinter.font.Font(family="Helvetica", size=14, weight="bold"))
        self.label16.place(x=580, y=485)

        self.label17 = Label(self.frame, text="Tumor:",
                             font=tkinter.font.Font(family="Helvetica", size=14, weight="bold"))
        self.label17.place(x=580, y=525)

        self.img_nan = r"common/NAN.jpg"
        self.img5i = ImageTk.PhotoImage(Image.open(self.img_nan).resize((250, 200), Image.ANTIALIAS))
        self.img5label = Label(self.frame, image=self.img5i, height=200, width=250, relief="solid")
        self.img5label.place(x=635, y=50)

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
            self.label12i = Label(self.frame, text="NaN",
                                  font=tkinter.font.Font(family="Helvetica", size=14, weight="bold"))
            self.label12ii = Label(self.frame, text="NaN",
                                   font=tkinter.font.Font(family="Helvetica", size=14, weight="bold"))
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
        self.label12i.place(x=690, y=485)
        self.label12ii.place(x=690, y=525)

        # border
        self.border = Label(self.frame, text="", bg="black", bd=1, height=745)
        self.border.place(x=510, y=0)

        self.img4 = ImageTk.PhotoImage(Image.open(r"brain_tumor/normal1.png").resize((20, 1)), Image.ANTIALIAS)
        self.border1 = Label(self.frame, image=self.img4, bg="black", bd=1, width=510, height=1)
        self.border1.place(x=0, y=210)
        self.img5 = ImageTk.PhotoImage(Image.open(r"brain_tumor/normal1.png").resize((20, 1)), Image.ANTIALIAS)
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
        self.label12i.config(text=str(round(prediction[0] * 100, 2)) + "%")
        self.label12ii.config(text=str(round(prediction[1] * 100, 2)) + "%")


    def predict(self,img):
        global prediction
        global predictionlabel
        global predictionAccuracy
        image_array2 = cv2.imread(img)
        image_array2 = cv2.cvtColor(image_array2, cv2.COLOR_BGR2RGB)
        new_array2 = cv2.resize(image_array2, (224, 224))
        new_array2 = new_array2.reshape(-1, 224, 224, 3)
        new_array2 = [new_array2 / 255]
        prediction = self.brain_tumor_model.predict(new_array2)
        prediction = prediction[0]
        no_Tumor = prediction[0]
        Tumor = prediction[1]
        if no_Tumor > Tumor:
            predictionlabel = "Normal"
            predictionAccuracy = no_Tumor * 100
        else:
            predictionlabel = "Tumor"
            predictionAccuracy = Tumor * 100
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
            return filename4
        else:
            return 0





