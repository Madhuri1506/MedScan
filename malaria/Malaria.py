import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tkinter.font
from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import cv2
import tensorflow as tf

class Malaria:
    def __init__(self, frame):
        self.frame = frame
        self.model_malaria = tf.keras.models.load_model(r'malaria/malaria.h5')
        self.filename = [r"common/NAN.jpg"]
        self.dir = r"C:/"


        #Example Section
        self.width = 110
        self.height = 110
        self.img1_dir = ImageTk.PhotoImage(Image.open(r"malaria/normal1.png").resize((110, 110)), Image.ANTIALIAS)
        self.img1 = Label(self.frame, image=self.img1_dir, width=self.width, height=self.height)
        self.img2_dir = ImageTk.PhotoImage(Image.open(r"malaria/infected.png").resize((110, 110)), Image.ANTIALIAS)
        self.img2 = Label(self.frame, image=self.img2_dir, width=self.width, height=self.height)
        self.img3_dir = ImageTk.PhotoImage(Image.open(r"malaria/normal2.png").resize((110, 110)), Image.ANTIALIAS)
        self.img3 = Label(self.frame, image=self.img3_dir, width=self.width, height=self.height)

        self.label1 = Label(self.frame, text="INPUT EXAMPLE", font=tkinter.font.Font(family="Helvetica", size=15, weight="bold"))
        self.img1_label = Label(self.frame, text="Example 1", font=tkinter.font.Font(family="Helvetica", size=12, weight="normal"))
        self.img2_label = Label(self.frame, text="Example 2", font=tkinter.font.Font(family="Helvetica", size=12, weight="normal"))
        self.img3_label = Label(self.frame, text="Example 3", font=tkinter.font.Font(family="Helvetica", size=12, weight="normal"))

        self.img1_tag = Label(self.frame, text="Normal", font=tkinter.font.Font(family="Helvetica", size=9, weight="normal"))
        self.img2_tag = Label(self.frame, text="Infected", font=tkinter.font.Font(family="Helvetica", size=9, weight="normal"))
        self.img3_tag = Label(self.frame, text="Normal", font=tkinter.font.Font(family="Helvetica", size=9, weight="normal"))

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
        self.label_6i = Label(self.frame, text="About Malaria", font=tkinter.font.Font(family="Helvetica", size=15, weight="bold"))
        self.label_6i.place(x=175, y=220)
        self.label6 = Label(self.frame, wraplength=500, justify=LEFT,
                         text="""A  disease caused  by a  plasmodium  parasite, transmitted bythe bite of infected mosquitoes. The severity of malaria varies based on the species of plasmodium. Symptoms are chills, fever and sweating, usually occurring a few weeks after being bitten.People travelling to areas where malaria is  common  typically  take protective  drugs before,  during and  after their  trip. Treatment includes antimalarial drugs.The above example show the picture of blood, using this type of pic, this model is going to predict Malaria.""")
        self.label6.place(x=5, y=250)

        # symptoms Section
        self.label7i = Label(self.frame, text="Symptoms", font=tkinter.font.Font(family="Helvetica", size=15, weight="bold"))
        self.label7i.place(x=185, y=370)
        self.label7 = Label(self.frame, wraplength=500, justify=LEFT,
                         text="""Symptoms are chills, fever and sweating, usually occurring a few weeks after being bitten.""")
        self.label7.place(x=5, y=400)
        self.label8 = Label(self.frame, wraplength=500, justify=LEFT, text="""Pain areas: in the abdomen or muscles
Whole body: chills, fatigue, fever, night sweats, shivering, or sweating
Gastrointestinal: diarrhoea, nausea, or vomiting
Also common: fast heart rate, headache, mental confusion, or pallor""")
        self.label8.place(x=5, y=420)

        # treatment Section
        self.label9i = Label(self.frame, text="Treatment", font=tkinter.font.Font(family="Helvetica", size=15, weight="bold"))
        self.label9i.place(x=185, y=510)
        self.label9 = Label(self.frame, wraplength=500, justify=LEFT, text="Treatment consists of anti-parasitics")
        self.label9.place(x=5, y=540)
        self.label9i = Label(self.frame, wraplength=500, justify=LEFT,
                          text="People travelling to areas where malaria is common typically take protective drugs before, during and after their trip. Treatment includes antimalarial drugs.")

        self.label9i.place(x=5, y=560)
        self.label_9iii = Label(self.frame, wraplength=500, justify=LEFT, text="""MEDICATION
Antiparasitic: Kills parasites.
AntibioticsStops: the growth of or kills bacteria.
        """)
        self.label_9iii.place(x=5, y=600)

        # right Section
        self.label10 = Label(self.frame, text="MALARIA PREDICTION",
                          font=tkinter.font.Font(family="Helvetica", size=15, weight="bold"))
        self.label10.place(x=650, y=10)

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

        self.label16 = Label(self.frame, text="Malaria:",
                             font=tkinter.font.Font(family="Helvetica", size=14, weight="bold"))
        self.label16.place(x=580, y=485)

        self.label17 = Label(self.frame, text="Normal:",
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

        self.img4 = ImageTk.PhotoImage(Image.open(r"malaria/normal1.png").resize((20, 1)), Image.ANTIALIAS)
        self.border1 = Label(self.frame, image=self.img4, bg="black", bd=1, width=510, height=1)
        self.border1.place(x=0, y=210)
        self.img5 = ImageTk.PhotoImage(Image.open(r"malaria/normal1.png").resize((20, 1)), Image.ANTIALIAS)
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
        image_array2 = cv2.cvtColor(image_array2, cv2.COLOR_BGR2RGB)
        new_array2 = cv2.resize(image_array2, (128, 128))
        new_array2 = new_array2.reshape(-1, 128, 128, 3)
        new_array2 = [new_array2]
        prediction = self.model_malaria.predict(new_array2)
        prediction = prediction[0]
        infected = prediction[0]
        non_infected = prediction[1]
        if infected > non_infected:
            predictionlabel = "INFECTED"
            predictionAccuracy = infected * 100
        else:
            predictionlabel = "Not Infected"
            predictionAccuracy = non_infected * 100
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





