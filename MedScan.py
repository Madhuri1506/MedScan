from tkinter import *
from tkinter import ttk
from Main import Main
from malaria.Malaria import Malaria
from Covid_Pnemonia.Covid_pnemonia import CovidPnemonia
from brain_tumor.Brain_Tumor import BrainTumor
from Cancer.skin_cancer import SkinCancer
from ChestXrayAbnormality.Chest_Xray_Abnormality import Chest_Xray_Abnormality
from BrainTumorClassification.Brain_Tumor_Classification import BrainTumorClassification
from EyeDibaties.Eye_Diabetic_retonopathy import EyeDibeticClassification
from BactarialVSCovid.viral_bacterial import ViralVSBacterial

root = Tk()
root.geometry("1024x750")
root.title("MedScan")
root.iconbitmap(r"common/icon.ico")
notebook = ttk.Notebook(root)
root.maxsize(1024, 750)
root.minsize(1024, 750)

# Main
Main_frame = Frame(notebook, width=1020, height=745)
Main(Main_frame)
Main_frame.grid(row=0, column=0)
notebook.add(Main_frame, text="  Main  ")
notebook.grid(row=0, column=0)

# Malaria part
Malaria_frame = Frame(notebook, width=1020, height=745)
Malaria(Malaria_frame)
Malaria_frame.grid(row=1, column=0)
notebook.add(Malaria_frame, text="  Malaria  ")
notebook.grid(row=0, column=1)

# Covid / Pnemonia
Covid_Pnemonia_frame = Frame(notebook, width=1020, height=745)
CovidPnemonia(Covid_Pnemonia_frame)
Covid_Pnemonia_frame.grid(row=1, column=0)
notebook.add(Covid_Pnemonia_frame, text="  Covid-19/Pneumonia  ")
notebook.grid(row=0, column=2)

# Brain Tumor
Brain_Tumor_frame = Frame(notebook, width=1020, height=745)
BrainTumor(Brain_Tumor_frame)
Brain_Tumor_frame.grid(row=1, column=0)
notebook.add(Brain_Tumor_frame, text="  Brain Tumor  ")
notebook.grid(row=0, column=3)

# Cancer
Skin_Cancer_frame = Frame(notebook, width=1020, height=745)
SkinCancer(Skin_Cancer_frame)
Skin_Cancer_frame.grid(row=1, column=0)
notebook.add(Skin_Cancer_frame, text="  Malignant/Benign  ")
notebook.grid(row=0, column=4)

# Abnormal Chest X-ray
Chest_Xray_abnormality_frame = Frame(notebook, width=1020, height=745)
Chest_Xray_Abnormality(Chest_Xray_abnormality_frame)
Chest_Xray_abnormality_frame.grid(row=1, column=0)
notebook.add(Chest_Xray_abnormality_frame, text="  Abnormal Chest X-ray  ")
notebook.grid(row=0, column=5)

# Brain Tumor Classification
Brain_tumor_Classification_frame = Frame(notebook, width=1020, height=745)
BrainTumorClassification(Brain_tumor_Classification_frame)
Brain_tumor_Classification_frame.grid(row=1, column=0)
notebook.add(Brain_tumor_Classification_frame, text="  Brain Tumor Classification  ")
notebook.grid(row=0, column=6)

# Eye Diabetic
Eye_Dibetic_frame = Frame(notebook, width=1020, height=745)
EyeDibeticClassification(Eye_Dibetic_frame )
Eye_Dibetic_frame .grid(row=1, column=0)
notebook.add(Eye_Dibetic_frame , text="  Diabetic Retinopathy  ")
notebook.grid(row=0, column=7)

# Bacterial vs Virus
Bacterial_viral_frame = Frame(notebook, width=1020, height=745)
ViralVSBacterial(Bacterial_viral_frame)
Bacterial_viral_frame.grid(row=1, column=0)
notebook.add(Bacterial_viral_frame, text="   Pnemonia Bacterial/Viral   ")
notebook.grid(row=0, column=8)

root.mainloop()
