import cv2 as cv
from PIL import Image
import os
import argostranslate.package
import argostranslate.translate
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import numpy as np
import torch
import time
from gtts import gTTS
from playsound import playsound

from_code = "en"
to_code = "it"

target_size = (448,448)

dtype = torch.bfloat16
local_model_path = "./model"

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = PaliGemmaForConditionalGeneration.from_pretrained(local_model_path, torch_dtype=dtype, device_map=device, revision="bfloat16").eval()
processor = AutoProcessor.from_pretrained(local_model_path)

def crop_and_resize(image, target_size):
    width, height = image.size
    source_size = min(image.size)
    left = width // 2 - source_size // 2
    top = height // 2 - source_size // 2
    right, bottom = left + source_size, top + source_size
    return image.resize(target_size, box=(left,top,right,bottom))

def read_image(image, target_size=(448,448)):
    image = Image.open(image)
    image = crop_and_resize(image,target_size)
    image = np.array(image)

    if image.shape[2] == 4:
        image = image[:,:,:3]
    return image

def translate(text):
     argostranslate.package.update_package_index()
     available_packages = argostranslate.package.get_available_packages()
     package_to_install = next(filter(lambda x: x.from_code == from_code and x.to_code == to_code, available_packages))
     argostranslate.package.install_from_path(package_to_install.download())
     translatedText = argostranslate.translate.translate(text, from_code, to_code)
     return translatedText

def paligemma(target):
    image = read_image(target)

    text_input = "caption en"
    model_inputs = processor(text=text_input, images=image, return_tensors="pt").to(model.device)
    input_len = model_inputs["input_ids"].shape[-1]

    with torch.inference_mode(): 
            generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
            generation = generation[0][input_len:]
            decoded = processor.decode(generation, skip_special_tokens=True)
    return translate(decoded)


# Funzione per scattare la foto e salvarla
def capture_photo():
    global frame
    if frame is not None:
       
        # Scatta la foto e salvala
        cv.imwrite("photo.jpg", frame)

        # Carica la foto con PIL
        image = Image.open("photo.jpg")

        # Procedi alla funzione personalizzata dopo il ridimensionamento
        text =  paligemma("photo.jpg")
        os.remove("photo.jpg")
        return text
    

cam = cv.VideoCapture(0)
cv.resizeWindow('FUTURO REMOTO', 700, 700)

while True:
    # Read the frame
    ret, frame = cam.read()
    
    # If user type "q" on the keyboard the programm will be closed
    if cv.waitKey(1) & 0xFF == ord("q"):
        break
    font = cv.FONT_HERSHEY_SIMPLEX 
    cv.imwrite("photo.jpg", frame)
    text =  paligemma("photo.jpg")
    tts = gTTS("Vedo "+text,lang='it')
    tts.save('audio.mp3')
    
    cv.rectangle(frame, (0, 0), (1920, 40), (255, 255, 255), -1)
    cv.putText(frame,text,(10, 20),  font, 1,  (0, 0, 0),  2,  cv.LINE_4)
    cv.imshow("FUTURO REMOTO", frame)
    if os.path.exists("audio.mp3"):
         playsound('audio.mp3')
    time.sleep(5)

# Release all resources after programm ending
cam.release()
# Rilascia la webcam quando l'app si chiude
cv.destroyAllWindows()
