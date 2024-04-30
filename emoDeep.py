
import cv2
import numpy as np
import pygetwindow as gw
import pyscreenrec
import pyautogui
import tensorflow as tf
from kivy.uix.image import Image
from kivymd.app import MDApp
from kivymd.uix.screen import Screen
from kivymd.uix.boxlayout import BoxLayout
from kivymd.uix.label import MDLabel
from kivymd.uix.button import MDRectangleFlatButton, MDFillRoundFlatButton
from kivy.graphics.texture import Texture
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.graphics import Color, Rectangle
from kivy.uix.button import Button
from kivy.clock import Clock
from tensorflow.keras.models import load_model 
from PIL import ImageGrab

import shutil
import time
from datetime import datetime
import os





class Demo(MDApp):
    # Le reste du code de la classe...

    def build(self):
        self.model = load_model('modele_emotions.h5')
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.capture = cv2.VideoCapture(0)
        self.is_running = False
        self.out = None  # Initialiser l'attribut out
        self.video_popup = None
        self.video_folder = 'video_test'
        self.recording_writer = None
        self.is_recording = False
        self.output_filename = None
        self.content = Label(text='La Video capture terminée', font_size=16)
        self.recorder = pyscreenrec.ScreenRecorder() 
        self.emotion_labels = {0: "Colere", 1: "Mepris", 2: "Degout", 3: "Peur", 4: "Heureux", 5: "Triste", 6: "Surprise"}
        self.emotion_report_labels = {}
        self.is_hidden = False
        screen = Screen()

        
        top_box = BoxLayout(orientation='vertical')
        top_box.pos_hint = {"center_x": 0.5, "center_y": 0.9}
        self.image = Image(source='facial-recognition-img2.jpg', allow_stretch=False)
        self.image.size_hint = (1, 1)  # Maintain aspect ratio
        top_box.add_widget(self.image)

        self.second_box = BoxLayout(orientation='horizontal')
        self.second_box.pos_hint = {"center_x": 0.5, "center_y": 0.6}
        self.second_box.size_hint_y = None
        self.second_box.height = 100
        
        left_box = BoxLayout(orientation='horizontal')
        left_box.size_hint_y = None
        left_box.height = 50
        right_box = BoxLayout(orientation='horizontal')
        right_box.size_hint_y = None
        right_box.height = 50

        # Initialisation des étiquettes de rapport pour chaque émotion

        # Ajout des boxlayouts à second_box
    



        third_box = BoxLayout(orientation='horizontal')
        third_box.pos_hint = {"center_x": 0.5, "center_y": 0.5}

        instructions_label = MDLabel(text="Bouton pour la cotrôle caméra", halign="center")
        third_box.add_widget(instructions_label)

 

        screen.add_widget(top_box)
        screen.add_widget(self.second_box)
        screen.add_widget(third_box)

        btn1 = MDFillRoundFlatButton(text="Lancer Camera", theme_text_color="Custom",  md_bg_color=(0, 0.9, 0.5, 1))  # Green background
        btn1.pos_hint = {"center_x": 0.2, "center_y": 0.4}
        btn1.bind(on_press = self.start_detection)  # Assign sspecific function
        self.button_runing_target = btn1
        

        btn2 = MDFillRoundFlatButton(text="Arreter Camera", icon="stop",theme_text_color="Custom",  md_bg_color=(0, 0.9, 0.5, 1))
        btn2.pos_hint = {"center_x": 0.8, "center_y": 0.4}
        btn2.bind(on_press = self.stop_camera  ) # Assign specific function
        #btn2.on_release = lambda instance: self.stop_camera(instance)


        btn3 = MDFillRoundFlatButton(text="Capture Vidéo" , theme_text_color="Custom", md_bg_color = (0, 0.5, 0.9, 1))
        btn3.pos_hint = {"center_x": 0.2, "center_y": 0.3}
        btn3.bind(on_press =  self.start_capture)  # Assign specific function
        self.target_button = btn3

        btn4 = MDFillRoundFlatButton(text=" Arréter Capture", theme_text_color="Custom", md_bg_color = (0, 0.5, 0.9, 1))
        btn4.pos_hint = {"center_x": 0.8, "center_y": 0.3}
        btn4.bind(on_press = self.stop_capture) # Assign specific function

        btn5 = MDFillRoundFlatButton(text="Caché Visage" , theme_text_color="Custom",  md_bg_color=(0.9, 0.8, 0, 1))
        btn5.pos_hint = {"center_x": 0.2, "center_y": 0.2}
        btn5.bind(on_press = self.hide_faces  ) # Assign specific function

        btn6 = MDFillRoundFlatButton(text="Face Only" , theme_text_color="Custom", md_bg_color=(0.9, 0.8, 0, 1))
        btn6.pos_hint = {"center_x": 0.8, "center_y": 0.2}
        btn6.bind(on_press =  self.show_faces  )# Assign specific function

        btn7 = MDFillRoundFlatButton(text="Capture Image" ,theme_text_color="Custom", md_bg_color=(0, 0, 0, 1))
        btn7.pos_hint = {"center_x": 0.5, "center_y": 0.1}
        btn7.style = "tonal"
        btn7.on_release = self.capture_image  # Assign specific function


        # Add more buttons with similar customizations

        screen.add_widget(btn1)
        screen.add_widget(btn2)
        screen.add_widget(btn3)
        screen.add_widget(btn4)
        screen.add_widget(btn5)
        screen.add_widget(btn6)
        screen.add_widget(btn7)
        
        
        return screen
    
    def start_detection(self, instance):
        self.is_running = True
        self.capture = cv2.VideoCapture(0)  # Réinitialiser la capture de la caméra
        Clock.schedule_interval(self.start_camera, 1.0 / 30.0)  # 30 FPS
        instance.disabled = True
        instance.md_bg_color=(0.5, 0.5, 0.5, 1)


    def start_camera(self, dt):
        if not self.is_running:
            return

        ret, frame = self.capture.read()
        self.rets = ret
        self.frames = frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        emotion_counts_per_face = {}
        predicted_class_index = -1 
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = np.expand_dims(face, axis=-1)
            face = np.expand_dims(face, axis=0)

            predictions = self.model.predict(face)
            predicted_class_index = np.argmax(predictions)
            predicted_emotion = self.emotion_labels[predicted_class_index]
            #predicted_emotion = self.emotion_labels_french.get(self.emotion_labels[predicted_class_index], "Émotion inconnue")

            if predicted_class_index in emotion_counts_per_face:
                emotion_counts_per_face[predicted_class_index] += 1
            else:
                emotion_counts_per_face[predicted_class_index] = 1
            predicted_emotion = self.emotion_labels[predicted_class_index]
            if  self.is_hidden:
                cv2.putText(frame, predicted_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), -1)
            cv2.putText(frame, predicted_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Flips are no longer necessary as emotion_labels_french is used for display
        frame = cv2.flip(frame, 0)
        # frame = cv2.flip(frame, 1)
        for emotion_label, count in emotion_counts_per_face.items():
            emotion_report_text = f"{self.emotion_labels[emotion_label]}: {count}"
            if emotion_label not in self.emotion_report_labels:
                # Créez une nouvelle étiquette pour cette émotion
                    new_label = Label(text=emotion_report_text, color=(0, 0, 0, 1))  # Couleur noire
                    self.second_box.add_widget(new_label)
                    self.emotion_report_labels[emotion_label] = new_label
            else:
                # Mettez à jour le texte de l'étiquette existante
                self.emotion_report_labels[emotion_label].text = emotion_report_text

        # Supprimez les étiquettes des émotions qui ne sont plus détectées
        emotions_to_remove = [emotion_label for emotion_label in self.emotion_report_labels.keys() if emotion_label not in emotion_counts_per_face]
        for emotion_label in emotions_to_remove:
            label = self.emotion_report_labels.pop(emotion_label)
            self.second_box.remove_widget(label)
            
        buf1 = frame.tostring()
        texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture1.blit_buffer(buf1, colorfmt='bgr', bufferfmt='ubyte')
        self.image.texture = texture1
        
    '''def generate_emotion_report(self):
        report = "Rapport des émotions:\n"
        for emotion_label, count in self.emotion_counts.items():
            report += f"{self.emotion_labels[emotion_label]}: {count}\n"
        return report  ''' 

    def stop_camera(self, obj):
        """
        Stops the camera, disables emotion detection, and resets the image.
        """
        # Release the camera resource
        self.is_running = False
        self.capture.release()
    
        # Reset the image to default texture
        buf = cv2.imread('facial-recognition-img2.jpg', cv2.IMREAD_COLOR)
        buf = cv2.flip(buf, 0)
        buf = cv2.flip(buf, 1)
        
        # Convert the image to numpy array
        buf = np.array(buf)
        
        # Create texture from numpy array
        texture = Texture.create(size=(buf.shape[1], buf.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf.tostring(), colorfmt='bgr', bufferfmt='ubyte')
        self.image.texture = texture

        # Stop scheduling the frame update loop
        Clock.unschedule(self.start_camera)
        if self.button_runing_target:
            self.button_runing_target.disabled = False
            self.button_runing_target.md_bg_color = (0, 0.9, 0.5, 1)
            
        
    def start_capture(self, instance):
        if self.is_running:   
            self.is_recording = True   
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.recorder.start_recording(f"{current_time}.mp4", 10)
            
            
            self.popup_capture(f"Capture vidéo en cours...\nFichier : {current_time}")
             # Changer la couleur du bouton start_capture pour montrer qu'il est enfoncé
            instance.md_bg_color = (0.5, 0.5, 0.5, 1)  # Couleur foncée
            instance.disabled = True  # Désactiver le bouton
        else:
             self.popup_error_capture("Vous devez d'abord lancer la caméra")

        
    def popup_capture(self, message):
        content = Label(text=message)
        popup = Popup(title="Capture en cours", content=content, size_hint=(None, None), size=(400, 200))
        popup.open()
        
    def popup_error_capture(self, message):
        content = Label(text=message)
        popup = Popup(title="ERREUR", content=content, size_hint=(None, None), size=(400, 200))
        popup.open()
           
            
    def stop_capture(self, instance):
        if self.is_recording:
            
            self.recorder.stop_recording()
            self.popup_stop("Capture vidéo terminée...\n")
            self.is_recording = False
            if self.target_button:
                self.target_button.disabled = False
                self.target_button.md_bg_color = (0, 0.5, 0.9, 1)
        else:
            self.popup_stop_error("La capture vidéo n'a pas été encore lancée")
            
    def popup_stop_error(self, message):
        content = Label(text=message)
        popup = Popup(title="ERREUR", content=content, size_hint=(None, None), size=(400, 200))
        popup.open()
        
    def popup_stop(self, message):
        content = Label(text=message)
        popup = Popup(title="Fin capture", content=content, size_hint=(None, None), size=(400, 200))
        popup.open()
        


    def capture_image(self):
        if self.is_running:
            my_screenshot = ImageGrab.grab()
            file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            my_screenshot.save(f"image_{file_name}.png")
            self.image_saved_popup(f"Image : {file_name}.png enregistrée")
            my_screenshot.close()
        else:
            self.image_error_popup("Vous devez d'abord lancer la caméra")
            
        print("Image Captured!")
        
    def image_saved_popup(self, message):
        content = Label(text=message)
        popup = Popup(title="PARFAIT", content=content, size_hint=(None, None), size=(400, 200))
        popup.open()
        
    def image_error_popup(self, message):
        content = Label(text=message)
        popup = Popup(title="ERREUR", content=content, size_hint=(None, None), size=(400, 200))
        popup.open()
        
    def show_faces(self, instance):
        self.is_hidden = False
        
    
    def hide_faces(self, instance):
        self.is_hidden = True
        
    
            
if __name__ == "__main__":
    Demo().run()
