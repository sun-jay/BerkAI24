from PIL import Image
import os
import numpy as np
import io
import pyaudio
import requests
import time
import cv2

from VisionSystem import VisionSystem
from GPT4 import GPT4
import serial_comm as sc
import inverse_kinematics
from Transcriber import Transcriber



def llm_exec(prompt, image):
    # image is ndarray read into PIL
    image = Image.fromarray(image)


    global gemini, manip

    full_prompt = f"""heres how to use a robot arm API. use the pick_and_drop function. all coords are in mm, z=0 is ground level.
  #implementation in class, coords must be lists of 2 ints
  def pick_and_drop(pick_coords, drop_coords):
        go_to_coords(manip, pick_coords[0], pick_coords[1], 10, 0)
        time.sleep(1)
        go_to_coords(manip, pick_coords[0], pick_coords[1], 7, 1)
        time.sleep(1)
        go_to_coords(manip, pick_coords[0], pick_coords[1], 10, 1)
        time.sleep(1)
        go_to_coords(manip, drop_coords[0], drop_coords[1], 10, 1)
        time.sleep(1)
        go_to_coords(manip, drop_coords[0], drop_coords[1], 7, 1)
        time.sleep(1)
        go_to_coords(manip, drop_coords[0], drop_coords[1], 7, 0)
        time.sleep(1)
: You are provided with a birds-eye view image of the workspace, with the centroids of the object marked as points where you interact with the objects. assume all imports and functions are already defined in the context. using this API, write the lines of python that will complete this task: {prompt}. use pick_and_drop(params) because this code will be running in a the class. Your response should have ONLY the python code. DO NOT define functions, just write the script out. The entirety of your response will be evaluated directly by an interpreter. use breif comments to show your plan."""

    if image.mode != "RGB":
        annotated_image = image.convert('RGB')
    else:
        annotated_image = image

    img_byte_arr = io.BytesIO()
    annotated_image.save(img_byte_arr, format = "JPEG")
    img_byte_arr.seek(0)

    # response = gemini.generate_content([full_prompt, Image.open(img_byte_arr)], safety_settings={
    #     'HATE': 'BLOCK_NONE',
    #     'HARASSMENT': 'BLOCK_NONE',
    #     'SEXUAL' : 'BLOCK_NONE',
    #     'DANGEROUS' : 'BLOCK_NONE'
    # })
    # generated_code = response.text

    # return response.text --> do not have to return here, can create a python script with the robot code

    # wite exec code here

    generated_code = llm.call_gpt(full_prompt, annotated_image)
    # coord mapping
    # (0,15) - (593,621)
    # (10,20) - (1016,433)
    # (0,25) - (622,234)


    def create_affine_matrix():
        vision_coords = np.array([[593, 621], [1016, 433], [622, 234]], dtype=np.float32)
        kinematics_coords = np.array([[0, 15], [10, 20], [0, 25]], dtype=np.float32)
        return cv2.getAffineTransform(vision_coords, kinematics_coords)

    affine_matrix = create_affine_matrix()

    def vision_to_kinematics(vision_coords):
        # Check and print the shape of the affine matrix
        print("Shape of affine_matrix:", affine_matrix.shape)

        # Convert the vision coordinates to a numpy array and ensure correct dimensions
        vision_coords_array = np.array([vision_coords], dtype=np.float32).reshape(1, 1, 2)
        print("Shape of vision_coords_array:", vision_coords_array.shape)

        # Apply the affine transformation
        transformed_coords = cv2.transform(vision_coords_array, affine_matrix)
        print("Shape of transformed_coords:", transformed_coords.shape)

        # Return the first element of the transformed coordinates
        return transformed_coords[0][0] 
    
    def kinematics_to_vision(kinematics_coords):
        kinematics_coords_array = np.array([kinematics_coords], dtype=np.float32)
        inverse_matrix = np.linalg.inv(affine_matrix)
        transformed_coords = cv2.transform(kinematics_coords_array, inverse_matrix)
        return transformed_coords[0]

    def go_to_coords(manip, x, y, z, M):
        kinematics_coords = vision_to_kinematics([x, y])
        x_k, y_k = kinematics_coords[0], kinematics_coords[1]

        print("Getting servo angles...")
        print("x_k:", x_k, "y_k:" , y_k, "z:", z)
        try:
            s1, s2, s3 = inverse_kinematics.get_srvo_angles_for_coord_linear(x_k, y_k, z)
        except Exception as e:
            print("Error:", e)
            return

        print("Angles are: ", s1, s2, s3)
        time.sleep(2)  # Assuming this delay is needed for processing or hardware response

        manip.send_signal(s1, s2, s3, M)

    def pick_and_drop(pick_coords, drop_coords):
        go_to_coords(manip, pick_coords[0], pick_coords[1], 12, 0)
        time.sleep(0.1)
        go_to_coords(manip, pick_coords[0], pick_coords[1], 8, 1)
        time.sleep(0.1)
        go_to_coords(manip, pick_coords[0], pick_coords[1], 12, 1)
        time.sleep(0.1)
        go_to_coords(manip, drop_coords[0], drop_coords[1], 12, 1)
        time.sleep(0.1)
        go_to_coords(manip, drop_coords[0], drop_coords[1], 8, 1)
        time.sleep(1)
        go_to_coords(manip, drop_coords[0], drop_coords[1], 8, 0)
        time.sleep(1)

    

    def clean_string(string):
        return string.replace("```python", "").replace("```", "")
    

    generated_code = clean_string(generated_code)

    print("Generated code:")
    print(generated_code)

    try:
    
        exec(generated_code)
    finally:
        manip.send_signal(000,118,127,0)
        time.sleep(0.5)


vis = VisionSystem()

llm = GPT4()

manip = sc.Controller()
manip.connect()
manip.send_signal(000,118,127,0)
time.sleep(1)

transcriber = Transcriber(constant_print=True, verbose=True)
# transcriber.start()
print("Transcriber started")

prev = transcriber.current_command

# while True:
#     time.sleep(0.1)
#     if transcriber.current_command == prev:
#         continue

#     print("New command detected: ", transcriber.current_command)

#     prev = transcriber.current_command

#     prompt = transcriber.current_command
#     img = vis.ret_annnotated_frame()

#     # show img

#     llm_exec(prompt, img)

import pyaudio
import wave
import threading

# Parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
FILENAME = "output.wav"

audio = pyaudio.PyAudio()

def record_audio():
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []

    while not stop_recording:
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    
    with wave.open(FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

while True:
    input("Press Enter to start recording...\n")
    # Flag to stop recording
    stop_recording = False

    # Start recording in a separate thread
    recording_thread = threading.Thread(target=record_audio)
    recording_thread.start()

    # Wait for user to press Enter
    input("Press Enter to stop recording...\n")
    stop_recording = True
    recording_thread.join()

    prompt = transcriber.transcribe_file(FILENAME)

    img = vis.ret_annnotated_frame()
    print("Transcription:", prompt)
    llm_exec(prompt, img)

audio.terminate()


    
