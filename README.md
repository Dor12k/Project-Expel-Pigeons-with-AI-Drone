
# **Project: CNN and OpenCV in Real Time**

## **Expel Pigeons with AI Drone** 

In this project, we implement algorithms for Movement Detection, Object Detection, and Object Tracking. Our system recognizes movement in video frames, detects objects, and tracks them in real-time using a camera. 
We utilize the ResNet50 model with Transfer Learning techniques on the CIFAR-10 dataset, achieving an accuracy of 95%.

### Project Overview

**Part 1 - Deep Learning:**
The first phase of this project focuses on **Deep Learning**, specifically **Convolutional Neural Networks** and related topics such as **Overfitting**, **Transfer Learning** (including **Feature Extraction** and **Fine Tuning**), and importing datasets from binary files.


**Part 2 - Computer Vision:**
The second phase involves **Computer Vision**, leveraging libraries such as **OpenCV**, **NumPy**, and **h5py**.

### Model Selection
At the beginning of the project, we experimented with various models and techniques to determine the best fit for recognizing birds. We found that **ResNet50** combined with **Transfer Learning** and **Fine Tuning** produced the highest accuracy of 95% on the **CIFAR-10** dataset.


### Application Functionality
After establishing our model, we developed the application, which operates in two states: **Detection** and **Tracking**. 
Instead of applying the ResNet50 model's predictions for every frame, we detect the object only once at the beginning and then track it throughout the video. For example, processing 2000 frames can significantly reduce computational load, resulting in running the model just once instead of 2000 times. Additionally, we initiate object detection only upon recognizing movement in the frame, optimizing performance during static conditions.

The detection process begins with recognizing movement, after which we isolate the suspicious portion of the frame and send it to the model for prediction. Once the object is detected, we commence tracking. After the tracking phase, the application resumes object detection. We log all detection events with timestamps, displaying this information on the screen, while also saving frames with detection events in a designated folder alongside a text file documenting all occurrences.

### Drone Integration
The system also features a drone mode, which automatically sends the drone to patrol whenever a bird is detected, aiming to expel it. We developed a software development kit (SDK) that allows remote control of the drone via keyboard. The SDK includes a radar display that synchronizes with the drone's location and distance during flight, along with important information such as speed, distance, camera capture capabilities, built-in tracks, detection, and tracking metrics.
User Interface

The application consists of a window divided into six screens:

  1. **Main Frame:** Displays the live camera feed.
  2. **Information Frame:** Contains detection event summaries and application details.
  3. **Tracking Frame:** Shows the object’s coordinates, confidence score, prediction time, and frames per second (FPS).
  4. **Drone Camera Frame:** Displays the drone's camera feed along with its speed, angle, and battery status.
  5. **Scores Frame:** Lists the scores for each detected label.
  6. **Drone Radar:** Synchronizes with the drone's movements and distance.

The application utilizes **TensorFlow** with **Keras** in **Python**.


Let's see a short video of our application:

https://github.com/Dor12k/Project-Expel-Pigeons-with-AI-Drone/assets/107938584/a852af70-bcc8-4bc4-b4ee-435e4684a480



Example of folder with the saved frames from detection event:
 
 ![Preview Folder](https://github.com/Dor12k/Project-Expel-Pigeons-with-AI-Drone/assets/107938584/76440b46-4d9d-4e74-8c4b-1050e967fd2b)



Example of the text file with detections events and application information:

![Preview Text](https://github.com/Dor12k/Project-Expel-Pigeons-with-AI-Drone/assets/107938584/f698a65a-992f-4e04-8d67-d83f2427922a)

