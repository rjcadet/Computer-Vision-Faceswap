# Computer-Vision-Faceswap

This project is a simple face swapping backend built with OpenCV, dlib, and FastAPI
It takes a face from image 1 and brings it on to body from image 2 and returns the resulting images.

1. Download the Expo Go app
2. Install python dependencies in Terminal:
   pip install fastapi uvicorn opencv-python dlib numpy
3. You also have to download the dlib facial landmark model and place it in the project root:
   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   unzip it to get "shape_predictor_68_face_landmarks.dat"
4. Open App.js and replace the "!!!!!REPLACE IP HERE!!!!!"

Once you're ready to start, run the backend in terminal:
1. navigate to faceswap_backend: cd ~/(wherever you downloaded it to)/faceswap_backend
2. run python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Next

