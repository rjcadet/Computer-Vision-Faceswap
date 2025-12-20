# Computer-Vision-Faceswap

This project is a simple face swapping backend built with OpenCV, dlib, and FastAPI
It takes a face from image 1 and brings it on to body from image 2 and returns the resulting images.
Python 3.9+

1. Download the Expo Go app
Note: When using the Expo go app, your phone and computer MUST be on the same wifi for the backend to work.
3. Install python dependencies in Terminal:
   pip install fastapi uvicorn opencv-python dlib numpy
4. You also have to download the dlib facial landmark model and place it in the project root:
   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   unzip it to get "shape_predictor_68_face_landmarks.dat"
5. Open App.js and replace the "!!!!!REPLACE IP HERE!!!!!" on line 72 to be your local ip (you can get this by running: "ipconfig getifaddr en0" in terminal (this won't change when you move around, it pertains to the wifi on your computer.)

Once you're ready to start, run the backend in terminal:
1. navigate to "faceswap_backend": cd ~/(wherever you downloaded it to)/faceswap_backend
2. run python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
3. don't close this once you're done you will need it running while you use the app

Next to run the frontend:
1. open a new terminal tab and navigate to the "faceswap" directory: cd ~/(wherever you downloaded it to)/faceswap
2. and run "npx expo start"
3. Scan the QR code with your phone and enjoy swapping faces!

Note that for the app to work properly you need to have a jpeg image with 1 face per image
The most convenient way to do this is to take the two photos that you planned to faceswap and take screenshots of them. Once you've done this, cropping both of the screenshots to be square gives the best resulting image.
