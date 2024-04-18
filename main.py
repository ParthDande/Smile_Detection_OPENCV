import cv2 as cv

# Load the smile cascade classifier
#smile_cascade = cv.CascadeClassifier(r'C:\Users\Parth\Desktop\DataScience\OPENCV\Smile_Detection_OPENcv\haar_smile.xml')
smile_cascade=cv.CascadeClassifier(cv.data.haarcascades +'haarcascade_smile.xml')
face_cascade =cv.CascadeClassifier(cv.data.haarcascades+'haarcascade_frontalface_default.xml')
# Start capturing video from webcam
capture = cv.VideoCapture(0)

while True:
    isTrue, frame = capture.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # Convert the frame to grayscale
    # Detect faces in the grayscale frame
    face_coordinates = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2)
    # Iterate over each detected face
    for (x, y, w, h) in face_coordinates:
        # Draw a rectangle around the face for reference
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

        # Extract the region of interest (ROI) for smile detection
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detect smiles in the ROI
        smile = smile_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.7,
            minNeighbors=22,
            minSize=(25, 25),
            flags=cv.CASCADE_SCALE_IMAGE
        )

        # Draw a line on the face to indicate a smile
        for (sx, sy, sw, sh) in smile:
            cv.line(frame, (x + sx, y + sy + int(sh / 2)), (x + sx + sw, y + sy + int(sh / 2)), (255, 0, 0), 2)

    # Display the frame with smile detection
    cv.imshow('Smile Detection', frame)

    # Check for quit command
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

# Release the capture and destroy the window
capture.release()
cv.destroyAllWindows()
