import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

def compare_face_landmarks(reference_landmarks, detected_landmarks):
  total_distance = 0
  for ref_landmark, det_landmark in zip(reference_landmarks.landmark, detected_landmarks.landmark):
    ref_coords = np.array([ref_landmark.x, ref_landmark.y, ref_landmark.z])
    det_coords = np.array([det_landmark.x, det_landmark.y, det_landmark.z])
    distance = np.linalg.norm(ref_coords - det_coords)
    total_distance += distance / len(reference_landmarks.landmark)  # Scale distances
  return 100 / (1 + total_distance * 10)  # Adjust the conversion to percentage


# Load the reference image of Bella Hadid and get its face mesh.
reference_image = cv2.imread('reference_image.jpg')
reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)
with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5) as face_mesh:
  results = face_mesh.process(reference_image)
  if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
      reference_face_landmarks = face_landmarks

# Start webcam feed.
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        # Compare the face landmarks with the reference face landmarks.
        match_percentage = compare_face_landmarks(reference_face_landmarks, face_landmarks)
        cv2.putText(image, f'Beauty Percentage: {match_percentage:.2f}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)

    cv2.imshow('MediaPipe FaceMesh', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()

