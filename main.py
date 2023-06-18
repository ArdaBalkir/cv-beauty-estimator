import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

def compare_face_landmarks(reference_landmarks, detected_landmarks):
  # Define pairs of points representing meaningful facial features.
  pairs = [
    # Right eye (inner corner to outer corner)
    (33, 263),
    # Left eye (inner corner to outer corner)
    (362, 466),
    # Width of the nose (base)
    (4, 296),
    # Mouth (corner to corner)
    (61, 291),
    # Chin (bottom to top)
    (152, 10),
    # Top of the head to chin
    (10, 169),
    # Right cheek (middle of cheek to outer corner of eye)
    (176, 263),
    # Left cheek (middle of cheek to outer corner of eye)
    (398, 466)
  ]
  
  total_distance = 0
  for pair in pairs:
    ref_coords1 = np.array([reference_landmarks.landmark[pair[0]].x, reference_landmarks.landmark[pair[0]].y, reference_landmarks.landmark[pair[0]].z])
    ref_coords2 = np.array([reference_landmarks.landmark[pair[1]].x, reference_landmarks.landmark[pair[1]].y, reference_landmarks.landmark[pair[1]].z])
    det_coords1 = np.array([detected_landmarks.landmark[pair[0]].x, detected_landmarks.landmark[pair[0]].y, detected_landmarks.landmark[pair[0]].z])
    det_coords2 = np.array([detected_landmarks.landmark[pair[1]].x, detected_landmarks.landmark[pair[1]].y, detected_landmarks.landmark[pair[1]].z])
    
    ref_distance = np.linalg.norm(ref_coords1 - ref_coords2)
    det_distance = np.linalg.norm(det_coords1 - det_coords2)
    
    total_distance += abs(ref_distance - det_distance)
  
  return 100 / (1 + total_distance)  # Convert distance to a 'percentage' score



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

