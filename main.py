import cv2
import mediapipe as mp
import pygame
import sys
import numpy as np

# Initialize Pygame
pygame.init()

# Constants for screen dimensions
screen_width, screen_height = 640, 480

# Create a Pygame window
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Hand Tracking Paint")

# Create a separate surface for the drawings
drawing_surface = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)

# Colors
white = (255, 255, 255)

# Initialize MediaPipe Hand
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

# Set the frame rate to 60 FPS
cap.set(cv2.CAP_PROP_FPS, 60)

# Set the frame width and height
cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)

# Variable to store the previous point
prev_point = None

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Flip the frame vertically for better visualization
    frame = cv2.flip(frame, 1)

    # Convert the frame color to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    # If a hand is detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the coordinates of the index finger tip and middle finger tip
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            # Calculate the distance between the index finger tip and the middle finger tip
            distance = np.sqrt((index_finger_tip.x - middle_finger_tip.x) ** 2 + (index_finger_tip.y - middle_finger_tip.y) ** 2)

            # If the distance is small, start drawing
            if distance < 0.05:  # Adjust this threshold as needed
                x = int(index_finger_tip.x * screen_width)
                y = int(index_finger_tip.y * screen_height)

                # If this is the first point, just draw a circle
                if prev_point is None:
                    pygame.draw.circle(drawing_surface, white, (x, y), 5)
                else:
                    # Otherwise, draw a line from the previous point to the current point
                    pygame.draw.line(drawing_surface, white, prev_point, (x, y), 5)

                # Update the previous point
                prev_point = (x, y)
            else:
                # If the distance is large, stop drawing
                prev_point = None

    # Draw the drawing surface onto the Pygame window
    screen.blit(drawing_surface, (0, 0))

    # Handle Pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            cap.release()
            cv2.destroyAllWindows()
            sys.exit()
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_x:
            pygame.quit()
            cap.release()
            cv2.destroyAllWindows()
            sys.exit()

    # Update the Pygame display
    pygame.display.flip()

    # Add a delay to control the frame rate (optional)
    pygame.time.delay(20)
