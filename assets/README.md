Place your README screenshots here:

- demo_output.jpg  -> a frame from your annotated output_speed.mp4
  Grab one in Python:
    import cv2
    cap = cv2.VideoCapture("output_speed.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 200)
    ok, frame = cap.read()
    cv2.imwrite("assets/demo_output.jpg", frame)
