from ultralytics import YOLO
import time
import streamlit as st
import cv2
import os
import format_file
import Self_Model_DhikaN

def load_model(path_model):
    """
    Loads a YOLO object detection model from the specified path_model.

    Parameters:
        path_model (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(path_model)
    return model


def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None


def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):
    """
    Display the detected objects on a video frame using the YOLOv8 model.

    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YoloV8): A YOLOv8 object detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.
    - is_display_tracking (bool): A flag indicating whether to display object tracking (default=None).

    Returns:
    None
    """

    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720*(9/16))))

    # Display object tracking, if specified
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf)

    # # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )


# in action_file.py
def play_stored_videos(conf, model, upload_video):
    """
    Plays a stored video file. Tracks and detects objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.
        upload_video: The uploaded video file.

    Returns:
        None

    Raises:
        None
    """
    source_vid = st.sidebar.selectbox("Choose a video...", ["upload_video"])

    is_display_tracker, tracker = display_tracker_options()

    if upload_video is not None:
        video_bytes = upload_video.getvalue()
        video_path = f"temp_video_{int(time.time())}.mp4"
        with open(video_path, "wb") as video_file:
            video_file.write(video_bytes)

        try:
            vid_cap = cv2.VideoCapture(video_path)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    image = cv2.resize(image, (720, int(720*(9/16))))
                    res = model.predict(image, conf=conf)  # Use the model for prediction
                    st_frame.image(res[0].plot(),
                                   caption='Detected Video',
                                   channels="BGR",
                                   use_column_width=True
                                   )

                    try:
                        with st.expander("Detection Results"):
                            for box in res[0].boxes:
                                st.write(box.data)
                    except Exception as ex:
                        st.write("No image is uploaded yet!")
                else:
                    vid_cap.release()
                    break
            os.remove(video_path)  # Remove the temporary video file
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))
    else:
        st.sidebar.error("No video uploaded")