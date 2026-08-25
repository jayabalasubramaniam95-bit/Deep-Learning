import cv2
import numpy as np
import tensorflow as tf


# ============================================================
# COMPATIBILITY INITIALIZER
# ============================================================

# GlorotUniform - neural-network weight initializer.
class CompatibleGlorotUniform(
    tf.keras.initializers.GlorotUniform
):

    def __init__(
        self,
        seed=None,
        input_axes=None,
        output_axes=None,
        **kwargs
    ):
        # input_axes and output_axes are ignored
        # because the installed Keras version
        # does not support them.

        super().__init__(
            seed=seed,
            **kwargs
        )


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = r"D:\AI\8. Deep Learning\Gender Prediction\mobilenetv3_model.keras"

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "GlorotUniform": CompatibleGlorotUniform,
        "keras.initializers.GlorotUniform": CompatibleGlorotUniform
    },
    compile=False
)

print("Model loaded successfully!")

model.summary()


# ============================================================
# IMAGE SIZE
# ============================================================

IMG_SIZE = (224, 224)


# ============================================================
# LOAD FACE DETECTOR
# ============================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    raise RuntimeError(
        "Could not load Haar Cascade face detector"
    )


# ============================================================
# WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError(
        "Cannot open webcam"
    )


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    # --------------------------------------------------------
    # READ FRAME
    # --------------------------------------------------------

    ret, frame = cap.read()

    if not ret:
        print("Cannot read webcam")
        break


    # --------------------------------------------------------
    # FACE DETECTION
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )


    # ========================================================
    # PROCESS EACH FACE
    # ========================================================

    for (x, y, w, h) in faces:

        # ----------------------------------------------------
        # CROP FACE
        # ----------------------------------------------------

        face = frame[
            y:y + h,
            x:x + w
        ]

        # Make sure face crop is valid
        if face.size == 0:
            continue


        # ----------------------------------------------------
        # CONVERT BGR → RGB
        # ----------------------------------------------------

        face_rgb = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2RGB
        )


        # ----------------------------------------------------
        # RESIZE TO MODEL INPUT SIZE
        # ----------------------------------------------------

        face_rgb = cv2.resize(
            face_rgb,
            IMG_SIZE
        )


        # ----------------------------------------------------
        # CONVERT TO FLOAT32
        # ----------------------------------------------------

        face_rgb = face_rgb.astype(
            np.float32
        )


        # ----------------------------------------------------
        # ADD BATCH DIMENSION
        # ----------------------------------------------------

        face_input = np.expand_dims(
            face_rgb,
            axis=0
        )


        # ====================================================
        # GENDER PREDICTION
        # ====================================================

        prediction = model.predict(
            face_input,
            verbose=0
        )


        probability = float(
            prediction[0][0]
        )


        # ====================================================
        # DETERMINE GENDER
        # ====================================================

        if probability < 0.5:

            # -----------------------------------------------
            # MALE
            # -----------------------------------------------

            gender = "Male"
            
            confidence = (
                (1 - probability) * 100
            )

           

            # RED rectangle
            # OpenCV uses BGR
            box_color = (
                0,
                0,
                255
            )

        else:

            # -----------------------------------------------
            # FEMALE
            # -----------------------------------------------

            gender = "Female"

            confidence = (
                probability * 100
            )

            # GREEN rectangle
            # OpenCV uses BGR
            box_color = (
                0,
                255,
                0
            )


        # ====================================================
        # DRAW FACE RECTANGLE
        # ====================================================

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            box_color,
            3
        )


        # ====================================================
        # CREATE LABEL
        # ====================================================

        text = (
            f"{gender}: "
            f"{confidence:.2f}%"
        )


        # ====================================================
        # GET TEXT SIZE
        # ====================================================

        (
            text_width,
            text_height
        ), baseline = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            2
        )


        # ====================================================
        # LABEL POSITION
        # ====================================================

        label_x1 = x

        label_y1 = max(
            0,
            y - text_height - baseline - 10
        )

        label_x2 = (
            x +
            text_width +
            10
        )

        label_y2 = y


        # ====================================================
        # DRAW LABEL BACKGROUND
        # ====================================================

        cv2.rectangle(
            frame,
            (label_x1, label_y1),
            (label_x2, label_y2),
            box_color,
            -1
        )


        # ====================================================
        # DRAW GENDER + CONFIDENCE TEXT
        # ====================================================

        cv2.putText(
            frame,
            text,
            (x + 5, y - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


    # ========================================================
    # SHOW WEBCAM
    # ========================================================

    cv2.imshow(
        "Gender Prediction - MobileNetV3",
        frame
    )


    # ========================================================
    # PRESS Q OR q TO QUIT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == ord("Q"):
        break


# ============================================================
# RELEASE WEBCAM
# ============================================================

cap.release()


# ============================================================
# CLOSE ALL OPENCV WINDOWS
# ============================================================

cv2.destroyAllWindows()

print("Program closed.")