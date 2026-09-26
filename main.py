
import cv2
import os
import time
import math
import csv
from collections import defaultdict, deque
from ultralytics import YOLO


# ============================================================
# ADVANCED AI AIRCRAFT DETECTION & TRACKING
# YOLO + BYTE TRACK + OPENCV
# ============================================================


# ------------------------------------------------------------
# 1. PROJECT SETTINGS
# ------------------------------------------------------------

INPUT_VIDEO = "input/aircraft.mp4"

OUTPUT_VIDEO = "output/aircraft_ai_tracking.mp4"

OUTPUT_CSV = "output/aircraft_telemetry.csv"

# YOLO model
MODEL_PATH = "yolo11m.pt"

# Detection confidence
CONFIDENCE = 0.25

# IoU threshold
IOU = 0.50

# ByteTrack tracker
TRACKER = "bytetrack.yaml"

# ------------------------------------------------------------
# VIDEO REGION
# ------------------------------------------------------------

# Your video contains some Unreal Editor/interface area.
# These values focus detection on the gameplay area.

ROI_TOP_RATIO = 0.12

ROI_BOTTOM_RATIO = 0.95

# ------------------------------------------------------------
# TRAJECTORY
# ------------------------------------------------------------

TRAIL_LENGTH = 40

# Show center crosshair
DRAW_RETICLE = True

# Show OpenCV window
SHOW_WINDOW = True


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs("output", exist_ok=True)


# ============================================================
# AIRCRAFT CLASS DETECTION
# ============================================================

def get_aircraft_class_ids(model):

    names = model.names

    if isinstance(names, list):
        names = {
            i: name
            for i, name in enumerate(names)
        }

    keywords = (
        "airplane",
        "aircraft",
        "plane",
        "jet",
        "fighter",
        "helicopter",
        "aeroplane"
    )

    aircraft_ids = []

    for class_id, name in names.items():

        name = str(name).lower()

        for keyword in keywords:

            if keyword in name:

                aircraft_ids.append(
                    int(class_id)
                )

                break

    # COCO airplane class
    if not aircraft_ids:

        if 4 in names:

            aircraft_ids = [4]

    return aircraft_ids


# ============================================================
# HUD CORNER BOX
# ============================================================

def draw_corner_box(
    image,
    x,
    y,
    width,
    height,
    color=(0, 255, 0),
    thickness=2
):

    length = min(
        25,
        width // 4,
        height // 4
    )

    # Top left
    cv2.line(
        image,
        (x, y),
        (x + length, y),
        color,
        thickness
    )

    cv2.line(
        image,
        (x, y),
        (x, y + length),
        color,
        thickness
    )

    # Top right
    cv2.line(
        image,
        (x + width, y),
        (x + width - length, y),
        color,
        thickness
    )

    cv2.line(
        image,
        (x + width, y),
        (x + width, y + length),
        color,
        thickness
    )

    # Bottom left
    cv2.line(
        image,
        (x, y + height),
        (x + length, y + height),
        color,
        thickness
    )

    cv2.line(
        image,
        (x, y + height),
        (x, y + height - length),
        color,
        thickness
    )

    # Bottom right
    cv2.line(
        image,
        (x + width, y + height),
        (x + width - length, y + height),
        color,
        thickness
    )

    cv2.line(
        image,
        (x + width, y + height),
        (x + width, y + height - length),
        color,
        thickness
    )


# ============================================================
# RETICLE
# ============================================================

def draw_reticle(frame):

    height, width = frame.shape[:2]

    center_x = width // 2
    center_y = height // 2

    size = 32
    gap = 9

    # Horizontal
    cv2.line(
        frame,
        (
            center_x - size,
            center_y
        ),
        (
            center_x - gap,
            center_y
        ),
        (0, 255, 255),
        1
    )

    cv2.line(
        frame,
        (
            center_x + gap,
            center_y
        ),
        (
            center_x + size,
            center_y
        ),
        (0, 255, 255),
        1
    )

    # Vertical
    cv2.line(
        frame,
        (
            center_x,
            center_y - size
        ),
        (
            center_x,
            center_y - gap
        ),
        (0, 255, 255),
        1
    )

    cv2.line(
        frame,
        (
            center_x,
            center_y + gap
        ),
        (
            center_x,
            center_y + size
        ),
        (0, 255, 255),
        1
    )

    cv2.circle(
        frame,
        (
            center_x,
            center_y
        ),
        3,
        (0, 255, 255),
        1
    )


# ============================================================
# TEXT FUNCTION
# ============================================================

def put_text(
    frame,
    text,
    position,
    scale=0.55,
    color=(255, 255, 255),
    thickness=1
):

    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    # --------------------------------------------------------
    # CHECK VIDEO
    # --------------------------------------------------------

    if not os.path.exists(INPUT_VIDEO):

        print()
        print("ERROR!")
        print("Video not found:")
        print(INPUT_VIDEO)
        print()
        print("Put your video inside:")
        print("input/aircraft.mp4")
        return


    # --------------------------------------------------------
    # LOAD YOLO
    # --------------------------------------------------------

    print()
    print("====================================")
    print("Loading YOLO model...")
    print("====================================")

    model = YOLO(
        MODEL_PATH
    )

    print("YOLO model loaded.")


    # --------------------------------------------------------
    # FIND AIRCRAFT CLASSES
    # --------------------------------------------------------

    aircraft_ids = get_aircraft_class_ids(
        model
    )

    print(
        "Aircraft class IDs:",
        aircraft_ids
    )


    if not aircraft_ids:

        print(
            "WARNING: Aircraft class was not found."
        )

        aircraft_ids = None


    # --------------------------------------------------------
    # OPEN VIDEO
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        INPUT_VIDEO
    )


    if not cap.isOpened():

        print(
            "ERROR: Could not open video."
        )

        return


    # --------------------------------------------------------
    # VIDEO INFORMATION
    # --------------------------------------------------------

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    source_fps = cap.get(
        cv2.CAP_PROP_FPS
    )


    if not source_fps or source_fps <= 0:

        source_fps = 30


    print()
    print("Video information")
    print("-----------------------------")
    print("Width :", width)
    print("Height:", height)
    print("FPS   :", source_fps)
    print("-----------------------------")


    # --------------------------------------------------------
    # VIDEO WRITER
    # --------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        fourcc,
        source_fps,
        (width, height)
    )


    # ========================================================
    # TRACK HISTORY
    # ========================================================

    trails = defaultdict(
        lambda: deque(
            maxlen=TRAIL_LENGTH
        )
    )


    # ========================================================
    # PREVIOUS POSITION
    # ========================================================

    previous_positions = {}


    # ========================================================
    # VARIABLES
    # ========================================================

    frame_number = 0

    start_time = time.time()


    # ========================================================
    # CSV FILE
    # ========================================================

    csv_file = open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    )

    csv_writer = csv.writer(
        csv_file
    )


    csv_writer.writerow([
        "frame",
        "time_seconds",
        "track_id",
        "class",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "center_x",
        "center_y",
        "pixel_speed",
        "distance_from_center"
    ])


    print()
    print("====================================")
    print("Starting aircraft detection...")
    print("Press Q to stop.")
    print("====================================")


    # ========================================================
    # VIDEO LOOP
    # ========================================================

    while True:

        ret, frame = cap.read()


        if not ret:

            break


        frame_number += 1


        current_video_time = (
            frame_number /
            source_fps
        )


        # ====================================================
        # REGION OF INTEREST
        # ====================================================

        roi_top = int(
            height *
            ROI_TOP_RATIO
        )

        roi_bottom = int(
            height *
            ROI_BOTTOM_RATIO
        )


        roi = frame[
            roi_top:roi_bottom,
            :
        ]


        # ====================================================
        # YOLO TRACKING
        # ====================================================

        results = model.track(

            roi,

            persist=True,

            tracker=TRACKER,

            conf=CONFIDENCE,

            iou=IOU,

            classes=aircraft_ids,

            verbose=False
        )


        detections = []


        # ====================================================
        # READ DETECTIONS
        # ====================================================

        if (
            results
            and
            results[0].boxes
            is not None
        ):

            boxes = results[0].boxes


            for box in boxes:

                # --------------------------------------------
                # Bounding box
                # --------------------------------------------

                coordinates = (
                    box.xyxy[0]
                    .tolist()
                )


                rx1, ry1, rx2, ry2 = map(
                    int,
                    coordinates
                )


                # --------------------------------------------
                # Convert ROI coordinates
                # --------------------------------------------

                x1 = max(
                    0,
                    rx1
                )

                y1 = max(
                    0,
                    ry1 + roi_top
                )

                x2 = min(
                    width - 1,
                    rx2
                )

                y2 = min(
                    height - 1,
                    ry2 + roi_top
                )


                # --------------------------------------------
                # Confidence
                # --------------------------------------------

                confidence = float(
                    box.conf[0]
                )


                # --------------------------------------------
                # Class
                # --------------------------------------------

                class_id = int(
                    box.cls[0]
                )

                class_name = str(
                    model.names[class_id]
                )


                # --------------------------------------------
                # Track ID
                # --------------------------------------------

                if box.id is not None:

                    track_id = int(
                        box.id[0]
                    )

                else:

                    track_id = -1


                # --------------------------------------------
                # Center
                # --------------------------------------------

                center_x = (
                    x1 + x2
                ) // 2

                center_y = (
                    y1 + y2
                ) // 2


                # --------------------------------------------
                # Screen center
                # --------------------------------------------

                screen_center_x = (
                    width // 2
                )

                screen_center_y = (
                    height // 2
                )


                # --------------------------------------------
                # Distance from center
                # --------------------------------------------

                center_distance = math.sqrt(

                    (
                        center_x -
                        screen_center_x
                    ) ** 2

                    +

                    (
                        center_y -
                        screen_center_y
                    ) ** 2
                )


                # --------------------------------------------
                # Pixel velocity
                # --------------------------------------------

                pixel_speed = 0.0


                if (
                    track_id >= 0
                    and
                    track_id in previous_positions
                ):

                    old_x = (
                        previous_positions[
                            track_id
                        ][0]
                    )

                    old_y = (
                        previous_positions[
                            track_id
                        ][1]
                    )

                    old_frame = (
                        previous_positions[
                            track_id
                        ][2]
                    )


                    time_difference = max(

                        (
                            frame_number -
                            old_frame
                        )
                        /
                        source_fps,

                        0.001
                    )


                    pixel_speed = math.sqrt(

                        (
                            center_x -
                            old_x
                        ) ** 2

                        +

                        (
                            center_y -
                            old_y
                        ) ** 2

                    ) / time_difference


                # --------------------------------------------
                # Save position
                # --------------------------------------------

                if track_id >= 0:

                    previous_positions[
                        track_id
                    ] = (

                        center_x,
                        center_y,
                        frame_number
                    )


                    trails[
                        track_id
                    ].append(

                        (
                            center_x,
                            center_y
                        )
                    )


                # --------------------------------------------
                # Save detection
                # --------------------------------------------

                detection = {

                    "id":
                        track_id,

                    "class":
                        class_name,

                    "confidence":
                        confidence,

                    "x1":
                        x1,

                    "y1":
                        y1,

                    "x2":
                        x2,

                    "y2":
                        y2,

                    "center_x":
                        center_x,

                    "center_y":
                        center_y,

                    "speed":
                        pixel_speed,

                    "distance":
                        center_distance
                }


                detections.append(
                    detection
                )


                # --------------------------------------------
                # CSV
                # --------------------------------------------

                csv_writer.writerow([

                    frame_number,

                    round(
                        current_video_time,
                        3
                    ),

                    track_id,

                    class_name,

                    round(
                        confidence,
                        4
                    ),

                    x1,

                    y1,

                    x2,

                    y2,

                    center_x,

                    center_y,

                    round(
                        pixel_speed,
                        2
                    ),

                    round(
                        center_distance,
                        2
                    )
                ])


        # ====================================================
        # PRIMARY TARGET
        # ====================================================

        primary_target = None


        if detections:

            primary_target = min(

                detections,

                key=lambda item:

                    (
                        item["distance"]
                        * 0.7

                        -

                        item["confidence"]
                        * 300
                    )
            )


        # ====================================================
        # DRAW TRAJECTORIES
        # ====================================================

        for track_id, points in trails.items():

            if len(points) < 2:

                continue


            points_list = list(
                points
            )


            for i in range(
                1,
                len(points_list)
            ):

                point1 = (
                    points_list[i - 1]
                )

                point2 = (
                    points_list[i]
                )


                thickness = max(

                    1,

                    int(
                        4 *
                        i /
                        len(points_list)
                    )
                )


                cv2.line(

                    frame,

                    point1,

                    point2,

                    (0, 255, 255),

                    thickness,

                    cv2.LINE_AA
                )


        # ====================================================
        # DRAW AIRCRAFT BOXES
        # ====================================================

        for detection in detections:

            x1 = detection["x1"]
            y1 = detection["y1"]

            x2 = detection["x2"]
            y2 = detection["y2"]


            # --------------------------------------------
            # Is this primary target?
            # --------------------------------------------

            is_primary = (

                primary_target is not None

                and

                detection["id"]

                ==

                primary_target["id"]
            )


            if is_primary:

                box_color = (
                    0,
                    255,
                    0
                )

                thickness = 3

            else:

                box_color = (
                    255,
                    200,
                    0
                )

                thickness = 2


            # --------------------------------------------
            # Corner box
            # --------------------------------------------

            draw_corner_box(

                frame,

                x1,

                y1,

                max(
                    1,
                    x2 - x1
                ),

                max(
                    1,
                    y2 - y1
                ),

                box_color,

                thickness
            )


            # --------------------------------------------
            # Center point
            # --------------------------------------------

            cv2.circle(

                frame,

                (
                    detection[
                        "center_x"
                    ],

                    detection[
                        "center_y"
                    ]
                ),

                5,

                (0, 255, 255),

                -1
            )


            # --------------------------------------------
            # Label
            # --------------------------------------------

            label = (

                f"AIRCRAFT "

                f"ID:{detection['id']} "

                f"{detection['confidence'] * 100:.1f}%"
            )


            label_y = max(
                25,
                y1 - 10
            )


            put_text(

                frame,

                label,

                (
                    x1,
                    label_y
                ),

                0.55,

                box_color,

                2
            )


            # --------------------------------------------
            # Velocity
            # --------------------------------------------

            velocity_text = (

                f"V: "

                f"{detection['speed']:.0f}"

                " px/s"
            )


            put_text(

                frame,

                velocity_text,

                (
                    x1,

                    min(
                        height - 10,
                        y2 + 22
                    )
                ),

                0.48,

                (255, 255, 255),

                1
            )


        # ====================================================
        # MAIN HUD PANEL
        # ====================================================

        panel_width = 390

        panel_height = 205


        overlay = frame.copy()


        cv2.rectangle(

            overlay,

            (18, 18),

            (
                panel_width,
                panel_height
            ),

            (0, 0, 0),

            -1
        )


        frame = cv2.addWeighted(

            overlay,

            0.70,

            frame,

            0.30,

            0
        )


        # ====================================================
        # HUD TITLE
        # ====================================================

        put_text(

            frame,

            "AIRCRAFT VISION AI",

            (32, 48),

            0.78,

            (0, 255, 255),

            2
        )


        put_text(

            frame,

            "YOLO + BYTE TRACK",

            (32, 73),

            0.52,

            (180, 220, 255),

            1
        )


        # ====================================================
        # PROCESSING FPS
        # ====================================================

        elapsed_time = max(

            time.time()
            -
            start_time,

            0.001
        )


        processing_fps = (

            frame_number
            /
            elapsed_time
        )


        put_text(

            frame,

            f"FPS       : {processing_fps:.1f}",

            (32, 103),

            0.55
        )


        # ====================================================
        # FRAME
        # ====================================================

        put_text(

            frame,

            f"FRAME     : {frame_number:05d}",

            (32, 128),

            0.55
        )


        # ====================================================
        # TRACK COUNT
        # ====================================================

        put_text(

            frame,

            f"TRACKS    : {len(detections)}",

            (32, 153),

            0.55
        )


        # ====================================================
        # STATUS
        # ====================================================

        if primary_target:

            put_text(

                frame,

                (
                    f"PRIMARY   : "
                    f"ID "
                    f"{primary_target['id']}"
                ),

                (32, 178),

                0.55,

                (0, 255, 0),

                2
            )

        else:

            put_text(

                frame,

                "PRIMARY   : SEARCHING",

                (32, 178),

                0.55,

                (0, 200, 255),

                1
            )


        # ====================================================
        # PRIMARY TARGET PANEL
        # ====================================================

        if primary_target:

            panel_x = width - 310

            panel_y = 25

            panel_width = 285

            panel_height = 155


            overlay = frame.copy()


            cv2.rectangle(

                overlay,

                (
                    panel_x,
                    panel_y
                ),

                (
                    panel_x +
                    panel_width,

                    panel_y +
                    panel_height
                ),

                (0, 0, 0),

                -1
            )


            frame = cv2.addWeighted(

                overlay,

                0.65,

                frame,

                0.35,

                0
            )


            # --------------------------------------------
            # Target title
            # --------------------------------------------

            put_text(

                frame,

                "PRIMARY AIRCRAFT",

                (
                    panel_x + 15,
                    panel_y + 28
                ),

                0.65,

                (0, 255, 0),

                2
            )


            # --------------------------------------------
            # ID
            # --------------------------------------------

            put_text(

                frame,

                (
                    f"ID       : "
                    f"{primary_target['id']}"
                ),

                (
                    panel_x + 15,
                    panel_y + 55
                ),

                0.50
            )


            # --------------------------------------------
            # Confidence
            # --------------------------------------------

            put_text(

                frame,

                (
                    f"CONF     : "
                    f"{primary_target['confidence'] * 100:.1f}%"
                ),

                (
                    panel_x + 15,
                    panel_y + 78
                ),

                0.50
            )


            # --------------------------------------------
            # Coordinates
            # --------------------------------------------

            put_text(

                frame,

                (
                    f"X,Y      : "
                    f"{primary_target['center_x']},"
                    f"{primary_target['center_y']}"
                ),

                (
                    panel_x + 15,
                    panel_y + 101
                ),

                0.50
            )


            # --------------------------------------------
            # Velocity
            # --------------------------------------------

            put_text(

                frame,

                (
                    f"VELOCITY : "
                    f"{primary_target['speed']:.0f}px/s"
                ),

                (
                    panel_x + 15,
                    panel_y + 124
                ),

                0.50
            )


            # --------------------------------------------
            # Status
            # --------------------------------------------

            put_text(

                frame,

                "STATUS   : TRACKING",

                (
                    panel_x + 15,
                    panel_y + 147
                ),

                0.50,

                (0, 255, 0),

                2
            )


        # ====================================================
        # CROSSHAIR
        # ====================================================

        if DRAW_RETICLE:

            draw_reticle(
                frame
            )


        # ====================================================
        # BOTTOM STATUS
        # ====================================================

        put_text(

            frame,

            "COMPUTER VISION ACTIVE",

            (
                25,
                height - 22
            ),

            0.55,

            (0, 255, 0),

            2
        )


        put_text(

            frame,

            "Q = EXIT",

            (
                width - 110,
                height - 22
            ),

            0.45,

            (220, 220, 220),

            1
        )


        # ====================================================
        # SAVE OUTPUT
        # ====================================================

        writer.write(
            frame
        )


        # ====================================================
        # DISPLAY
        # ====================================================

        if SHOW_WINDOW:

            cv2.imshow(

                "Advanced YOLO Aircraft Tracking",

                frame
            )


            key = cv2.waitKey(
                1
            ) & 0xFF


            if key == ord("q"):

                break


    # ========================================================
    # RELEASE
    # ========================================================

    cap.release()

    writer.release()

    csv_file.close()

    cv2.destroyAllWindows()


    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    print()
    print("====================================")
    print("PROCESSING COMPLETED")
    print("====================================")

    print(
        "Output video:"
    )

    print(
        OUTPUT_VIDEO
    )

    print()

    print(
        "Telemetry CSV:"
    )

    print(
        OUTPUT_CSV
    )

    print("====================================")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
