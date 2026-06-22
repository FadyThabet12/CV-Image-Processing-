
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="CV Pro", layout="wide")

st.title(" CV Image Processing App")

# Sidebar Menu
algorithm = st.sidebar.selectbox(
    "Choose Algorithm",
    [
        "Original Image",
        "Resize & Rotate",
        "Invert & Grayscale",
        "Histogram",
        "Filtering",
        "Edge Detection",
        "Scaling",
        "Harris Corner Detection",
        "Otsu Segmentation",
        "KMeans Segmentation",
        "GrabCut Segmentation"
    ]
)

uploaded_file = st.file_uploader(
    f"Upload Image for {algorithm}",
    type=["jpg", "jpeg", "png", "jfif"]
)

if uploaded_file:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    
    # Original Image
  
    if algorithm == "Original Image":

        st.subheader("Original Image")
        st.image(img, use_container_width=True)

   
    # Resize & Rotate
   
    elif algorithm == "Resize & Rotate":

        size = st.sidebar.slider(
            "Resize Dimension",
            100,
            800,
            256
        )

        img_resize = cv2.resize(img, (size, size))

        rotation = st.sidebar.selectbox(
            "Rotation",
            ["90", "180", "270"]
        )

        if rotation == "90":
            img_rotate = cv2.rotate(
                img_resize,
                cv2.ROTATE_90_CLOCKWISE
            )

        elif rotation == "180":
            img_rotate = cv2.rotate(
                img_resize,
                cv2.ROTATE_180
            )

        else:
            img_rotate = cv2.rotate(
                img_resize,
                cv2.ROTATE_90_COUNTERCLOCKWISE
            )

        col1, col2 = st.columns(2)

        with col1:
            st.image(img_resize, caption="Resized")

        with col2:
            st.image(img_rotate, caption="Rotated")

    
    # Invert & Grayscale
   
    elif algorithm == "Invert & Grayscale":

        invert = 255 - img

        gray = cv2.cvtColor(
            invert,
            cv2.COLOR_RGB2GRAY
        )

        col1, col2 = st.columns(2)

        with col1:
            st.image(invert, caption="Inverted")

        with col2:
            st.image(gray, caption="Grayscale")

    
    # Histogram
   
    elif algorithm == "Histogram":

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_RGB2GRAY
        )

        fig, ax = plt.subplots()

        ax.hist(
            gray.flatten(),
            bins=50
        )

        ax.set_title("Histogram")

        st.pyplot(fig)

    
    # Filtering
    
    elif algorithm == "Filtering":

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_RGB2GRAY
        )

        kernel = st.sidebar.slider(
            "Kernel Size",
            3,
            15,
            5,
            step=2
        )

        mean = cv2.blur(
            gray,
            (kernel, kernel)
        )

        gaussian = cv2.GaussianBlur(
            gray,
            (kernel, kernel),
            0
        )

        col1, col2 = st.columns(2)

        with col1:
            st.image(mean, caption="Mean Filter")

        with col2:
            st.image(
                gaussian,
                caption="Gaussian Filter"
            )

    
    # Edge Detection
   
    elif algorithm == "Edge Detection":

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_RGB2GRAY
        )

        threshold1 = st.sidebar.slider(
            "Threshold 1",
            0,
            255,
            100
        )

        threshold2 = st.sidebar.slider(
            "Threshold 2",
            0,
            255,
            200
        )

        edges = cv2.Canny(
            gray,
            threshold1,
            threshold2
        )

        st.image(
            edges,
            caption="Canny Edge Detection"
        )

    
    # Scaling
    
    elif algorithm == "Scaling":

        h, w = img.shape[:2]

        half = cv2.resize(
            img,
            (w // 2, h // 2)
        )

        double = cv2.resize(
            img,
            (w * 2, h * 2)
        )

        fixed = cv2.resize(
            img,
            (200, 200)
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(
                half,
                caption="Half Size"
            )

        with col2:
            st.image(
                double,
                caption="Double Size"
            )

        with col3:
            st.image(
                fixed,
                caption="200 x 200"
            )

   
    # Harris Corner Detection
   
    elif algorithm == "Harris Corner Detection":

        threshold = st.sidebar.slider(
            "Corner Threshold %",
            1,
            20,
            1
        )

        gray_corner = cv2.cvtColor(
            img,
            cv2.COLOR_RGB2GRAY
        )

        gray_corner = np.float32(
            gray_corner
        )

        dst = cv2.cornerHarris(
            gray_corner,
            2,
            3,
            0.04
        )

        dst = cv2.dilate(
            dst,
            None
        )

        result = img.copy()

        result[
            dst > (threshold / 100) * dst.max()
        ] = [255, 0, 0]

        st.image(
            result,
            caption="Detected Corners"
        )

 
    # Otsu Segmentation
    
    elif algorithm == "Otsu Segmentation":

        gray = cv2.cvtColor(
            img_bgr,
            cv2.COLOR_BGR2GRAY
        )

        threshold_value, mask = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        foreground = cv2.bitwise_and(
            img_bgr,
            img_bgr,
            mask=mask
        )

        st.write(
            f"Otsu Threshold = {threshold_value:.2f}"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                mask,
                caption="Mask"
            )

        with col2:
            st.image(
                cv2.cvtColor(
                    foreground,
                    cv2.COLOR_BGR2RGB
                ),
                caption="Foreground"
            )

    
    # KMeans Segmentation
    
    elif algorithm == "KMeans Segmentation":

        K = st.sidebar.slider(
            "Number of Clusters",
            2,
            10,
            4
        )

        lab = cv2.cvtColor(
            img_bgr,
            cv2.COLOR_BGR2LAB
        )

        Z = lab.reshape((-1, 3))
        Z = np.float32(Z)

        criteria = (
            cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER,
            20,
            1.0
        )

        _, labels, centers = cv2.kmeans(
            Z,
            K,
            None,
            criteria,
            5,
            cv2.KMEANS_PP_CENTERS
        )

        centers = np.uint8(
            centers
        )

        segmented = centers[
            labels.flatten()
        ]

        segmented = segmented.reshape(
            lab.shape
        )

        segmented = cv2.cvtColor(
            segmented,
            cv2.COLOR_LAB2RGB
        )

        st.image(
            segmented,
            caption="KMeans Segmentation"
        )

   
    # GrabCut Segmentation
    
    elif algorithm == "GrabCut Segmentation":

        height, width = img_bgr.shape[:2]

        rectangle = (
            int(0.1 * width),
            int(0.1 * height),
            int(0.8 * width),
            int(0.8 * height)
        )

        mask_gc = np.zeros(
            (height, width),
            np.uint8
        )

        bgd = np.zeros(
            (1, 65),
            np.float64
        )

        fgd = np.zeros(
            (1, 65),
            np.float64
        )

        cv2.grabCut(
            img_bgr,
            mask_gc,
            rectangle,
            bgd,
            fgd,
            5,
            cv2.GC_INIT_WITH_RECT
        )

        binary_mask = np.where(
            (mask_gc == 1) |
            (mask_gc == 3),
            255,
            0
        ).astype("uint8")

        result = cv2.bitwise_and(
            img_bgr,
            img_bgr,
            mask=binary_mask
        )

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                binary_mask,
                caption="Foreground Mask"
            )

        with col2:
            st.image(
                cv2.cvtColor(
                    result,
                    cv2.COLOR_BGR2RGB
                ),
                caption="Extracted Foreground"
            )
