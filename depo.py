import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="CV Pro", layout="wide")

st.title(" CV Image Processing App")

# Sidebar: Choose Algorithm
algorithm = st.sidebar.selectbox(
    "Choose Algorithm",
    [
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


# Algorithm Info

st.sidebar.markdown("---")
st.sidebar.subheader(" Algorithm Info:")

if algorithm == "Resize & Rotate":
    st.sidebar.info("""
- Input Image:
- Any RGB image

 What it does:
- Resize image
- Rotate image (90/180/270)
""")

elif algorithm == "Invert & Grayscale":
    st.sidebar.info("""
 Input Image:
- Any colored image

 What it does:
- Inverts colors
- Converts to grayscale
""")

elif algorithm == "Histogram":
    st.sidebar.info("""
 Input Image:
- Any image

 What it does:
- Shows pixel intensity distribution
""")

elif algorithm == "Filtering":
    st.sidebar.info("""
 Input Image:
- Noisy image recommended

 What it does:
- Mean filter
- Gaussian filter
""")

elif algorithm == "Edge Detection":
    st.sidebar.info("""
 Input Image:
- Clear object image like bulidings 

 What it does:
- Detects edges using Canny
""")

elif algorithm == "Scaling":
    st.sidebar.info("""
 Input Image:
- Any image

 What it does:
- Resize image (half, double, fixed)
""")

elif algorithm == "Harris Corner Detection":
    st.sidebar.info("""
 Input Image:
- Images with corners (buildings, objects)

 What it does:
- Detects corners in image
""")

elif algorithm == "Otsu Segmentation":
    st.sidebar.info("""
 Input Image:
- Image with clear foreground/background

 What it does:
- Automatic threshold segmentation
""")

elif algorithm == "KMeans Segmentation":
    st.sidebar.info("""
 Input Image:
- Any colored image

 What it does:
- Clusters image into K regions
""")

elif algorithm == "GrabCut Segmentation":
    st.sidebar.info("""
 Input Image:
- Object in center of image

 What it does:
- Extracts foreground object
""")


# Upload Image

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
        st.image(img, caption="Original Image", use_container_width=True)

 
    # Resize & Rotate

    elif algorithm == "Resize & Rotate":

        size = st.sidebar.slider("Resize Size", 100, 800, 256)

        img_resize = cv2.resize(img, (size, size))

        rotate = st.sidebar.selectbox("Rotation", ["90", "180", "270"])

        if rotate == "90":
            img_rotate = cv2.rotate(img_resize, cv2.ROTATE_90_CLOCKWISE)
        elif rotate == "180":
            img_rotate = cv2.rotate(img_resize, cv2.ROTATE_180)
        else:
            img_rotate = cv2.rotate(img_resize, cv2.ROTATE_90_COUNTERCLOCKWISE)

        col1, col2 = st.columns(2)

        with col1:
            st.image(img_resize, caption="Resized")

        with col2:
            st.image(img_rotate, caption="Rotated")

  
    # Invert & Grayscale
 
    elif algorithm == "Invert & Grayscale":

        invert = 255 - img
        gray = cv2.cvtColor(invert, cv2.COLOR_RGB2GRAY)

        col1, col2 = st.columns(2)

        with col1:
            st.image(invert, caption="Inverted")

        with col2:
            st.image(gray, caption="Grayscale")

    
    # Histogram
   
    elif algorithm == "Histogram":

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        fig, ax = plt.subplots()
        ax.hist(gray.flatten(), bins=50)
        ax.set_title("Histogram")

        st.pyplot(fig)

    
    # Filtering
   
    elif algorithm == "Filtering":

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        kernel = st.sidebar.slider("Kernel Size", 3, 15, 5, step=2)

        mean = cv2.blur(gray, (kernel, kernel))
        gaussian = cv2.GaussianBlur(gray, (kernel, kernel), 0)

        col1, col2 = st.columns(2)

        with col1:
            st.image(mean, caption="Mean Filter")

        with col2:
            st.image(gaussian, caption="Gaussian Filter")

   
    # Edge Detection

    elif algorithm == "Edge Detection":

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        t1 = st.sidebar.slider("Threshold 1", 0, 255, 100)
        t2 = st.sidebar.slider("Threshold 2", 0, 255, 200)

        edges = cv2.Canny(gray, t1, t2)

        st.image(edges, caption="Canny Edge Detection")

  
    # Scaling
  
    elif algorithm == "Scaling":

        h, w = img.shape[:2]

        half = cv2.resize(img, (w//2, h//2))
        double = cv2.resize(img, (w*2, h*2))
        fixed = cv2.resize(img, (200, 200))

        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(half, caption="Half")

        with col2:
            st.image(double, caption="Double")

        with col3:
            st.image(fixed, caption="200x200")

   
    # Harris Corner Detection
  
    elif algorithm == "Harris Corner Detection":

        threshold = st.sidebar.slider("Threshold %", 1, 20, 1)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = np.float32(gray)

        dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        dst = cv2.dilate(dst, None)

        result = img.copy()
        result[dst > (threshold/100)*dst.max()] = [255, 0, 0]

        st.image(result, caption="Corners")

    
    # Otsu

    elif algorithm == "Otsu Segmentation":

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        _, mask = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        result = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)

        col1, col2 = st.columns(2)

        with col1:
            st.image(mask, caption="Mask")

        with col2:
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Result")

    # KMeans
   
    elif algorithm == "KMeans Segmentation":

        K = st.sidebar.slider("Clusters", 2, 10, 4)

        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        Z = lab.reshape((-1, 3)).astype(np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)

        _, labels, centers = cv2.kmeans(
            Z, K, None, criteria, 5, cv2.KMEANS_PP_CENTERS
        )

        centers = np.uint8(centers)
        segmented = centers[labels.flatten()]
        segmented = segmented.reshape(lab.shape)

        segmented = cv2.cvtColor(segmented, cv2.COLOR_LAB2RGB)

        st.image(segmented, caption="KMeans Result")

    # GrabCut
   
    elif algorithm == "GrabCut Segmentation":

        h, w = img_bgr.shape[:2]

        rect = (int(0.1*w), int(0.1*h), int(0.8*w), int(0.8*h))

        mask = np.zeros((h, w), np.uint8)
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)

        cv2.grabCut(img_bgr, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)

        binary = np.where((mask==1) | (mask==3), 255, 0).astype("uint8")

        result = cv2.bitwise_and(img_bgr, img_bgr, mask=binary)

        col1, col2 = st.columns(2)

        with col1:
            st.image(binary, caption="Mask")

        with col2:
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Result")