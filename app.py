import cv2
from flask import Flask, render_template, request, send_file
from PIL import Image, ImageEnhance, ImageOps, ImageMath, ImageFilter
import math
from io import BytesIO

import numpy as np


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])

def upload():
    # Xử lý chuyển đổi ảnh ở đây
    image_conversion_type = request.form['conversionType']

    # Lấy hình ảnh từ form
    image_file = request.files['file']
    if not image_file:
        return 'No image file provided'

    # Đọc hình ảnh từ BytesIO
    image_bytes = BytesIO(image_file.read())
    img = Image.open(image_bytes).convert('RGB')
    # Áp dụng chuyển đổi tùy thuộc vào loại chuyển đổi
    if image_conversion_type == 'negative':
        img = apply_negative(img)
    elif image_conversion_type == 'threshold':
        img = apply_threshold(img)
    elif image_conversion_type == 'logarithmic':
        img = apply_logarithmic(img)
    elif image_conversion_type == 'exponential':
        img = apply_exponential(img)
    elif image_conversion_type == 'histogram_equalization':
        img = apply_histogram_equalization(img)
    elif image_conversion_type == 'weighted_averaging':
        img = apply_weighted_averaging(img)
    elif image_conversion_type == 'median_filter':
        img = apply_median_filter(img)
    elif image_conversion_type == 'roberts_operator':
        img = apply_roberts_operator(img)
    elif image_conversion_type == 'sobel_operator':
        img = apply_sobel_operator(img)
    elif image_conversion_type == 'prewitt_operator':
        img = apply_prewitt_operator(img)
    elif image_conversion_type == 'laplacian_operator':
        img = apply_laplacian_operator(img)
    elif image_conversion_type== 'canny_operator':
        img = apply_canny_operator(img)
    elif image_conversion_type== 'otsu_threshold':
        img = apply_otsu_threshold(img)
    elif image_conversion_type== 'erosion_image':
        img = erosion_image(img,3)
    elif image_conversion_type== 'dilation_image':
        img = dilation_image(img,3)
    elif image_conversion_type== 'closing_image':
        img = closing_image(img,3)
    elif image_conversion_type== 'opening_image':
        img = opening_image(img,3)    
    else:
        return 'Invalid conversion type'

    # Lưu hình ảnh kết quả vào BytesIO
    result_bytes = BytesIO()
    img.save(result_bytes, format='JPEG')
    result_bytes.seek(0)

    return send_file(result_bytes, mimetype='image/jpeg')

def apply_negative(image):
    # Chuyển đổi thành ảnh âm bản
    inverted_image = ImageOps.invert(image)
    
    return inverted_image


def apply_threshold(image):
    # Chuyển đổi thành ảnh xám
    gray_image = ImageOps.grayscale(image)

    # Áp dụng ngưỡng
    threshold_value = 128  # Giá trị ngưỡng, có thể điều chỉnh
    thresholded_image = gray_image.point(lambda p: p > threshold_value and 255)

    return thresholded_image

def apply_logarithmic(image):
    # Chuyển đổi thành ảnh xám
    gray_image = ImageOps.grayscale(image)

    # Áp dụng chuyển đổi logarithmic
    log_enhancer = ImageEnhance.Color(gray_image)
    log_image = log_enhancer.enhance(1.5)  

    return log_image

def apply_exponential(image):
    # Chuyển đổi thành ảnh xám
    gray_image = ImageOps.grayscale(image)

    # Áp dụng chuyển đổi exponential
    exp_enhancer = ImageEnhance.Brightness(gray_image)
    exp_image = exp_enhancer.enhance(1.5)  

    return exp_image



def apply_histogram_equalization(image):
    # Chuyển đổi thành ảnh xám
    gray_image = ImageOps.grayscale(image)

    # Áp dụng Histogram Equalization
    equalized_image = ImageOps.equalize(gray_image)

    return equalized_image

def apply_weighted_averaging(image):
    # Áp dụng trung bình có trọng số
    weighted_average_image = image.filter(ImageFilter.BLUR)

    return weighted_average_image

def apply_median_filter(image):
    # Áp dụng Median Filter
    median_filtered_image = image.filter(ImageFilter.MedianFilter(size=3))

    return median_filtered_image

def apply_roberts_operator(image):
    # Áp dụng Roberts Operator
    roberts_image = image.filter(ImageFilter.FIND_EDGES)

    return roberts_image

def apply_sobel_operator(image):
    # Áp dụng Sobel Operator
    sobel_image = image.filter(ImageFilter.CONTOUR)

    return sobel_image

def apply_prewitt_operator(image):
    # Bộ lọc Prewitt theo hướng x (đạo hàm theo chiều ngang)
    kernel_prewitt_x = ImageFilter.Kernel((3, 3), [-1, 0, 1, -1, 0, 1, -1, 0, 1])

    # Bộ lọc Prewitt theo hướng y (đạo hàm theo chiều dọc)
    kernel_prewitt_y = ImageFilter.Kernel((3, 3), [-1, -1, -1, 0, 0, 0, 1, 1, 1])

    # Áp dụng bộ lọc Prewitt theo chiều ngang và chiều dọc
    prewitt_x = image.filter(kernel_prewitt_x)
    prewitt_y = image.filter(kernel_prewitt_y)

    # Tổng hợp kết quả từ cả hai hướng
    prewitt_result = Image.blend(prewitt_x, prewitt_y, alpha=0.5)

    return prewitt_result

def apply_laplacian_operator(image):
    # Convert PIL Image to NumPy array
    img_np = np.array(image)
    # Convert the image to grayscale if it's a color image
    if len(img_np.shape) == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    # Apply the Laplacian operator
    laplacian = cv2.Laplacian(img_np, cv2.CV_64F)
    # Convert the result to uint8 (0-255 range) for display
    laplacian = np.uint8(np.abs(laplacian))
    # Convert NumPy array back to PIL Image
    laplacian_image = Image.fromarray(laplacian)
    return laplacian_image

def apply_canny_operator(image):
    # Convert PIL Image to NumPy array
    img_np = np.array(image)
    # Convert the image to grayscale if it's a color image
    if len(img_np.shape) == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise and improve edge detection
    blurred_img = cv2.GaussianBlur(img_np, (5, 5), 0)
    # Apply Canny edge detector
    edges = cv2.Canny(blurred_img, 50, 150)  # Adjust the thresholds as needed
    # Convert the result to PIL Image
    edges_image = Image.fromarray(edges)
    return edges_image

def apply_otsu_threshold(image):
    # Convert PIL Image to NumPy array
    img_np = np.array(image)

    # Convert the image to grayscale if it's a color image
    if len(img_np.shape) == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding
    _, otsu_thresholded = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Convert the result to PIL Image
    otsu_thresholded_image = Image.fromarray(otsu_thresholded)

    return otsu_thresholded_image

def erosion_image(image, kernel_size=3):
    img_np = np.array(image)

    # Convert the image to grayscale if it's a color image
    if len(img_np.shape) == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

    # Binarize the image (convert to black and white)
    _, binary_image = cv2.threshold(img_np, 127, 255, cv2.THRESH_BINARY)

    # Define a kernel for erosion
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Perform erosion using OpenCV's erode function
    eroded_image = cv2.erode(binary_image, kernel, iterations=1)

    # Convert the result to PIL Image
    eroded_image_pil = Image.fromarray(eroded_image)

    return eroded_image_pil
def dilation_image(image, kernel_size=3):
    img_np = np.array(image)

    # Convert the image to grayscale if it's a color image
    if len(img_np.shape) == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

    # Binarize the image (convert to black and white)
    _, binary_image = cv2.threshold(img_np, 127, 255, cv2.THRESH_BINARY)

    # Define a kernel for dilation
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Perform dilation using OpenCV's dilate function
    dilated_image = cv2.dilate(binary_image, kernel, iterations=1)

    # Convert the result to PIL Image
    dilated_image_pil = Image.fromarray(dilated_image)

    return dilated_image_pil

def closing_image(image, kernel_size=3):
    img_np = np.array(image)

    # Convert the image to grayscale if it's a color image
    if len(img_np.shape) == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

    # Binarize the image (convert to black and white)
    _, binary_image = cv2.threshold(img_np, 127, 255, cv2.THRESH_BINARY)

    # Define a square structuring element for closing
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Perform closing using OpenCV's morphologyEx function
    closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    # Convert the result to PIL Image
    closed_image_pil = Image.fromarray(closed_image)

    return closed_image_pil

def opening_image(image, kernel_size=3):
    img_np = np.array(image)
    # Convert the image to grayscale if it's a color image
    if len(img_np.shape) == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    # Binarize the image (convert to black and white)
    _, binary_image = cv2.threshold(img_np, 127, 255, cv2.THRESH_BINARY)

    # Define a square structuring element for opening
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Perform opening using OpenCV's morphologyEx function
    opened_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

    # Convert the result to PIL Image
    opened_image_pil = Image.fromarray(opened_image)

    return opened_image_pil
if __name__ == '__main__':
    app.run(debug=True)
