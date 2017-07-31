import cv2
import numpy as np
import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    #cv2.fillPoly(mask, np.int32([np.array(vertices)]), ignore_mask_color)
    mask = cv2.inRange(img,(80,80,80), (255,255,255)) * 255
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
    Returns array of line segments.
    """
    return cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)


def weighted_img(img, initial_img, alpha=0.8, beta=1., gamma=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, alpha, img, beta, gamma)


def transform_line_to_polar(coords):
    """
    Transforms line segment into parametrized lines (r, theta)

    `coords` array of coordinates of line segment (x1,y1,x2,y2)

    Returns: Parameters r and theta of the parametrized line.
    """
    x1,y1,x2,y2 = coords
    delta_x = float(x2 - x1)
    delta_y = float(y2 - y1)
    if delta_y == 0.0:
        return x1,math.pi/2
    else:
        theta = math.atan(-delta_x/delta_y)
        r = x1 * math.cos(theta) + y1 * math.sin(theta)
        return r, theta


def transform_line_from_polar(line, y1, y2):
    """
    Transforms parametrized line (r, theta) into line segment endpoint coordinates.

    `line`: array [r,theta] of the parametrized line in polar form
    `y1, y2`: y-coordinates of the resulting line segment

    Returns: array with line endpoint coordinates [x1,y1,x2,y2]
    """
    r, theta = line
    if theta == math.pi/2:
        raise ValueError
    x1 = (r - y1 * math.sin(theta)) / math.cos(theta)
    x2 = (r - y2 * math.sin(theta)) / math.cos(theta)
    return [x1, y1, x2, y2]


def draw_line(img, line, color, thickness=2):
    """
    Draw a line into an image.

    `img`: destination image
    `line`: array with line endpoint coordinates [x1,y1,x2,y2]
    `color`: color of the line
    `thickness`: thickness of the line
    """
    x1,y1,x2,y2 = line
    cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)


def interpolate_lane_line(polar_line_segments):
    """
    Interpolates a resulting lane line in polar form from a collection of line:

    `polar_line_segments`: array of lines in polar form [r,theta]

    Returns: resulting line by calculating the mean of r_i and theta_i of the provided lines.
    """
    if len(polar_line_segments):
        return np.array(polar_line_segments).mean(axis=0)
    else:
        return None

# Persistent lane line parameters for low pass filtering of form [r,theta]
last_left_line = None
last_right_line = None


def reset_lane_lines():
    """
    Reset persistent line parameters before a new line lane annotation.
    """
    global last_left_line
    global last_right_line
    last_left_line = None
    last_right_line = None


def identify_lane_lines(lines, use_low_pass_filter=False):
    """
    Identify left and right lane lines given a collection of line segments extracted from the
    image.

    `lines`: array of line segment endpoint coordinates [x1,y1,x2,y2]
    `use_low_pass_filter`: flag to activate low_pass_filtering for video annotation.

    Returns: array of left and right lane lines in polar form (r, theta)
    """
    left_lines = []
    right_lines = []
    global last_left_line
    global last_right_line

    if lines is None:
        return [None, None]

    for line in lines:
        r, theta = transform_line_to_polar(line[0])
        x11,y11,x12,y12 = line[0]
        if abs(theta) > math.pi * 0.45:
            pass
        elif theta > 0.0:
            left_lines.append([r, theta])
        else:
            right_lines.append([r, theta])

    left_line = interpolate_lane_line(left_lines)
    right_line = interpolate_lane_line(right_lines)

    if False and use_low_pass_filter:
        if left_line is not None:
            if last_left_line is not None:
                left_line = low_pass_filter(left_line, last_left_line)
        elif last_left_line is not None:
            left_line = list(last_left_line)

        if right_line is not None:
            if last_right_line is not None:
                right_line = low_pass_filter(right_line, last_right_line)
        elif last_right_line is not None:
            right_line = last_right_line


        last_left_line = left_line
        last_right_line = right_line

    return [left_line, right_line]


def low_pass_filter(line, last_line):
    """
    Simple low pass filter to smooth out lane line paramters during video annotation.

    `line`: line in polar form [r,theta]
    `last_line`: parameters of previous iteration in polar form [r,theta]
    """
    a = 0.20
    b = 1.0 - a
    new_r = line[0] * a + last_line[0] * b
    new_theta = line[1] * a + last_line[1] * b
    return [new_r, new_theta]


def draw_lane_lines(img, lines):
    """
    Draw the lane lines into the image.

    `img`: Destination image. Lane lines will be alpha blended into this image.
    `line`: array with left/right lane line of form [r,theta]
    `y1,y2`: y-coordinates of resulting lane lines.

    Returns: annotated image.
    """

    h,w = img.shape[:2]
    y1 = 0
    y2 = h

    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    left_line = lines[0]
    right_line = lines[1]
    if left_line is not None:
        left_line = transform_line_from_polar(left_line, y1, y2)
        draw_line(line_img, left_line, (255,0,0), thickness=8)

    if right_line is not None:
        right_line = transform_line_from_polar(right_line, y1, y2)
        draw_line(line_img, right_line, (0,255,0), thickness=8)

    return weighted_img(line_img, img)



def detect_lane_lines(img, use_low_pass_filter=False):
    h,w = img.shape[0:2]
    y1 = h
    y2 = 0
    dimg = grayscale(img)
    #dimg = img
    dimg = gaussian_blur(dimg, 7)
    dimg = canny(dimg, 30, 100)
    #dimg = region_of_interest(dimg, [[0.00 * w,h], [0.48 * w, y1], [0.52 * w, y1], [1.0 * w,h]])
    lines = hough_lines(dimg, 1, math.pi/180, 16, 10, 5)
    lane_lines = identify_lane_lines(lines, use_low_pass_filter)
    return lane_lines



def steering_angle(lane_lines):
    left_line, right_line = lane_lines
    if left_line is None and right_line is None:
        return 0.0

    if left_line is not None:
        return -0.5
        #print left_line
        #return left_line[1]

    if right_line is not None:
        #print right_line
        #return right_line[1]
        return 0.5    
    

def predict_angle(img):
    lane_lines = detect_lane_lines(img)
    angle = steering_angle(lane_lines)
    return angle

        

def process_img_and_show(img):
    lane_lines = detect_lane_lines(img, use_low_pass_filter=True)
    img = draw_lane_lines(img, lane_lines)
    angle = steering_angle(lane_lines)
  #  cv2.imshow("test", img)
  #  cv2.waitKey(10)    


def process_and_show(in_file_name):
    img = cv2.imread(in_file_name)
    process_img_and_show(img)
   


if __name__ == "__main__":    
    import glob
    import random
    file_list = glob.glob("IMG/*.png")

    while True:
        file_name = random.choice(file_list)
        process_and_show(file_name)

