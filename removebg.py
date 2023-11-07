# Import necessary libraries
from PIL import Image
import math
from rembg import remove

from io import BytesIO
import base64



# Define a function to automatically crop an image
def autocrop_image(img, border=0):
    """
    Automatically crop an image while preserving the original aspect ratio.

    Args:
        img (PIL.Image.Image): The input image.
        border (int): The width of the border to add around the cropped image.

    Returns:
        PIL.Image.Image: The cropped and bordered image.
    """
    # Get the bounding box of the image
    bbox = img.getbbox()

    # Crop the image to the contents of the bounding box
    img = img.crop(bbox)

    # Determine the scale and height of the cropped image
    (scale, height) = img.size

    # Add a border
    scale += border * 2
    height += border * 2

    # Create a new image object for the output image
    cropped_image = Image.new("RGBA", (scale, height), (0, 0, 0, 0))

    # Paste the cropped image onto the new image with a border
    cropped_image.paste(img, (border, border))

    # Return the cropped image
    return cropped_image


# Define a function to resize an image while maintaining the aspect ratio
def resize_image(img, myScale):
    """
    Resize an image while maintaining the aspect ratio.

    Args:
        img (PIL.Image.Image): The input image.
        myScale (int): The desired size for the larger dimension (width or height).

    Returns:
        PIL.Image.Image: The resized image.
    """

    img_width, img_height = img.size

    if img_height > img_width:  # Portrait image
        hpercent = (myScale / float(img_height))
        wsize = int((float(img_width) * float(hpercent)))
        resized_img = img.resize((wsize, myScale), Image.Resampling.LANCZOS)
    elif img_width > img_height:  # Landscape image
        wpercent = (myScale / float(img_width))
        hsize = int((float(img_height) * float(wpercent)))
        resized_img = img.resize((myScale, hsize), Image.Resampling.LANCZOS)

    return resized_img


# Define a function to resize the canvas and center the image
def resize_canvas(img, canvas_width, canvas_height):
    """
    Resize the canvas and center the image on it.

    Args:
        img (PIL.Image.Image): The input image.
        canvas_width (int): The width of the canvas.
        canvas_height (int): The height of the canvas.

    Returns:
        PIL.Image.Image: The image centered on the new canvas.
    """
    old_width, old_height = img.size

    # Center the image on the new canvas
    x1 = int(math.floor((canvas_width - old_width) / 2))
    y1 = int(math.floor((canvas_height - old_height) / 2))

    # Determine the background color based on the image mode
    mode = img.mode
    if len(mode) == 1:  # L, 1
        new_background = (255)
    if len(mode) == 3:  # RGB
        new_background = (255, 255, 255)
    if len(mode) == 4:  # RGBA, CMYK
        new_background = (255, 255, 255, 255)

    # Create a new image with the specified background color and dimensions
    newImage = Image.new(mode, (canvas_width, canvas_height), new_background)

    # Paste the image onto the new canvas, centering it
    newImage.alpha_composite(
        img, ((canvas_width - old_width) // 2, (canvas_height - old_height) // 2))

    # Return the new image
    return newImage

