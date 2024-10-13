from PIL import Image
import numpy as np

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = r * 0.2989 + g * 0.5870 + b * 0.1140

    return gray

def sobel_convolution(image, sobel_x, sobel_y):
    if len(image.shape) == 3:
        img_height, img_width, _ = image.shape
    elif len(image.shape) == 2:
        img_height, img_width = image.shape
    
    if len(image.shape) == 3:
        gray_image = rgb2gray(image)
    elif len(image.shape) == 2:
        gray_image = image
    
    padded_img = np.pad(gray_image, ((1, 1), (1, 1)), mode='constant')
    grad_x = np.zeros(gray_image.shape)
    grad_y = np.zeros(gray_image.shape)
    
    sobel_x_flipped = np.flipud(np.fliplr(sobel_x))
    sobel_y_flipped = np.flipud(np.fliplr(sobel_y))
    for i in range(1, img_height):
        for j in range(1, img_width):
            grad_x[i, j] = np.sum(padded_img[i - 1 : i + 2, j - 1 : j + 2] * sobel_x_flipped)
            grad_y[i, j] = np.sum(padded_img[i - 1 : i + 2, j - 1 : j + 2] * sobel_y_flipped)

    output_img = np.sqrt(np.square(grad_x) + np.square(grad_y))
    output_img = (output_img / np.max(output_img) * 255).astype(np.uint8)

    return output_img

def main():
    img = Image.open("images/pic4.jpg")
    img_arr = np.array(img)
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]])
    output_img = sobel_convolution(img_arr, sobel_x, sobel_y)
    output_img = Image.fromarray(output_img.astype(np.uint8))
    output_img.save("results/sobel_pic4.jpg")

if __name__ == "__main__":
    main()