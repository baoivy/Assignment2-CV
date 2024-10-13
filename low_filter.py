from PIL import Image
import numpy as np

def gaussin_kernel(kernel_size, sigma):
    ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))

    return kernel / np.sum(kernel) 

def gaussian_convolution(image, kernel):
    if len(image.shape) == 3:
        img_height, img_width, _ = image.shape
    elif len(image.shape) == 2:
        img_height, img_width = image.shape 
    kernel_size = kernel.shape[0]
    
    output_img = np.zeros(image.shape)
    if len(image.shape) == 3:
        padded_img = np.pad(image, ((kernel_size // 2, kernel_size // 2), (kernel_size // 2, kernel_size // 2), (0,0)), mode='constant')
    elif len(image.shape) == 2:
        padded_img = np.pad(image, ((kernel_size // 2, kernel_size // 2), (kernel_size // 2, kernel_size // 2)), mode='constant')
    
    kernel_flipped = np.flipud(np.fliplr(kernel))
    for i in range(img_height):
        for j in range(img_width):
            if len(image.shape) == 3:
                for k in range(3):
                    output_img[i, j, k] = np.sum(padded_img[i : i + kernel_size, j : j + kernel_size, k] * kernel_flipped)
            else:
                output_img[i, j] = np.sum(padded_img[i : i + kernel_size, j : j + kernel_size] * kernel_flipped)

    return output_img

def main():
    img = Image.open("images/cat_224x224.jpg")
    img_arr = np.array(img)
    kernel = gaussin_kernel(10, 15)
    #kernel = np.array([[1/25]*5]*5, dtype=np.float32)
    output_img = gaussian_convolution(img_arr, kernel)
    output_img = Image.fromarray(output_img.astype(np.uint8))
    output_img.save("results/gaussian_15_cat_224x224.jpg")

if __name__ == "__main__":
    main()