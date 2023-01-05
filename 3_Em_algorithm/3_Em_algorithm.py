import numpy as np
import matplotlib.pyplot as plt
import math


# load all image from a folder
def load_images(img_folder):
    img = np.load(img_folder)
    return img


def get_img_average(images):
    avg_img = np.zeros(images[0].shape)
    n_images = images.shape[0]
    for i in range(n_images):
        avg_img += images[i]
    avg_img /= n_images
    return avg_img


def shape_mle(avg_image, new_etas):
    pix_avg_eta = [0, 0]
    etas = np.copy(new_etas)
    img_shape = np.random.randint(2, size=avg_image.shape)

    n_1 = np.sum(img_shape)
    n_0 = np.size(img_shape) - n_1
    epsilon = 1000
    step_id = 1
    while abs(epsilon) > 0.01:
        # maybe pick better param than average pixel and avg_image[i,j] ??
        if step_id % 2 == 1:
            eta_image = avg_image*img_shape*etas[1] + avg_image*(1-img_shape)*etas[0]
            average_pixel = np.sum(eta_image) / np.size(eta_image)

            plt.imshow(eta_image, cmap='gray')
            plt.title("Step " + str(step_id) + ": with etas " + str(etas))
            plt.show()

            for i in range(avg_image.shape[0]):
                for j in range(avg_image.shape[1]):
                    img_shape[i, j] = 1 if (eta_image[i, j] >= average_pixel) else 0

            n_1 = np.sum(img_shape)
            n_0 = np.size(img_shape) - n_1

            step_id += 1

        elif step_id % 2 == 0:
            pix_avg_eta[0] = 1/n_0*np.sum(avg_image*(1-img_shape))
            pix_avg_eta[1] = 1/n_1*np.sum(avg_image*img_shape)

            new_etas[0] = math.log(pix_avg_eta[0]/(1-pix_avg_eta[0]))
            new_etas[1] = math.log(pix_avg_eta[1]/(1-pix_avg_eta[1]))

            epsilon = np.sum(new_etas) - np.sum(etas)
            etas = np.copy(new_etas)
            step_id += 1

    return img_shape, etas

# --------------------------------------------------------------------------------------------------------


images_data = load_images('em_data/images0.npy')
avg_image = get_img_average(images_data)

#plt.imshow(avg_image, cmap='gray')
#plt.title("avg_image")
#plt.show()

etas_init = [0, 1]
img_shape, etas = shape_mle(avg_image, etas_init)

print(etas)
plt.imshow(img_shape, cmap='gray')
plt.title("Final shape")
plt.show()
