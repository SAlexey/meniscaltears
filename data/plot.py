import matplotlib.pyplot as plt
import numpy as np

normalized_image = np.load('/Users/davidluedke/Desktop/10014409_0_cam.npy')

print(normalized_image.shape)

plt.imshow(normalized_image[32])
plt.show()