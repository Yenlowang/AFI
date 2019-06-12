def plot_image(image_path, heigth, width):

  import matplotlib.pyplot as plt
  import matplotlib.image as mpimg
  img=mpimg.imread(image_path)
  imgplot = plt.imshow(img)

  plt.axis('off');

  fig = plt.gcf()
  fig.set_size_inches(heigth, width)