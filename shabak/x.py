from PIL import Image

im = Image.open("hint_-_vision.png")
rgb_im = im.convert('RGB')
rgb_im.save('rgbhint.jpg')