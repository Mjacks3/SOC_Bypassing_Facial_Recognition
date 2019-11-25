
import base64

pic = ""
with open("test.txt", 'r') as f:
        pic =  f.read()
pic = pic.replace("data:image/jpeg;base64,","")

imgdata = base64.b64decode(pic)
filename = 'some_image.jpeg'  # I assume you have a way of picking unique filenames
with open(filename, 'wb') as f:
    f.write(imgdata)

with open('some_image.png', 'wb') as f:
    f.write(imgdata)

with open('some_image.pgm', 'wb') as f:
    f.write(imgdata)
