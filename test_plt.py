from PIL import Image,ImageDraw
#from matplotlib import pylot as plt
import matplotlib.pyplot as plt
import io
# img = Image.open('office1.jpg')
# #print img
# im = img.copy()
# draw = ImageDraw.Draw(im)
# draw.rectangle([(1,2),(100,200)],outline='blue')
#
# im.show()
fpr = [0,1,1,1,1,1]
tpr = [0,1,1,1,1,1]
fig = plt.figure()
plt.xlabel('frr',fontsize=14)
plt.ylabel('trr',fontsize=14)
plt.title('roc',fontsize=14)
plot = plt.plot(fpr,tpr,linewidth=2)
#plt.show()
canvas = fig.canvas
buf = io.BytesIO()
canvas.print_png(buf)
plt.savefig('e.png',format='png')
# #plt.savefig('a.jpg',format='jpg')
buf.seek(0)
# data = buf.getvalue()
# buf.close()
#
# buff = io.BytesIO()
# buff.write(data)
im = Image.open(buf)
im.save('d.png')
im.show()