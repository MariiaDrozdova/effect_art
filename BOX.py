from ctypes import *
import numpy as np
import random
import PIL
import matplotlib.pyplot as plt
verbose=True
class BOX():
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.h = h
        self.w = w
        
    
    def __copy__(self):
        res = BOX(self.x, self.y, self.w, self.h)
        return res
    
    def copy(self):
        res = BOX(self.x, self.y, self.w, self.h)
        return res
    
    def __repr__(self):
        return "[x : %f, y : %f, w : %f, h : %f]" %  (self.x, self.y, self.w, self.h)
    
    def __add__(self, box):
        res = BOX(
            self.x + box.x,
            self.y + box.y,
            self.w + box.w,#(self.x)*k_w,
            self.h + box.h,#(self.x)*k_h,
            #self.h + (box.y)*k_h,
            )
        return res

    def __sub__(self, box):
        res = BOX(
            self.x - box.x,
            self.y - box.y,
            self.w - box.w,# (-box.x)*k_w,
            self.h - box.h,# (-box.x)*k_h,
            #box.h + (self.y - 2*box.y)*k_h,

            )
        return res
        
    def __mul__(self, a):
        #print(a)
        res = BOX(a*(self.x), a*self.y, a*self.w, a*self.h)
        return res

    def dist_centers(self, a):
        delta_w = (self.x + self.w/2 - a.x  - a.w/2)
        delta_h = (self.y + self.h/2 - a.y  - a.h/2)
        return np.sqrt(delta_w**2+delta_h**2)
    
    def dist(self, a):
        def dist_two_points(a,b):
            delta_x = a[0] - b[0]
            delta_y = a[1] - b[1]
            return np.sqrt(delta_x**2+delta_y**2)
    
        left = self.x + self.w < a.x
        right = a.x + a.w < self.x
        bottom = self.y + self.h < a.y
        top = a.y + a.h < self.y
        if top and left:
            return dist_two_points([a.x, a.y + a.h], [self.x+self.w, self.y])
        elif left and bottom:
            return dist_two_points([a.x, a.y], [self.x+self.w, self.y+self.h])
        elif bottom and right:
            return dist_two_points([a.x+a.w, a.y], [self.x, self.y+self.h])
        elif right and top:
            return dist_two_points([a.x+a.w, a.y+a.h], [self.x, self.y])
        elif left:
            return a.x-self.x-self.w
        elif right:
            return self.x - a.x - a.w
        elif bottom:
            return a.y - self.y - self.h
        elif top:
            return self.y - a.y - a.h
        return 0
        
    def intersect(self, box):
        c1 = max(box.x, self.x)
        c2 = min(box.x+box.w, self.x+self.w)
        if c1 > c2:
            return -1
        d1 = max(box.y, self.y)
        d2 = min(box.y+box.h, self.y+self.h)
        if d1 > d2:
            return -1
        
        return (c2-c1)*(d2-d1)
    
    def getSurface(self):
        return self.h*self.w
    
    def IoU(self,box):
        return self.intersect(box)/(self.getSurface()+box.getSurface())
    
def cropImage(image, b, k = 1.0):
    
    W, H, = image.shape[0], image.shape[1]
    
    if (b.x >=0 and b.x + b.w >=0 and 
        b.x<W and b.x+b.w<W and 
        b.y >=0 and b.y + b.h >=0 and 
        b.y<H and b.y+b.h<H):
            return cropImage0(image, b)
    
    if len(image.shape) == 3:
        final_image= np.uint8(np.zeros((W+2*int(b.w), H+2*int(b.h), image.shape[2])))
    else: 
        final_image= np.uint8(np.zeros((W+2*int(b.w), H+2*int(b.h))))
    final_image[int(b.w):int(b.w)+W, int(b.h):int(b.h)+H] = image
    #print(np.max(final_image))
    #print(np.max(image))
    #plt.imshow(final_image)
    #plt.show()
    return final_image[int(b.w)+int(b.x):int(b.w)+int(b.x)+int(b.w), int(b.h)+int(b.y):int(b.h)+int(b.y)+int(b.h)]

def cropImage0(image, b, k = 1.0):
    return image[int(b.x):int(b.x)+int(b.w), int(b.y):int(b.y)+int(b.h)]

def cleanRepeatsBoxes(boxes):
    print('Cleaning overlapping boxes...')
    n = len(boxes)
    i, j = 0, 0
    while (i<len(boxes)):
        while (j <len(boxes)):
            if j > len(boxes) - 1:
                break
            if i > len(boxes) - 1:
                break
            #print(i)
            #print(j)
            #print((detections[i].bbox.IoU(detections[j].bbox)))
            if (i != j) and (boxes[i].IoU(boxes[j]) > 0.3):
                if boxes[i].getSurface() > boxes[j].getSurface():
                    del boxes[j]
                    continue
                else:
                    del boxes[i]
                    #j = j + 1 
                    continue
            else:
                 j = j + 1
        i = i + 1
        j = 0
    print('Number of boxes reduced from %d to %d' % (n, len(boxes)))
    print(len(boxes))
    return boxes
    
    
    
def cropImag2(image, b, k = 1.0):
    W, H, = image.shape[0], image.shape[1]
    x = b.x# - b.w/2
    y = b.y# - b.h/2
    w = b.w
    h = b.h
    x_paste = 0
    y_paste = 0
    
    if len(image.shape) == 3:
        final_image= np.uint8(np.zeros((int(b.w), int(b.h), image.shape[2])))
    else: 
        final_image= np.uint8(np.zeros((int(b.w), int(b.h))))
    W_new, H_new = W, H
    if x < 0:
        W_new = W + (-int(x))
        x = 0
        
        
    if y < 0:
        H_new = H + (-int(y))
        y = 0
        
    if len(image.shape) == 3:
        new_image = np.uint8(np.zeros((W_new, H_new, image.shape[2])))
    else: 
        new_image = np.uint8(np.zeros((W_new, H_new)))
        
    
    new_image[W_new-W:, H_new-H:] = image
    W, H = W_new, H_new
    image = new_image.copy()
    w_resize = int(W*k)
    h_resize = int(H*k)

    cur_image = PIL.Image.fromarray(image).resize((h_resize,w_resize),PIL.Image.ANTIALIAS)
    cur_image = np.asarray(cur_image)
        
    w_paste = int(min(w_resize+x_paste, w, w))
    h_paste = int(min(h_resize+y_paste, h, h))
    
    w_paste1 = int(min(cur_image.shape[0], w-x_paste))
    h_paste1 = int(min(cur_image.shape[1], h-y_paste))

    
    x1_paste = min(int(x*k)+(w_paste-x_paste), cur_image.shape[0])
    y1_paste = min(int(y*k)+(h_paste-y_paste), cur_image.shape[1])

    fin_x = x_paste+x1_paste - int(x*k)
    fin_y = y_paste+y1_paste - int(y*k)
    final_image[x_paste:fin_x, y_paste:fin_y]=255

    
    final_image[x_paste:fin_x, y_paste:fin_y]=np.uint8(cur_image[int(x*k):x1_paste, int(y*k):y1_paste])
    
    #if x + b.w > W:
    #    x = W - 1- int(b.w)
    #if y + b.h > H:
    #    y = H - 1 - int(b.h)
    return final_image

def linearInterpolation(box1, box2, delta=5):
    pass
    
def create_frames_linear(image_old,box1_old, box2_old, nb_interp, k=1.0, w=400, delta_x=None):
    if verbose:
        print('linear')
        print(box1_old)
        print(box2_old)
    image = image_old.copy()
    #assert box1.h == box2.h
    #assert box1.w == box2.w#int(max(box1.w, box2.w))
    #w_old = w
    #w = w*4
    
    h = int(w*k)#int(max(box2.h, box1.h))
    

    
    w_old = w
    h_old = h
    box1 = box1_old.copy()
    box2 = box2_old.copy()
    
    W, H = image.shape[0], image.shape[1]
    k_scale = 4
    w = w*k_scale
    W = W*k_scale
    H = H*k_scale
    h = int(k*w)
    print(image.shape)
    print(w)
    print(h)
    
    image_resized = resize_image(image, (int(W), int(H)))
    print(image_resized.shape)

    
    box1.x = k_scale*box1.x
    box2.x = k_scale*box2.x
    box1.y = k_scale*box1.y
    box2.y = k_scale*box2.y 
    box1.w = k_scale*box1.w
    box2.w = k_scale*box2.w
    box1.h = k_scale*box1.h
    box2.h = k_scale*box2.h 
    
    
    frames = []
    x, y = 0, 0
    
    delta = (box2 - box1)*(1.0/(nb_interp-1))
    if delta_x is not None:
        delta_x = k_scale*delta_x
        nb_interp = (int)(np.sqrt(np.abs(box2.x**2 + box2.y**2 - (box1.x**2 + box1.y**2)))/delta_x)
        delta = (box2 - box1)*(1.0/(nb_interp-1))
    cur_box = box1.copy()
    for i in range(nb_interp):
       
        cur_box = (box1 + delta*i).copy()
        w_resize=w
        h_resize=h
        cur_w = cur_box.w
        cur_h = cur_box.h


        img = cropImage(image_resized, cur_box)
        img=np.uint8(np.array(img))
        img = resize_image(img, (int(w_old), int(h_old)))
        
        #plt.imshow(img)
        #plt.show()
        frames.append(img)

    return frames
def linear_step(x):
    return x

def create_frames_cubic(image,box0, box1, box2, box3, nb_interp, delta_x = None, step_function=linear_step):
    #assert box1.h == box2.h
    #assert box1.w == box2.w
    
    h = int(w*k)
    frames = []
    x, y = 0, 0

    #print(delta)
    k = 0
    
    cur_box = box1#.copy()
    for i in range(nb_interp):
        cur_box = interpolate(box0, box1, box2, box3, step_function(i/(nb_interp)))#cur_box + delta
        w_resize=w
        h_resize=h
        cur_w = cur_box.x
        cur_h = cur_box.y
        if np.abs(cur_box.w * k - cur_box.h) >= 5:
            if cur_w * k < cur_h:
                w_resize = int(h * (cur_box.h/cur_box.w))
            else:
                h_resize = int(w * (cur_box.w/cur_box.h))
        x_paste = int((w - w_resize)*0.5)
        y_paste = int((h - h_resize)*0.5)
        img = cropImage(image, cur_box)
        img=np.uint8(np.array(img))
        img = resize(img, (int(w), int(h)))
        
        new_frame = np.asarray(img)
        frames.append(new_frame)
    return frames

def create_frames_fading(image, box0, fade='in', color=(255, 255, 255), nb_interp=20, w=400, k=1.0):

    frames = []
    h = int(w*k)
    if box0.w*k > box0.h:
        box0.h = box0.w*k
    else:
        box0.w = box0.h/k    
    
    #image_color = [[[*color] for i in range(int(image.shape[1]))] for j in range(int(image.shape[0]))]
    #image_color = np.array(image_color)
    box_faded = BOX(box0.x, box0.y, box0.w, box0.h)
    k_up = max(box0.h/h, box0.w/w)
    #image_cropped = np.asarray(PIL.Image.fromarray(cropImage(image, box0, k_up)))
    cur_box = box0
    k_up = min(box0.h/image.shape[1], box0.w/image.shape[0])
    box0.x = (int((image.shape[0]*k_up-box0.w)*0.5/k_up ))
    box0.y = (int((image.shape[1]*k_up-box0.h)*0.5/k_up ))
    im = np.array(PIL.Image.fromarray((cropImage(image, box0, k_up))).resize((h, w)))
    im_c = [[[*color] for i in range(int(im.shape[1]))] for j in range(int(im.shape[0]))]
    im_c = np.array(im_c)
    
    for i in range(nb_interp+1):
        new_frame = PIL.Image.new('RGB', (w, h), 'white')
        if fade == 'in':
            cur_image = im+ (im_c-im)*(i/nb_interp)
        else:
            cur_image = im_c - (im_c-im)*((i)/nb_interp)
        cur_image=np.uint8(cur_image)
        frames.append(cur_image)
    #'''
    return frames


def create_frames_fading_new(frame0, frame1, fade='in', nb_interp=20, w=400, k=1.0):
    frame1 = resize_image(frame1, (int(w), int(w*k)))
    frames = []
    assert frame0.shape[0] == frame1.shape[0] == w
    assert frame0.shape[1] == frame1.shape[1]
    for i in range(nb_interp+1):
        alpha = ((i)/nb_interp)
        cur_image = frame0 * (1 - alpha) + frame1 * alpha
        cur_image=np.uint8(cur_image)
        frames.append(cur_image)

    return frames

def create_frames_fading_patch(image, box0, fade='in', color=(255, 255, 255), nb_interp=20, w=400, k=1.0):
    if verbose:
        print('create_frames_fading_patch')
    frames = []
    h = int(w*k)
    W, H = image.shape[0], image.shape[1]
    
 
    k_up = min(h*1.0/box0.h, w*1.0/box0.w)
    new_im_W = W * k_up
    new_im_H = H * k_up
    
    #resize current image to fit the box
    cur_img = resize_image(image, (int(new_im_W), int(new_im_H)))    

    #recompute the position of the box for this resized image 
    #with an assumption that an image is in the center
    #box0.x = (int((W*k_up-w)*0.5))
    #box0.y = (int((H*k_up-h)*0.5))

    cur_box = BOX(box0.x*k_up, box0.y*k_up, w, h)

    im = cropImage(cur_img, cur_box, k)  
    
    im_c = [[[*color] for i in range(int(h))] for j in range(int(w))]
    im_c = np.array(im_c)
    
    for i in range(nb_interp+1):
        if fade == 'in':
            cur_image = im+ (im_c-im)*(i/nb_interp)
        else:
            cur_image = im_c - (im_c-im)*((i)/nb_interp)
        img=np.uint8(np.array(cur_image))
        img = resize_image(img, (int(w), int(h)))
        frames.append(img)
    #'''
    return frames

def putSquares(image_input,box1, box2, nb_interp, confidence1=1.0, confidence2=1.0):
    from skimage import io, draw
    image = image_input.copy()
    w = int(max(box1.w, box2.w))
    h = int(max(box2.h, box1.h))
    frames = []
    shape = image.shape
    x, y = 0, 0
    delta = (box2 - box1)*(1.0/(nb_interp-1))
    cur_box = box1
    for i in range(nb_interp):
                #cur_box = cur_box + delta
                cur_box = box1 + delta*i
                #print(cur_box)
                confidence = confidence1 + (confidence2 - confidence1)*i*(1.0/(nb_interp-1))
                boundingBox = [
                    [cur_box.x - cur_box.w//2, cur_box.y - cur_box.h//2],
                    [cur_box.x - cur_box.w//2,  cur_box.y + cur_box.h//2],
                    [cur_box.x + cur_box.w//2, cur_box.y + cur_box.h//2],
                    [cur_box.x + cur_box.w//2, cur_box.y - cur_box.h//2]
                ]
                # Wiggle it around to make a 3px border
                rr, cc = draw.polygon_perimeter([x[1] for x in boundingBox], [x[0] for x in boundingBox], shape= shape)
                rr2, cc2 = draw.polygon_perimeter([x[1] + 1 for x in boundingBox], [x[0] for x in boundingBox], shape= shape)
                rr3, cc3 = draw.polygon_perimeter([x[1] - 1 for x in boundingBox], [x[0] for x in boundingBox], shape= shape)
                rr4, cc4 = draw.polygon_perimeter([x[1] for x in boundingBox], [x[0] + 1 for x in boundingBox], shape= shape)
                rr5, cc5 = draw.polygon_perimeter([x[1] for x in boundingBox], [x[0] - 1 for x in boundingBox], shape= shape)
                boxColor = (int(255 * (1 - (confidence ** 2))), int(255 * (confidence ** 2)), 0)
                draw.set_color(image, (rr, cc), boxColor, alpha= 0.8)
                draw.set_color(image, (rr2, cc2), boxColor, alpha= 0.8)
                draw.set_color(image, (rr3, cc3), boxColor, alpha= 0.8)
                draw.set_color(image, (rr4, cc4), boxColor, alpha= 0.8)
                draw.set_color(image, (rr5, cc5), boxColor, alpha= 0.8)
    return image#PIL.Image.fromarray(image)

from PIL import ExifTags
from PIL import Image
from skimage.transform import resize

def resize_image(img, shape):
    w, h = shape[0], shape[1]
    img = resize(img, (int(w), int(h)))
    img = np.uint8(255*img)
    return img

def create_frames_zoom_failed(image, box0, box2, w=400, k=1.0, nb_interp=15, color=(255,255,255)):
    #takes less spatial memory but because of roundings looses precision
    frames = []
    h = int(w*k)
    
    W, H = image.shape[0], image.shape[1]
    
    k_up = min(h*1.0/box0.h, w*1.0/box0.w)
    new_im_W = W * k_up
    new_im_H = H * k_up
    
    k_up_final = min(h*1.0/box2.h, w*1.0/box2.w)
    final_im_W = W * k_up_final
    final_im_H = H * k_up_final
    
    #resize current image to fit the box
    cur_img = resize_image(image, (int(new_im_W), int(new_im_H)))
    
    plt.imshow(cur_img)
    plt.show()
    
    #recompute the position of the box for this resized image 
    #with an assumption that an image is in the center
    box0.x = (((W*k_up-w)*0.5))
    box0.y = (((H*k_up-h)*0.5))
    box2.x =  (box2.x*k_up_final)
    box2.y = (box2.y*k_up_final)

    cur_box = BOX(box0.x, box0.y, w, h)
    box1 = box0
    res = cropImage(cur_img, cur_box, k)

    cur_img = resize_image(image, (int(final_im_W), int(final_im_H)))
    cur_box = BOX(box2.x, box2.y, w, h)
    box1 = box0
    res = cropImage(cur_img, cur_box, k)
    frames = []
    
    k_new = min(box1.h/H, box1.w/image.shape[0])   

    center = np.array([box2.x+int(box2.w)//2, box2.y+int(box2.h)//2])
    center0 = np.array([box1.x+int(box1.w)//2, box1.y+int(box1.h)//2])

    xs = []
    ys = []
    for i in range(nb_interp):
        
            #center moves
            cur_w = box1.w + (box2.w - box1.w)*i/(nb_interp-1)
            cur_h = box1.h + (box2.h - box1.h)*i/(nb_interp-1)
            cur_im_center = (center-center0)*i/(nb_interp-1)
            
            #rescaling computed
            k_cur = k_up + (k_up_final - k_up)*i/(nb_interp-1)
            cur_im_W = W * k_cur
            cur_im_H = H * k_cur
            cur_img = resize_image(image, (int(cur_im_W), int(cur_im_H)))
            #cur_img = PIL.Image.fromarray(image).resize((int(cur_im_W), int(cur_im_H)),PIL.Image.ANTIALIAS)
            cur_img = np.array(cur_img)
            
            #poisiton of the box taking into accound the center and rescaling
            cur_box.x = box1.x + (box2.x - box1.x)*i/(nb_interp-1)
            cur_box.y = box1.y + (box2.y - box1.y)*i/(nb_interp-1)
            cur_box.w = w
            cur_box.h = h
            xs.append(cur_box.x)
            ys.append(cur_box.y)

            img = cropImage(cur_img, cur_box, k)
            img=np.uint8(np.array(img))
            

            frames.append(img)

    #'''
    cur_box = BOX(0, 0, 0, 0)
    plt.plot(range(len(xs)), xs)
    plt.show()
    plt.plot(range(len(ys)), ys)
    plt.show()

    frames.append(frames[-1])
    frames.append(frames[-1])
    frames.append(frames[-1])

    #'''
    return frames



def create_frames_zoom(image_old, box1_old, box2_old, w=400, k=1.0, nb_interp=15, delta_x=None):
    if verbose:
        print('zoom')
    frames = []
    h = int(w*k)
    

    
    w_old = w
    h_old = h
    box0 = box1_old.copy()
    box2 = box2_old.copy()
    
    W, H = image_old.shape[0], image_old.shape[1]
    k_scale = 2.0
    w = w*k_scale
    W = W*k_scale
    H = H*k_scale
    h = int(k*w)
    print(image_old.shape)
    print(w)
    print(h)
    image = resize_image(image_old, (int(W), int(H)))
    print(image.shape)

    
    box0.x = k_scale*box0.x
    box2.x = k_scale*box2.x
    box0.y = k_scale*box0.y
    box2.y = k_scale*box2.y 
    box0.w = k_scale*box0.w
    box2.w = k_scale*box2.w
    box0.h = k_scale*box0.h
    box2.h = k_scale*box2.h 
    
    
    if box0.w*k > box0.h:
        box0.h = box0.w*k
    else:
        box0.w = box0.h/k


    k_up = min(h/image.shape[1], w/image.shape[0])
    box0.x = (int((image.shape[0]*k_up-box0.w)*0.5/k_up ))
    box0.y = (int((image.shape[1]*k_up-box0.h)*0.5/k_up ))
    im = (cropImage(image, box0, k_up))
    cur_box = box0
    box1 = box0

    frames = []
    cur_box = box1.copy()
    k_new = min(box1.h/image.shape[1], box1.w/image.shape[0])

    center = np.array([box2.x+int(box2.w/k_new)//2, box2.y+int(box2.h/k_new)//2])
    center0 = np.array([box1.x+int(box1.w/k_up)//2, box1.y+int(box1.h/k_up)//2])


    if delta_x is not None:
        nb_interp = abs(box2.w - box1.w)/delta_x + 1
        nb_interp = int(nb_interp)
    

    #'''
    cur_box = BOX(0, 0, 0, 0)
    for i in range(nb_interp):
            cur_w = box1.w + (box2.w - box1.w)*i/(nb_interp-1)
            cur_h = box1.h + (box2.h - box1.h)*i/(nb_interp-1)
            cur_center = center0+ (center-center0)*i/(nb_interp-1)

            cur_box.x = cur_center[0] - int(cur_w)//2
            cur_box.y = cur_center[1] - int(cur_h)//2
            cur_box.w = cur_w
            cur_box.h = cur_h
            cur_image1 = cropImage(image, cur_box, k)
            cur_image1 = resize_image(cur_image1, (int(w_old), int(h_old)))
            #PIL.Image.fromarray(cropImage(image, cur_box, k)).resize((h, w), PIL.Image.ANTIALIAS)
            frames.append(np.array(cur_image1))
    frames.append(np.array(cur_image1))
    frames.append(np.array(cur_image1))
    frames.append(np.array(cur_image1))
    frames.append(np.array(cur_image1))
    #'''
    return frames

def make_path_clusters0(image, detections, cluster_boxes, cluster_labels,
                       in_points, out_points, in_detections, out_detections, lines, 
                       path,
                       proba_clusters=1, w = 300, k = 4.6,
                       min_delta_x=1, max_delta_x=2) :
    threshold = 0.8*(image.shape[0]**2 + image.shape[1]**2)**(0.5)
    threshold_zoom = 0.8*(image.shape[0]**2 + image.shape[1]**2)**(0.5)
    all_frames = []
    color = (0,0,0)
    latest_box = BOX(0, 0, image.shape[0], image.shape[1])
    max_square = 2000000
        
    cur_box = path[0].cluster_box
    
    im = (cropImage(image, cur_box))
    first_box = latest_box.copy()
    initial_box = latest_box.copy()
    initial_image = image.copy()
    frames = []
    frames = create_frames_fading(image, latest_box, fade='out', color=color, nb_interp=150, w = w, k = k)
    frames_initial = frames.copy()
    all_frames.extend(frames)
    
    #cur_delta_x
    if image.shape[0]*image.shape[1] > max_square:
        frames1 = create_frames_fading_patch(image, latest_box, fade='out', color=color, nb_interp=50, w = w, k = k)
        frames2 = create_frames_fading_patch(image, cur_box, fade='out', color=color, nb_interp=50, w = w, k = k)
        frames1 = frames1[::-1]
        frames = frames1 + frames2   
    else:
        cur_delta_x = random.random()*(-min_delta_x+max_delta_x)+min_delta_x
        frames = create_frames_zoom(image, latest_box, cur_box, nb_interp=150, w=w, k=k,delta_x = cur_delta_x)       
        frames2 = frames.copy()
    latest_box = cur_box.copy()
    all_frames.extend(frames)
    first=True
    
    for i in range(len(path)-1):#len(boxes)-1):
        if random.random() <= proba_clusters:
            cur_box = path[i+1].cluster_box.copy()
            if latest_box.dist_centers(cur_box) < threshold and not first and random.random() < 0.7:
                print('if latest_box.dist_centers(cur_box) < threshold and not first:')
                cur_delta_x = random.random()*(-min_delta_x+max_delta_x)+min_delta_x
                frames = create_frames_linear(image,latest_box, cur_box, nb_interp=150,  w = w, k = k, delta_x = cur_delta_x)       
                all_frames.extend(frames) 
            else:
                    if random.random() < 0.5:
                        frames1 = create_frames_fading_patch(image, latest_box, fade='out', color=color, nb_interp=50, w = w, k = k)
                        frames2 = create_frames_fading_patch(image, cur_box, fade='out', color=color, nb_interp=50, w = w, k = k)
                        frames1 = frames1[::-1]
                        frames = frames1 + frames2
                    else:
                        frame1 = (cropImage(image, cur_box)) 
                        frames = create_frames_fading_new(all_frames[-1], frame1, nb_interp=50, w=w, k=k)
                    all_frames.extend(frames)  
            first=False
            
            latest_box = cur_box.copy()
            cluster_box_cur = cur_box.copy()

        if True:
            print('lines')
            in_p_nb =  len(path[i+1].in_detections)
            out_p_nb = len(path[i+1].out_detections)
            cluster_flag = True
            if in_p_nb < 3:
                print('if')
                detections_in_clusters = path[i+1].in_detections + path[i+1].out_detections
                k_det = min(random.choice([0,2]), len(detections_in_clusters))
                if k_det > 0:
                    print('>0')
                    weights_for_cur_cluster_detections = [cur_detection.score for cur_detection in detections_in_clusters]
                    weights_for_cur_cluster_detections = np.array(weights_for_cur_cluster_detections)
                    weights_for_cur_cluster_detections = weights_for_cur_cluster_detections/np.sum(weights_for_cur_cluster_detections)
                    detections_in_cluster_to_show = random.choices(detections_in_clusters,
                                                                   weights=weights_for_cur_cluster_detections, k=k_det)
                    for detection in detections_in_cluster_to_show:
                        print('for')
                        #if detection.final_box.getSurface() < threshold*threshold:
                        #    continue
                        cur_box = detection.final_box.copy()
                        if latest_box.dist_centers(cur_box) < threshold*0.3 and not cluster_flag:
                            cur_delta_x = random.random()*(-min_delta_x+max_delta_x)+min_delta_x
                            frames = create_frames_linear(initial_image,latest_box, cur_box, nb_interp=150,  w = w, k = k, delta_x = cur_delta_x) 
                            all_frames.extend(frames) 
                        else:
                            if random.random() < 0.5:
                                frames1 = create_frames_fading_patch(image, latest_box, fade='out', color=color, nb_interp=50, w = w, k = k)
                                frames2 = create_frames_fading_patch(image, cur_box, fade='out', color=color, nb_interp=50, w = w, k = k)
                                frames1 = frames1[::-1]
                                frames = frames1 + frames2
                            else:
                                frame1 = (cropImage(image, cur_box)) 
                                frames = create_frames_fading_new(all_frames[-1], frame1, nb_interp=100, w=w, k=k) 
                        cluster_flag = False
                        latest_box = cur_box.copy()
                        print(cur_box)
                        plt.imshow((cropImage(image, cur_box)) )
                        plt.show()
                    #frames = create_frames_linear(image,latest_box, cluster_box_cur, nb_interp=150,  w = w, k = k) 
                    #all_frames.extend(frames) 
            else:
                print('else')
                inline_detections_sorted = sorted(path[i+1].in_detections, key = lambda a : a.final_box.x)
                for detection in [inline_detections_sorted[0], inline_detections_sorted[-1]]:
                    #if detection.final_box.getSurface() < threshold*threshold:
                    #    continue
                    cur_box = detection.final_box.copy()
                    cur_delta_x = random.random()*(-min_delta_x+max_delta_x)+min_delta_x
                    frames = create_frames_linear(image,latest_box, cur_box, nb_interp=150,  w = w, k = k, delta_x=cur_delta_x) 
                    all_frames.extend(frames)
                    latest_box = cur_box.copy()
                #frames = create_frames_linear(image,latest_box, cluster_box_cur, nb_interp=150,  w = w, k = k) 
                #all_frames.extend(frames) 
                #latest_box = cluster_box_cur.copy()
                    
        '''print(i)
        if detections[i].final_box.dist(detections[i+1].final_box) > max(image.shape)/8:
            color=(0, 0, 0)
            print(detections[i].final_box)
            frames = create_frames_fading(image, detections[i].final_box, fade='in', color=color, nb_interp=5)
            all_frames.extend(frames)
            
            frames = create_frames_fading(image, detections[i+1].final_box, fade='out', color=color, nb_interp=5)
            all_frames.extend(frames)
            all_frames.extend([frames[-1]]*5)
        else:
            frames = create_frames_linear(image,detections[i].final_box, detections[i+1].final_box, nb_interp=50, delta_x=None)
            all_frames.extend(frames)#'''
    print(first_box)
    print(latest_box)
    #cur_delta_x
    if image.shape[0]*image.shape[1] > max_square:
        frames1 = create_frames_fading_patch(initial_image, latest_box, fade='out', color=color, nb_interp=50, w = w, k = k)
        frames2 = create_frames_fading(initial_image, initial_box, fade='out', color=color, nb_interp=50, w = w, k = k)
        frames1 = frames1[::-1]
        frames = frames1 + frames2   
    else:
        cur_delta_x = random.random()*(-min_delta_x+max_delta_x)+min_delta_x
        frames = create_frames_zoom(initial_image, initial_box, latest_box, nb_interp=150, w=w, k=k, delta_x=cur_delta_x)
        frames = frames[::-1]
    all_frames.extend(frames)
    frames = []

    #frames = create_frames_fading(initial_image, first_box, fade='out', color=color, nb_interp=30, w = w, k = k)       
    frames = frames_initial[::-1]
    all_frames.extend(frames)
    return all_frames