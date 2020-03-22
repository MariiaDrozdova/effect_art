       
import numpy as np
import random
import matplotlib.pyplot as plt
from BOX import *
from sklearn.cluster import DBSCAN, AgglomerativeClustering, MeanShift
from scipy.spatial import ConvexHull

from sklearn.linear_model import RANSACRegressor
from sklearn.datasets import make_regression
from sklearn import linear_model, datasets

from scipy import signal
from scipy import misc

import PIL
        
def interpolate(p0, p1, p2, p3, t):
    a = p0*(-0.5)+p1*1.5+p2*(-1.5)+p3*0.5
    b = p0+p1*(-2.5)+p2*2.0+p3*(-0.5)
    c = p0*(-0.5)+p2*0.5
    d = p1
    return d + c*t + b*t*t + a*t*t*t

    
class DETECTION():
    def __init__(self, nclass, confidence, bbox, depth=-1):
        self.nclass = nclass
        self.confidence = confidence
        self.bbox = BOX(bbox[1]-bbox[3]/2, bbox[0]-bbox[2]/2, int(bbox[3]), int(bbox[2]))
        
        self.depth = depth
        #finally not used! 
        priority_one = ['person']
        priority_two = ['sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'cat', 'dog', 'horse']
        priority_three = ['bottle', 'knife', 'banana', 'apple', 'orange', 'clock', 'vase']
        if self.nclass in priority_one:
            self.score = 0.8
        elif self.nclass in priority_two:
            self.score = 0.6
        elif self.nclass in priority_one:
            self.score = 0.4
        else:
            self.score = 0.3
        self.area = -1
        self.final_box = BOX(bbox[1]-bbox[3]/2, bbox[0]-bbox[2]/2, int(bbox[3]), int(bbox[2]))
        
    def __repr__(self):
        return "nclass : %s,\nconfidence : %f,\nbbox :%s,\ndepth : %f\n\n" %  (str(self.nclass), self.confidence, self.bbox, self.depth)
    def __str__(self):
        return "nclass : %s,\nconfidence : %f,\nbbox :%s,\ndepth : %f\n\n" %  (str(self.nclass), self.confidence, self.bbox, self.depth)
    
def coordinatesToDetections(coords_x, coords_y, size, data):
    n = len(coords_x)
    vgg_indexes, vgg_table, hog_indexes, hog_table, lab_indexes, lab_table = data
    detections = []
    for i in range(n):
        x = coords_x[i]
        y = coords_y[i]
        w = size
        h = size
        bbox = [y+h//2, x+w//2, h, w]
        detections.append(DETECTION('COI', 1.0, bbox))
    return detections

def resToDetections(res, depthMap):
    n = len(res)
    detections = []
    for i in range(n):        
        
        detection = DETECTION(res[i][0], res[i][1], res[i][2])
        depth = np.mean(cropImage(depthMap, detection.bbox))
        detection.depth = depth
        detections.append(detection)
    return detections


def scaleDepth(detections):
    max_detection = (max(detections, key=lambda p : p.depth)).depth
    min_detection = (min(detections, key=lambda p : p.depth)).depth
    for i in range(len(detections)):
        detection = detections[i]
        if min_detection != max_detection:
            detection.depth = (detection.depth-min_detection)/(max_detection-min_detection)
        
def scaleScore(detections):
    for i in range(len(detections)):
        detection = detections[i]
        detection.score = (detection.depth*detection.confidence)
    detections.sort(key = lambda x : x.score)
    
def cleanRepeats(detections):
    print('Cleaning overlapping boxes...')
    n = len(detections)
    i, j = 0, 0
    while (i<len(detections)):
        while (j <len(detections)):
            #print(i)
            #print(j)
            if j > len(detections) - 1:
                break
            if i > len(detections) - 1:
                break
            #print((detections[i].bbox.IoU(detections[j].bbox)))
            if (i != j) and (detections[i].bbox.IoU(detections[j].bbox) > 0.03):
                if detections[i].area > detections[j].area:
                    del detections[j]
                    continue
                else:
                    del detections[i]
                    #j = j + 1 
                    continue
            else:
                 j = j + 1
        i = i + 1
        j = 0
    print('Number of boxes reduced from %d to %d' % (n, len(detections)))
    print(len(detections))
    return detections
    
def scaleAreaCLusters(detections, shape):
    areas = []

    for i in range(len(detections)):
        detection = detections[i]
        detection.area = (detection.bbox.w*detection.bbox.h)
        areas.append(detection.area)
        detection.h_mean = 0
        detection.w_mean = 0
        
    areas = np.array(areas).reshape(-1,1)
    clustering = DBSCAN(eps=0.002*max(shape)**2, min_samples=2).fit(areas)
    labels = clustering.labels_
    
    unique_labels = np.unique(labels)
    #for i in range()
    d = {unique_labels[i]:i for i in range(len(unique_labels))}
    
    
    h_means = [0]*len(unique_labels)
    w_means = [0]*len(unique_labels)
    cardinality = [np.sum(labels==i) for i in unique_labels]
    
    
    
    for i in range(len(detections)):
        detection = detections[i]
        detection.label = labels[i]
        index_in_unique = d[detections[i].label]
        #h_means[detection.label] += detection.bbox.h
        #w_means[detection.label] += detection.bbox.w
        h_means[index_in_unique] = max(detection.bbox.h, h_means[index_in_unique])
        w_means[index_in_unique] = max(detection.bbox.w, w_means[index_in_unique])
        
        
    #for i in range(len(unique_labels)):
    #    h_means[i] /= cardinality[i]
    #   w_means[i] /= cardinality[i]  

        
    for i in range(len(detections)):
        index_in_unique = d[detections[i].label]
        detections[i].final_box = BOX(detections[i].bbox.x, detections[i].bbox.y, w_means[index_in_unique], h_means[index_in_unique])
        #print(detections[i].final_box)
    return detections, h_means, w_means, labels
    
def scaleAreaCLustersWithCustomShape(detections, shape, k, w_min=-1):
    print('Scaling clusters according to the scores forming custom.shape...')
    areas = []
    points = []

            
    
            
    for i in range(len(detections)):
        detection = detections[i]
        detection.area = (detection.bbox.w*detection.bbox.h)
        areas.append(detection.area)
        detection.h_mean = 0
        detection.w_mean = 0
        #points.append([detections[i].bbox.x, detections[i].bbox.y, step*detections[i].depth//step])
        
    areas = np.array(areas).reshape(-1,1)
    
    
    labels = range(len(detections))
    unique_labels = np.unique(labels)
    #for i in range()
    d = {unique_labels[i]:i for i in range(len(unique_labels))}
    
    
    h_means = [0]*len(unique_labels)
    w_means = [0]*len(unique_labels)
    cardinality = [np.sum(labels==i) for i in unique_labels]
    
    
    
    for i in range(len(detections)):
        detection = detections[i]
        detection.label = labels[i]
        index_in_unique = d[detections[i].label]
        #h_means[detection.label] += detection.bbox.h
        #w_means[detection.label] += detection.bbox.w
        h_means[index_in_unique] = max(detection.bbox.h, h_means[index_in_unique])
        w_means[index_in_unique] = max(detection.bbox.w, w_means[index_in_unique])
        if w_means[index_in_unique] < w_min:
            print(w_min - w_means[index_in_unique])
            detections[i].bbox.x = max(detections[i].bbox.x - (w_min - w_means[index_in_unique])//2, 0)
            w_means[index_in_unique] = w_min
        print(w_means[index_in_unique])
        if (h_means[index_in_unique] < w_means[index_in_unique]*k):
            detections[i].bbox.y = max(detections[i].bbox.y - (w_means[index_in_unique]*k - h_means[index_in_unique])//2, 0)
            h_means[index_in_unique] = w_means[index_in_unique]*k
            
        else:
            detections[i].bbox.y = max(detections[i].bbox.y - (w_means[index_in_unique] - h_means[index_in_unique]/k)//2, 0)
            w_means[index_in_unique] = h_means[index_in_unique]/k
        detections[i].bbox.w = w_means[index_in_unique]
        detections[i].bbox.h = h_means[index_in_unique]
            
        
        
    #for i in range(len(unique_labels)):
    #    h_means[i] /= cardinality[i]
    #   w_means[i] /= cardinality[i]  

        
    for i in range(len(detections)):
        index_in_unique = d[detections[i].label]

        box_w = w_means[index_in_unique]
        box_h = h_means[index_in_unique]
        print(detections[i].bbox.x)
        box_x = detections[i].bbox.x 
        box_y = detections[i].bbox.y
        print(box_x)

        if box_x + box_w > shape[0]:
            box_x = shape[0] - box_w 
        if box_y + box_h > shape[1]:
            box_y = shape[1] - box_h 
        detections[i].final_box = BOX(box_x, box_y, box_w, box_h)
        #print(detections[i].final_box)
    
    
    return detections, h_means, w_means, labels    

def order_show(detections, shape, k, image=None, min_zoom=0):
    w_min = 0
    h_min = 0
    if min_zoom:
        w_min = shape[0]/min_zoom
        h_min = k*w_min
    points = [] 
    box_wh = []
    plt.figure(figsize=(20,20))
    min_depth=+np.inf
    max_depth=-np.inf
    for i in range(len(detections)):
        if min_depth > detections[i].depth:
            min_depth = detections[i].depth
        if max_depth < detections[i].depth:
            max_depth = detections[i].depth

    step = max(((max_depth - min_depth)*0.1), 0.001)
    scores = []
    for i in range(len(detections)):    
        points.append(np.array([detections[i].bbox.x, detections[i].bbox.y, 0, int(max(shape)*0.1)*int(detections[i].depth/step)]))
        box_wh.append(np.array([detections[i].bbox.w, detections[i].bbox.h]))
        scores.append(detections[i].score)
        
    points = np.array(points)
    box_wh = np.array(box_wh)
    
    clustering = MeanShift(max(shape)//6).fit(points)
    labels_clusters = clustering.labels_
    boxes = []
    mean_scores = []
    scores = np.array(scores)
    for label in np.unique(labels_clusters):
        cur_points = (points[labels_clusters==label])
        cur_box_wh = (box_wh[labels_clusters==label])
        cur_score = (scores[labels_clusters==label])
        x_min = np.min(cur_points[:, 0])
        x_max = np.max(cur_points[:, 0]+cur_box_wh[:, 0])

        y_min = np.min(cur_points[:, 1])
        y_max = np.max(cur_points[:, 1]+cur_box_wh[:, 1])

        h = max(y_max - y_min,w_min)
        w = max(x_max - x_min,h_min)

        #im = (cropImage(image, box))
        #plt.imshow(im)
        #plt.show()  
        if h < w * k:
            y_min = max(y_min - (w*k - h), 0)
            h = w * k
        else:
            x_min = max(x_min - (h/k - w), 0)
            w = h/k
        if x_min + w > shape[0]:
            x_min = shape[0] - w
        if y_min + h > shape[1]:
            y_min = shape[1] - h             
        box = BOX(x_min, y_min, w, h)
        boxes.append(box)
        mean_scores.append(cur_score)
        #im = (cropImage(image, box))
        #plt.imshow(im)
        #plt.show() 
    return labels_clusters, boxes#, mean_scores
        
        

def putDetections(image_input,detections, final_box=False, flag_detections=True):
    from skimage import io, draw
    image = image_input.copy()
    frames = []
    shape = image.shape
    x, y = 0, 0
    nb_detections = len(detections)
    for i in range(nb_detections):
        if flag_detections:
            if final_box:
                cur_box = detections[i].final_box
            else:
                cur_box = detections[i].bbox
        else:
            cur_box = detections[i]
            

        confidence = 1.0
        boundingBox = [
                    
                    [cur_box.y, cur_box.x],
                    [cur_box.y + cur_box.h, cur_box.x],
                    [cur_box.y + cur_box.h, cur_box.x + cur_box.w],
                    [cur_box.y, cur_box.x + cur_box.w]
                ]
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



def detectionsOnlyTrue(detections, indexes):
    new_detections = []
    for num, i in enumerate(indexes):
        if i:
            new_detections.append(detections[num])
    return new_detections
            

def lines_in_clusters(image, detections, cluster_labels, inverse=False):
    plt.imshow(image)
    points = []  
    wh = []
  
    for i in range(len(detections)):    
        #INVERSION X Y
        if not inverse:
            points.append(np.array([detections[i].bbox.x, detections[i].bbox.y]))
            wh.append([detections[i].bbox.h, detections[i].bbox.w])
        else:
            points.append(np.array([detections[i].bbox.y, detections[i].bbox.x]))
            wh.append([detections[i].bbox.w, detections[i].bbox.h])            
    points = np.array(points)
    wh = np.array(wh)

    labels_clusters = cluster_labels

    image_l = np.asarray(PIL.Image.fromarray(image).convert('L'))
    scharr = np.array([[ -3-3j, 0-10j,  +3 -3j],
                       [-10+0j, 0+ 0j, +10 +0j],
                       [ -3+3j, 0+10j,  +3 +3j]]) # Gx + j*Gy
    grad = signal.convolve2d(image_l, scharr, boundary='symm', mode='same')
    im = (1 - np.absolute(grad)/np.max(np.absolute(grad)))*255
    #c, path = minCost(im[1358:1959, 1586:2600], 600, 1000)
    #plt.imshow(image[1358:1959, 1586:2600])
    #plt.plot(path[:, 1], path[:, 0])

    in_points = []
    in_detections = []
    out_points = []
    out_detections = []
    lines = []
    for label in np.unique(labels_clusters):
            in_points.append([])
            lines.append([])
            in_detections.append([])
            cur_points = (points[labels_clusters==label])
            cur_wh = (wh[labels_clusters==label])
            
            cur_points[:,0],cur_points[:,1]= cur_points[:,0]+cur_wh[:, 0]//2, cur_points[:,1]+cur_wh[:, 1]//2
            if len(cur_points) < 3:
                out_points.append(cur_points)
                cur_detections = detectionsOnlyTrue(detections, labels_clusters==label)
                out_detections.append(cur_detections)
                continue
            try:
                ransac = linear_model.RANSACRegressor()
                ransac.fit(cur_points[:,0].reshape(-1, 1), cur_points[:,1].reshape(-1, 1))
            except:
                out_points.append(np.array(cur_points))
                cur_detections = detectionsOnlyTrue(detections, labels_clusters==label)
                out_detections.append(cur_detections)
                continue

            line_X = np.array(range(int(cur_points[:,0].min()), int(cur_points[:,0].max()))).reshape(-1, 1)
            line_y_ransac = ransac.predict(line_X)
            
            line_X = line_X.reshape(-1)
            line_y_ransac = line_y_ransac.reshape(-1)
            line_X = np.array(line_X, dtype=np.float32)
            line_y_ransac = np.array(line_y_ransac, dtype=np.float32)

            inside_indexes = np.min( (np.vstack([(line_y_ransac < image.shape[1] )*1, (line_y_ransac > 0.0 )*1])), axis=0)

            plt.plot(line_X[inside_indexes==1], line_y_ransac[inside_indexes==1], linewidth=5,
                 label='RANSAC regressor')

            inlier_mask = ransac.inlier_mask_
            outlier_mask = np.logical_not(inlier_mask)

            inliers = cur_points[inlier_mask]
            outliers = cur_points[outlier_mask]
            inliers = sorted(inliers, key=lambda a : a[0])
            
            cur_detections = detectionsOnlyTrue(detections, labels_clusters==label)
            detections_inliers = detectionsOnlyTrue(cur_detections, inlier_mask)
            detections_outliers = detectionsOnlyTrue(cur_detections, outlier_mask)
            
            in_detections[-1] = sorted(detections_inliers, key = lambda a : a.bbox.x)
            in_points[-1] = np.array(inliers)
            out_points.append(np.array(outliers))
            out_detections.append(detections_outliers)
            lines[-1] = np.array(line_y_ransac)
    return in_points, out_points, in_detections, out_detections, lines


class paintingClusters():
    def __init__(self, cluster_box, in_points, out_points, in_detections, out_detections, lines):
        self.cluster_box = cluster_box
        self.in_points = in_points
        self.out_points = out_points
        self.in_detections = in_detections
        self.out_detections = out_detections
        self.lines = lines
        
def shortestPath(paintingClusterBoxes):
    n = len(paintingClusterBoxes)
    
    x_left = np.inf
    i_left = -1
    
    x_right = 0
    i_right = -1
    for i, box0 in enumerate(paintingClusterBoxes):
        cur_x0 = box0.cluster_box.x
        cur_x1 = box0.cluster_box.x + box0.cluster_box.w
        cur_y0 = box0.cluster_box.y
        cur_y1 = box0.cluster_box.y + box0.cluster_box.h
        if cur_x0 <x_left:
            x_left = cur_x0
            i_left = i
        if cur_x1 > x_right:
            i_right = i
    
    paintingClusterBoxes0 = paintingClusterBoxes[i_left]
    
    for paintingClusterBox in paintingClusterBoxes:
        paintingClusterBox.d = paintingClusterBox.cluster_box.dist(paintingClusterBoxes0.cluster_box)
        paintingClusterBox.next = paintingClusterBoxes0
        paintingClusterBox.next_d = 100000000000
        paintingClusterBox.visited = 0
        paintingClusterBox.next_n = -1
    
    path = [paintingClusterBoxes0]
    detection = paintingClusterBoxes0
    visited = [0]*len(paintingClusterBoxes)
    paintingClusterBoxes[i_left].visited = 1
    k = 0
    while (k < len(paintingClusterBoxes)-1):
        print(k)

        for j, Cluster_Box in enumerate(paintingClusterBoxes):
            if detection == Cluster_Box:
                continue
            cur_dist = Cluster_Box.cluster_box.dist(detection.cluster_box)+detection.d
            if (cur_dist <= detection.next_d) and (Cluster_Box.visited == 0):
                #print("if")
                detection.next_d = cur_dist
                detection.next = Cluster_Box
                detection.next_n = j
        if detection.next.visited == 0:
            path.append(detection.next)
            detection.visited = 1
            detection = detection.next
            k = k + 1
    return path
        
    table = np.zeros((n,n))
    for i, box0 in enumerate(paintingClusterBoxes):
        for j, box1 in enumerate(paintingClusterBoxes):
            table[i, j] = box0.cluster_box.dist_centers(box1.cluster_box)
    print(table)