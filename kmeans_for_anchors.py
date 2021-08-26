import numpy as np
import xml.etree.ElementTree as ET
import glob
import random
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams

def cas_iou(box,cluster):
    x = np.minimum(cluster[:,0],box[0])
    y = np.minimum(cluster[:,1],box[1])

    intersection = x * y
    area1 = box[0] * box[1]

    area2 = cluster[:,0] * cluster[:,1]
    iou = intersection / (area1 + area2 -intersection)

    return iou

def avg_iou(box,cluster):
    return np.mean([np.max(cas_iou(box[i],cluster)) for i in range(box.shape[0])])


def kmeans(box,k):
    # 取出一共有多少框
    row = box.shape[0]
    
    # 每个框各个点的位置
    distance = np.empty((row,k))
    
    # 最后的聚类位置
    last_clu = np.zeros((row,))

    np.random.seed()

    # 随机选k个当聚类中心
    cluster = box[np.random.choice(row,k,replace = False)]
    # cluster = random.sample(row, k)
    while True:
        # 计算每一行距离五个点的iou情况。
        for i in range(row):
            distance[i] = 1 - cas_iou(box[i],cluster)
        
        # 取出最小点
        near = np.argmin(distance,axis=1)

        if (last_clu == near).all():
            break
        
        # 求每一个类的中位点
        for j in range(k):
            cluster[j] = np.median(
                box[near == j],axis=0)

        last_clu = near

    return cluster

def load_data(path):
    data = []
    # 对于每一个xml都寻找box
    for xml_file in glob.glob('{}/*xml'.format(path)):
        tree = ET.parse(xml_file)
        height = int(tree.findtext('./size/height'))
        width = int(tree.findtext('./size/width'))
        # 对于每一个目标都获得它的宽高
        for obj in tree.iter('object'):
            xmin = int(float(obj.findtext('bndbox/xmin'))) / width
            ymin = int(float(obj.findtext('bndbox/ymin'))) / height
            xmax = int(float(obj.findtext('bndbox/xmax'))) / width
            ymax = int(float(obj.findtext('bndbox/ymax'))) / height

            xmin = np.float64(xmin)
            ymin = np.float64(ymin)
            xmax = np.float64(xmax)
            ymax = np.float64(ymax)
            # 得到宽高
            data.append([xmax-xmin,ymax-ymin])
    return np.array(data)


if __name__ == '__main__':
    path = r'./VOCdevkit/VOC2007/Annotations'
    data = load_data(path)
    SIZE = 416
    b = 11
    c = b
    result = {}
    for i in range(1,b):
        out = kmeans(data,i)
        out = out[np.argsort(out[:,0])]
        result[i] = avg_iou(data,out)

    f = open("yolo_anchors.txt", 'w')
    a = out*SIZE
    row = np.shape(a)[0]
    for i in range(row):
        if i == 0:
            x_y = "%d,%d" % (a[i][0], a[i][1])
        else:
            x_y = ", %d,%d" % (a[i][0], a[i][1])
        f.write(x_y)
    f.close()

    print('acc:{:.2f}%'.format(avg_iou(data,out) * 100))
    print(a)

    keys = list(result.keys())
    values = list(result.values())
    
    x = keys
    y = values

    # plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.tick_params(labelsize=15)
    
    font1 = {'family' : 'SimHei',
    'weight' : 'normal',
    'size'   : 15,
    }
    font2 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 15,
    }

    plt.xlabel("k", font2)#x轴上的名字
    plt.ylabel("Avg_IOU", font2)#y轴上的名字
    plt.plot(x,y,label='Avg_IOU')
    plt.plot(x, y, '.r')
    plt.xlim(1, c) 
    plt.xticks(range(1, c, 1))  # 设置X轴坐标点的值，为[0， 22]之间的以2为差值的等差数组   
    plt.legend(prop = font2)
    plt.savefig('C:/Users/17333/Desktop/56.jpg', dpi=300)