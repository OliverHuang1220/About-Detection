import xml.etree.ElementTree as ET
import glob as gb
import os
import shutil
import cv2
def save_region(img,obj,obj_area,xml_path,idx):
    image_path = xml_path[0:-4]
    if not os.path.isdir(image_path):
        os.mkdir(image_path)
    img_crop = img[obj_area[1]:obj_area[3],obj_area[0]:obj_area[2]].copy()
    if img_crop.shape[0] !=0 and img_crop.shape[1] != 0:
        cv2.imwrite('{}/{}_{}.jpg'.format(image_path,idx,obj), img_crop)

def crop_object_region(data_path):
    xml_path_list = gb.glob(data_path + '/*.xml')
    print('xml:', len(xml_path_list))
    jpg_list = gb.glob(data_path + '/*.jpg')
    print('jpg:', len(jpg_list))
    if len(xml_path_list) != len(jpg_list):
        print('Missing xml or image file')
    for xml_path in xml_path_list:
        tree = ET.parse(xml_path)
        if tree.find('size'):
            size = tree.find('size')
        else:
            size = tree.find('szie')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        if os.path.exists(xml_path[0:-3]+'jpg'):
            img = cv2.imread(xml_path[0:-3]+'jpg')
            width_t = img.shape[1]
            height_t = img.shape[0]
            if width != width_t or height != height_t:
                print('{} size unequal to image size '.format(xml_path[0:-3]))
                continue
            else:
                root = tree.getroot()
                idx = 1
                for obj in root.findall('object'):
                    name = obj.find('name').text
                    bnd_box = obj.find('bndbox')
                    xmin = int(bnd_box.find('xmin').text)
                    ymin = int(bnd_box.find('ymin').text)
                    xmax = int(bnd_box.find('xmax').text)
                    ymax = int(bnd_box.find('ymax').text)
                    obj_area = [xmin,ymin,xmax,ymax]
                    if xmin >= xmax or ymin >= ymax:
                        continue
                    else:
                        save_region(img,name,obj_area,xml_path,idx)
                        idx += 1
        else:
            print('Lack{}'.format(xml_path[0:-3]+'jpg'))

if __name__ == '__main__':
    data_path = 'data'
    crop_object_region(data_path)
