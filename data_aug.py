# coding: utf-8
# author: hxy
# 2021-4-25,python3.8
"""
针对图片内小目标数据增强的脚本；
增加目标的数量；
更新标签文件.xml;
"""
import os
import cv2
import time

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

from xml.dom.minidom import parseString

from lxml.etree import Element, SubElement, tostring


# 获取原始xml标签文件中的标注信息：obj_name、bbox
def GetAnnotBoxLoc(AnotPath):
    tree = ET.ElementTree(file=AnotPath)
    root = tree.getroot()
    ObjectSet = root.findall('object')
    ObjBndBoxSet = list()
    for Object in ObjectSet:
        BndBoxLoc = dict()
        ObjName = Object.find('name').text
        BndBox = Object.find('bndbox')
        x1 = int(BndBox.find('xmin').text.split('.')[0])
        y1 = int(BndBox.find('ymin').text.split('.')[0])
        x2 = int(BndBox.find('xmax').text.split('.')[0])
        y2 = int(BndBox.find('ymax').text.split('.')[0])
        BndBoxLoc[ObjName] = [x1, y1, x2, y2]
        ObjBndBoxSet.append(BndBoxLoc)
    return ObjBndBoxSet


# 裁剪标注的目标，并将其粘贴至原始目标上方和下方
def paste_object(img_file, bndboxes):
    img = cv2.imread(img_file)
    new_all_bbox = list()
    for i in range(len(bndboxes)):
        new_bbox_up = dict()
        new_bbox_down = dict()
        for obj_name, bboxes in zip(bndboxes[i].keys(), bndboxes[i].values()):
            xmin = bboxes[0]
            ymin = bboxes[1]
            xmax = bboxes[2]
            ymax = bboxes[3]

            # 裁剪原始标注的目标物体
            obj_pic = img[ymin - 5:ymax + 5, xmin :xmax]
            h, w = obj_pic.shape[:2]

            # 将目标下移
            img[ymin - 5 + h:ymax + 5 + h, xmin:xmin + w] = obj_pic
            new_bbox_down[obj_name] = [xmin, ymin + h, xmin + w, ymax + h]
            new_all_bbox.append(new_bbox_down)
            # 将目标上移:这里为了防止越界，进行一个逻辑判断；
            if ymin - h - 5 > 0:
                img[ymin - 5 - h:ymax + 5 - h, xmin:xmin + w] = obj_pic
                new_bbox_up[obj_name] = [xmin, ymin - h, xmin + w, ymax - h]
            else:
                img[ymin - 5 + 2 * h:ymax + 5 + 2 * h, xmin:xmin + w] = obj_pic
                new_bbox_up[obj_name] = [xmin, ymin + 2 * h, xmin + w, ymax + 2 * h]
            new_all_bbox.append(new_bbox_up)
    output_all_bbox = bndboxes + new_all_bbox

    # valid_data_aug(img, output_all_bbox, "val_check")  # 绘制box,验证数据增强的正确性
    return output_all_bbox, img.shape, img


# 验证数据增强的结果
def valid_data_aug(img, bndboxes, val_check_dir):
    if not os.path.exists(val_check_dir):
        os.mkdir(val_check_dir)
    # img = cv2.imread(img_file)
    for i in range(len(bndboxes)):
        for obj_name, bboxes in zip(bndboxes[i].keys(), bndboxes[i].values()):
            cv2.rectangle(img, (bboxes[0], bboxes[1]), (bboxes[2], bboxes[3]), (255, 0, 0), 1)
            cv2.putText(img, str(obj_name), (bboxes[0], bboxes[1]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)

    cv2.imwrite(os.path.join(val_check_dir, 'result' + str(time.time()) + '.jpg'), img)

    return


# 生成新的标签文件：.xml
def write_xml(all_bndboxes, img_name, img_path, shape):
    node_root = Element('annotation')
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'JPEGImage'

    node_img_name = SubElement(node_root, 'filename')
    node_img_name.text = img_name + '.jpg'
    node_img_path = SubElement(node_root, 'path')
    node_img_path.text = img_path

    node_source = SubElement(node_root, 'source')
    node_database = SubElement(node_source, 'database')
    node_database.text = 'Unknown'

    node_img_size = SubElement(node_root, 'size')
    node_img_width = SubElement(node_img_size, 'width')
    node_img_width.text = str(shape[1])  # 照片的w
    node_img_height = SubElement(node_img_size, 'height')
    node_img_height.text = str(shape[0])  # 照片的h
    node_img_depth = SubElement(node_img_size, 'depth')
    node_img_depth.text = str(shape[2])  # 照片的depth

    node_img_seg = SubElement(node_root, 'segmented')
    node_img_seg.text = '0'

    for i in range(len(all_bndboxes)):
        for obj_name, bboxes in zip(all_bndboxes[i].keys(), all_bndboxes[i].values()):
            node_obj = SubElement(node_root, 'object')
            node_obj_name = SubElement(node_obj, 'name')
            node_obj_name.text = obj_name  # obj的名字

            node_bbox = SubElement(node_obj, 'bndbox')
            node_bbox_xmin = SubElement(node_bbox, 'xmin')
            node_bbox_xmin.text = str(bboxes[0])
            node_bbox_ymin = SubElement(node_bbox, 'ymin')
            node_bbox_ymin.text = str(bboxes[1])
            node_bbox_xmax = SubElement(node_bbox, 'xmax')
            node_bbox_xmax.text = str(bboxes[2])
            node_bbox_ymax = SubElement(node_bbox, 'ymax')
            node_bbox_ymax.text = str(bboxes[3])

            node_difficult = SubElement(node_obj, 'difficult')
            node_difficult.text = '0'  # 全部设定为0

    xml = tostring(node_root)
    dom = parseString(xml)
    return dom


if __name__ == '__main__':
    xml_dir = './traffic_voc/Annotations'  # 原始xml文件存储文件夹
    img_dir = 'D:/Downloads/train'  # 原始照片存储文件夹

    new_xml_dir = './traffic_voc/aug_xmls'  # 生成的新xml存储文件夹
    if not os.path.exists(new_xml_dir):
        os.mkdir(new_xml_dir)

    new_pic_dir = './traffic_voc/aug_imgs'  # 生成的新照片存储文件夹
    if not os.path.exists(new_pic_dir):
        os.mkdir(new_pic_dir)

    for file in os.listdir(xml_dir):
        file_name = file.split('.')[0]
        xml = file_name + '.xml'
        pic = file_name + '.jpg'
        #new_file_name = 'aug_' + file_name
        new_file_name = file_name
        # 生成新的照片文件
        bndboxes = GetAnnotBoxLoc(AnotPath=os.path.join(xml_dir, xml))
        all_bboxes, img_shape, img = paste_object(img_file=os.path.join(img_dir, pic),
                                                  bndboxes=bndboxes)
        cv2.imwrite(os.path.join(new_pic_dir, new_file_name + '.jpg'), img)  # 存储数据增强后的照片

        # 生成新的标签文件xml
        dom = write_xml(all_bndboxes=all_bboxes, img_name=file_name,
                        img_path=os.path.join(img_dir, pic), shape=img_shape)
        # 存储数据增强后的标签文件
        with open(os.path.join(new_xml_dir, new_file_name + '.xml'), 'wb') as x:
            x.write(dom.toprettyxml(indent='\t', encoding='utf-8'))
        x.close()
