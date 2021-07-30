import os
import shutil
from os.path import join
import cv2
import glob

root_dir = "./fruit"  # 原始图片保存的位置
save_dir = "./bbox"  # 生成截取图片的位置

jpg_list = glob.glob(root_dir + "/*.jpg")

fo = open("dpj_small.txt", "w")  # 截取出来的图片位置

max_s = -1
min_s = 1000

for jpg_path in jpg_list:  # 遍历所有图片
    # jpg_path = jpg_list[3]
    txt_path = jpg_path.replace("jpg", "txt")  # 得到文件中相应注释的文件
    jpg_name = os.path.basename(jpg_path)  #

    f = open(txt_path, "r")  # 打开注释

    img = cv2.imread(jpg_path)  # 打开图片

    height, width, channel = img.shape  # 得到图片的尺寸

    file_contents = f.readlines()  # 读取注释

    for num, file_content in enumerate(file_contents):  #
        print(num)  # 打印种类
        clss, xc, yc, w, h = file_content.split()  # 得到种类和具体的坐标
        xc, yc, w, h = float(xc), float(yc), float(w), float(h)  # 对坐标浮点化
        # 将归一化的坐标转换为实际的坐标
        xc *= width
        yc *= height
        w *= width
        h *= height
        # 防止坐标超出实际范围
        max_s = max(w * h, max_s)
        min_s = min(w * h, min_s)
        # 得到图像坐标系下的位置
        half_w, half_h = w // 2, h // 2

        x1, y1 = int(xc - half_w), int(yc - half_h)
        x2, y2 = int(xc + half_w), int(yc + half_h)
        # 进行截取
        crop_img = img[y1:y2, x1:x2]

        new_jpg_name = jpg_name.split('.')[0] + "_crop_" + str(num) + ".jpg"  # 存储图片的名称
        cv2.imwrite(os.path.join(save_dir, new_jpg_name), crop_img)  # 截取的图片
        # cv2.imshow("croped",crop_img)
        # cv2.waitKey(0)
        fo.write(os.path.join(save_dir, new_jpg_name) + "\n")  # 截取后的注释

    f.close()

fo.close()

print(max_s, min_s)
