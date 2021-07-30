/***********和鲸社区水下目标检测**************/
/***********基于mmdet测试程序****************/
from mmdet.apis import init_detector, inference_detector
import os
import mmcv
import numpy as np
import json
import csv
import time
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"

# json编码规范
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def s_result(img,
             result,
             class_names,
             score_thr=0.3,
             wait_time=0,
             show=False,
             out_file=None,
             predict=[]):
    assert isinstance(class_names, (tuple, list))
    class_name = tuple(['holothurian', 'echinus', 'scallop', 'starfish'])
    
    img = mmcv.imread(img)
    img = img.copy()
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    for ann in range(len(labels)):
        # image_result = {
            # 'image_id': out_file.split('/')[-1][:-4],
            # 'category_id': int(labels[ann])+1,
            # 'score': float(bboxes[ann][4]),
            # 'bbox': [bboxes[ann][0], bboxes[ann][1], bboxes[ann][2]-bboxes[ann][0], bboxes[ann][3]-bboxes[ann][1]],
        # }
        image_result = [class_name[int(labels[ann])], 
                        out_file.split('/')[-1][:-4],
                        float(bboxes[ann][4]),
                        bboxes[ann][0], bboxes[ann][1], bboxes[ann][2], bboxes[ann][3]]
        if float(bboxes[ann][4])>0.001: ######################################
            predict.append(image_result)
    
    # if out_file specified, do not show image in window
    if out_file is not None:
        show = False
    # draw bounding boxes
    
    mmcv.imshow_det_bboxes(
        img,
        bboxes,
        labels,
        class_names=class_name,
        score_thr=score_thr,
        show=show,
        wait_time=wait_time,
        out_file=out_file
        )
    return predict


# 配置文件以及模型文件，更换时请同时修改结果保存路径
config_file = "/home1/huangqiangHD/mmdetection-master/water_merge_dir/data_merge1_giou_decay16_dir/cascade_rcnn_r101_fpn_dconv_c3-c5_1x_coco.py"##############################
checkpoint_file = "/home1/huangqiangHD/mmdetection-master/water_merge_dir/data_merge1_giou_decay16_dir/epoch_23.pth"####################################



# 模型初始化
model = init_detector(config_file, checkpoint_file, device='cuda:0')
print("开始计时")
t0 = time.time()
# 获取测试图像文件名
images = []
test_path = '/home1/huangqiangHD/dataset/UnderWater/test/test-B-image/'
import glob
images= sorted(glob.glob(test_path+'*.jpg'))
# 检测
num = 0
predict = []

save_path = '/home1/huangqiangHD/mmdetection-master/water_det_results/5_9_det_res_verB_2/' # 检测图像保存路径################################


for image in images:
    #print(image)
    
    image = str(image)
    result = inference_detector(model, image)
    
    predict = s_result(image, result, model.CLASSES, out_file=save_path +'visual_res/'+image.split('/')[-1], predict=predict)
    print('processing img {} in {}...'.format(num + 1, image))
    num += 1




with open(save_path+'det_results.csv', 'w', encoding='utf-8', newline='')as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['name','image_id','confidence','xmin','ymin','xmax','ymax'])
    results = []
    for pred in predict:
        results.append(pred)
    writer.writerows(results)
    print("平均时间：",((time.time() - t0)/ 1200.))
    print('copy py to save_path')
