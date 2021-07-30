/***********科大讯飞X光安检**************/
/***********基于mmdet测试程序****************/
from mmdet.apis import init_detector, inference_detector
import os
import mmcv
import numpy as np
import json
import csv
import time
import os
import glob

config_file = "/home1/huangqiangHD/mmdetection-master/xray_work/xray_1780_dir/cascade_rcnn_r101_fpn_dconv_c3-c5_1x_coco.py"
checkpoint_file = "/home1/huangqiangHD/mmdetection-master/xray_work/xray_1780_dir/epoch_12.pth"

model = init_detector(config_file, checkpoint_file, device='cuda:0')
images = []
#test_path = '/home1/huangqiangHD/dataset/xray2020/train/images/'
test_path='/home1/huangqiangHD/dataset/X_ray/test/'
images = sorted(glob.glob(test_path + '*.jpg'))
save_path = '/home1/huangqiangHD/mmdetection-master/xray_work/'  
result_path = save_path+'/test1780.json'
if not os.path.exists(save_path):
    os.makedirs(save_path)
class_name = tuple(['knife', 'scissors', 'sharpTools', 'expandableBaton', 'smallGlassBottle', 'electricBaton', 'plasticBeverageBottle',
    'plasticBottleWithaNozzle', 'electronicEquipment', 'battery', 'seal', 'umbrella'])

def s_result(img,
             result,
             class_names,
             score_thr=0.3,
             wait_time=0,
             show=True,
             out_file=None,
             ):
    img_res = []
    assert isinstance(class_names, (tuple, list))

    class_detlist = [[] for _ in range(12)]
    img = mmcv.imread(img)
    img = img.copy()
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.array(np.vstack(bbox_result)).tolist()
    bboxes1 = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    for ann in range(len(labels)):
        if float(bboxes[ann][4]) > 0.001:########################
            class_detlist[int(labels[ann])].append([bboxes[ann][0], bboxes[ann][1],
                                                    bboxes[ann][2], bboxes[ann][3],round(float(bboxes[ann][4]), 3)])
                                                    
    
    img_res.append(class_detlist)

    if out_file is not None:
        show = False
    # draw bounding boxes

    mmcv.imshow_det_bboxes(
        img,
        bboxes1,
        labels,
        class_names=class_name,
        score_thr=score_thr,
        show=show,
        wait_time=wait_time,
        out_file=out_file
    )
    return img_res


def main():
    num = 0
    predict = []
   
    t0 = time.time()
    
    for image in images:
        image = str(image)
        result = inference_detector(model, image)
        img_res = s_result(image, result, model.CLASSES, out_file=save_path + 'vis_res_test1780/' + image.split('/')[-1])
        predict.append(img_res)
        print('processing img {} in {}...'.format(num + 1, image))
        num += 1
    #print(predict)
    results = []
    for i in range(len(predict)):
        results.append(predict[i][0])
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=4)
 
if __name__ =='__main__':
    main()


