import os
import os.path as pt
import json
import funcy
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sklearn.model_selection import train_test_split


#*********************#
#将数据集划分为train和val#
#*********************#

def split_train_val(annotations, trainpath, valpath, splitrate=0.99):#划分比例为99：1
    with open(annotations, 'rt', encoding='UTF-8') as annotations:
        coco = json.load(annotations)
        images = coco['images']
        annotations = coco['annotations']
        categories = coco['categories']

        # number_of_images = len(images)
        train, val = train_test_split(images, train_size=splitrate)

        save_coco(trainpath, train, filter_annotations(annotations, train),
                  categories)
        save_coco(valpath, val, filter_annotations(annotations, val),
                  categories)

        print("Saved {} entries in {} and {} in {}".format(
            len(train), trainpath, len(val), valpath))
            
def save_coco(file, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump(
            {
                'images': images,
                'annotations': annotations,
                'categories': categories
            },
            coco,
            indent=4,
            sort_keys=True)           

def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids,
                         annotations)
def main():
    COCO_FORMAT_JSON_PATH='/home1/huangqiangHD/dataset/X_ray/train'
    
    annotations = pt.join(COCO_FORMAT_JSON_PATH, 'train.json')#总的数据集
    trainpath = pt.join(COCO_FORMAT_JSON_PATH, 'train_.json')#train_set
    valpath = pt.join(COCO_FORMAT_JSON_PATH,'val.json')#val_set
    split_train_val(annotations, trainpath, valpath)
 
if __name__ == '__main__':
    main()