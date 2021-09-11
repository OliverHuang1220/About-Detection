# About-Detection
目标检测比赛常用数据处理&amp;测试程序
```
crop_image.py
裁剪出xml文件中的目标区域并保存
This directory should contain the following data:
$File_Path
├── data
│   ├── 1.jpg
│   │—— 1.xml
│   │—— 2.jpg
│   └── 2.xml
└── crop_image.py
```
```
xml2coco.py
将VOC格式数据转成COCO数据格式
```

```
coco2voc.py
将COCO格式数据转成VOC数据格式
```

```
data_aug.py
针对图片内小目标数据增强的脚本；
增加目标的数量；
更新标签文件xml
```
