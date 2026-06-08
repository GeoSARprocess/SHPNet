import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
if __name__ == '__main__':
    model.train(data=R'hierarchical_configs/class_hier_FAIR-CSAR.yaml',  #FAIR-CSAR
                cache=False,
                imgsz=1024,
                epochs=200,
                batch=8,
                close_mosaic=10,
                workers=2,
                device= 0 ,
                lr0=0.001,
                amp = True ,#
                use_simotm="RGB",
                channels=3,
                project='runs/',
                name='Contrastive_Learning_3level_classarch',
               )
