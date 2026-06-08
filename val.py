import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
       model = YOLO(r"arch_analysis_in_match_weight/best.pt")  # select your model.pt path  # select your model.pt path

    metrics =  model.val(data=R'hierarchical_configs/class_hier_v2.yaml',
            split='val',
            imgsz= 1024,
            batch= 16,
            workers= 2,
            device = 1,
            use_simotm="RGB",  # 3 通道 RGB 
            channels=3,
            save_txt=True,
            save_conf=True,
            project='runs/val/',
            name='class_hier',
            )
    print('map0.5:%.3f'%(metrics.box.map50))
    # print('F1:%.3f'%(metrics.box.f1.mean()))

