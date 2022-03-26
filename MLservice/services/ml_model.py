import yaml
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import  DefaultPredictor
import os

FILE_INFER_CONFIG = os.path.join("config", "inference.yaml")
with open(FILE_INFER_CONFIG) as file:
    params = yaml.load(file, Loader = yaml.FullLoader)

def setup_config_infer(params):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(params["MODEL"]))
    cfg.OUTPUT_DIR = params["OUTPUT_DIR"]
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, params["TRANSFER_LEARNING"])
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.MODEL.DEVICE = params["DEVICE"]
    if "retina" in params["MODEL"]:
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = params["SCORE_THR"]
        cfg.MODEL.RETINANET.NUM_CLASSES = params["NUM_CLASSES"]
        cfg.MODEL.RETINANET.NMS_THRESH_TEST = params["NMS_THR"]
    else:
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = params["SCORE_THR"]
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = params["NUM_CLASSES"]
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = params["NMS_THR"]

    return cfg

def load_model(cfg):
    return DefaultPredictor(cfg)

class MLMgr:
    @staticmethod
    def GetModel():
        # load model
        params['SCORE_THR']=0.2
        params['NMS_THR']=0.5
        cfg = setup_config_infer(params)
        ai_model = load_model(cfg)
        return ai_model
