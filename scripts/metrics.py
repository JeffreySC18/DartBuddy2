from ultralytics import YOLO

if __name__ == '__main__':
    # new model against old dataset
    model = YOLO("model/best.pt")
    model.val(data="data_yolo/ogdartboard_pose.yaml")

    # old model against new dataset
    ogmodel = YOLO("model/ogbest.pt")
    ogmodel.val(data="data_yolo/dartboard_pose.yaml")