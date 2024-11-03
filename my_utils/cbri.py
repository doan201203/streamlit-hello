from superpoint import SuperPointFrontend

class CBRI:
    def __init__(self):
        self.detector = SuperPointFrontend(weights_path='./my_utils/superpoint_v1.pth', nms_dist=4, conf_thresh=0.015, nn_thresh=0.7)
        self.
    def embed(self):
      pass