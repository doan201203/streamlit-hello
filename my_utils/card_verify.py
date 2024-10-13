from miscs.face_models.sface import SFace
from miscs.face_models.yunet import YuNet
import numpy as np
import cv2 as cv

class Verification:
  def __init__(self, detector: YuNet, recognizer: SFace):
    self.detector = detector
    self.recognizer = recognizer
  
  def set_card(self, card_number):
    self.card_number = card_number
  
  def set_selfie(self, selfie):
    self.selfie = selfie

  def detect(self, image):
    self.detector.setInputSize((image.shape[1], image.shape[0]))

    return self.detector.infer(image)
  
  def verify_card(self):
    face_card = self.detect(self.card_number)
    face2 = self.detect(self.selfie)
    if face_card.shape[0] > 0 and face2.shape[0] > 0:
      score, matches = [], []
      for face in face2:
        rs = self.recognizer.match(self.card_number, face_card[0][:-1], self.selfie, face[:-1])
        score.append(rs[0])
        matches.append(rs[1])
      return face_card, face2, score, matches
    return None, None, None, None
  
  def visualize(self, img1, faces1, img2, faces2, matches, scores, target_size=[512, 512]): # target_size: (h, w)
    out1 = img1.copy()
    out2 = img2.copy()
    matched_box_color = (0, 255, 0)    # BGR
    mismatched_box_color = (0, 0, 255) # BGR

    # Resize to 256x256 with the same aspect ratio
    padded_out1 = np.zeros((target_size[0], target_size[1], 3)).astype(np.uint8)
    h1, w1, _ = out1.shape
    ratio1 = min(target_size[0] / out1.shape[0], target_size[1] / out1.shape[1])
    new_h1 = int(h1 * ratio1)
    new_w1 = int(w1 * ratio1)
    resized_out1 = cv.resize(out1, (new_w1, new_h1), interpolation=cv.INTER_LINEAR).astype(np.float32)
    top = max(0, target_size[0] - new_h1) // 2
    bottom = top + new_h1
    left = max(0, target_size[1] - new_w1) // 2
    right = left + new_w1
    padded_out1[top : bottom, left : right] = resized_out1

    # Draw bbox
    bbox1 = faces1[0][:4] * ratio1
    x, y, w, h = bbox1.astype(np.int32)
    cv.rectangle(padded_out1, (x + left, y + top), (x + left + w, y + top + h), matched_box_color, 2)

    # Resize to 256x256 with the same aspect ratio
    padded_out2 = np.zeros((target_size[0], target_size[1], 3)).astype(np.uint8)
    h2, w2, _ = out2.shape
    ratio2 = min(target_size[0] / out2.shape[0], target_size[1] / out2.shape[1])
    new_h2 = int(h2 * ratio2)
    new_w2 = int(w2 * ratio2)
    resized_out2 = cv.resize(out2, (new_w2, new_h2), interpolation=cv.INTER_LINEAR).astype(np.float32)
    top = max(0, target_size[0] - new_h2) // 2
    bottom = top + new_h2
    left = max(0, target_size[1] - new_w2) // 2
    right = left + new_w2
    padded_out2[top : bottom, left : right] = resized_out2

    # Draw bbox
    assert faces2.shape[0] == len(matches), "number of faces2 needs to match matches"
    assert len(matches) == len(scores), "number of matches needs to match number of scores"
    for index, match in enumerate(matches):
        bbox2 = faces2[index][:4] * ratio2
        x, y, w, h = bbox2.astype(np.int32)
        box_color = matched_box_color if match else mismatched_box_color
        cv.rectangle(padded_out2, (x + left, y + top), (x + left + w, y + top + h), box_color, 2)

        score = scores[index]
        text_color = matched_box_color if match else mismatched_box_color
        cv.putText(padded_out2, "{:.2f}".format(score), (x + left, y + top - 5), cv.FONT_HERSHEY_DUPLEX, 0.4, text_color)

    return (padded_out1, padded_out2)
