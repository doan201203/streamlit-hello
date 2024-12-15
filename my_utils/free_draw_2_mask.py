import cv2
import numpy as np

class Free2Mask:
  def __init__(self, img) -> None:
    self.mask = np.zeros(img.shape[:2], np.uint8)
    
  def get_mask(self):
    return self.mask
  
  def quadratic_bezier(self, p0, p1, p2, t):
          return (1-t)**2 * np.array(p0) + 2 * (1-t) * t * np.array(p1) + t**2 * np.array(p2)

  def draw_quad(self, p0, p1, p2, color, thickness=6):
      for i in np.linspace(0, 1, 10):
          poo = self.quadratic_bezier(p0, p1, p2, i)
          poo = tuple(map(int, poo))
          self.mask = cv2.circle(self.mask, poo, thickness, color, -1)

  def path(self, command, cc, thickness=6):
      current_pos = None
      for comman in command:
          if comman[0] == 'M':
              current_pos = comman[1:]
          elif comman[0] == 'Q':
              p0 = current_pos
              p1 = comman[1:3]
              p2 = comman[3:]
              self.draw_quad(p0, p1, p2, cc, thickness)
              current_pos = p2
          elif comman[0] == 'L':
              p0 = current_pos
              p1 = comman[1:]
              self.mask = cv2.line(self.mask, tuple(map(int, p0)), tuple(map(int, p1)), cc, thickness)
              current_pos = p1