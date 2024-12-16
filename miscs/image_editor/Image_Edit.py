import cv2
import numpy as np

MAX_WIDTH = 800

class ImageProcessor:
    def __init__(self, image_path):
         self.image = image_path
        
    def flip(self, direction='horizontal'):
        if direction == 'horizontal':
           self.image = cv2.flip(self.image, 1) # 1 is for horizontal flip
        elif direction == 'vertical':
            self.image = cv2.flip(self.image, 0) # 0 is for vertical flip
        return self

    def rotate(self, angle):
         rows, cols = self.image.shape[:2]
         M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1) # center, angle, scale
         self.image = cv2.warpAffine(self.image, M, (cols, rows))
         return self
        
    def change_colorspace(self, target_space):
      if target_space == 'L':  # Grayscale
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
      elif target_space == 'HSV':
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
      elif target_space == 'RGB':
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
      return self

    def translate(self, x_offset, y_offset):
        rows, cols = self.image.shape[:2]
        M = np.float32([[1, 0, x_offset], [0, 1, y_offset]]) # translation matrix
        self.image = cv2.warpAffine(self.image, M, (cols+abs(x_offset), rows+abs(y_offset))) # (width, height)
        return self

    def crop(self, x1, y1, x2, y2):
        self.image = self.image[y1:y2, x1:x2] # [row_start:row_end, col_start:col_end]
        return self
    
    def save_image(self, output_path):
         cv2.imwrite(output_path, self.image)
   
    def calculate_scale(self):
        """Tính toán tỷ lệ scale để vừa với MAX_WIDTH."""
        width = self.image.shape[1]
        if width > MAX_WIDTH:
            scale = MAX_WIDTH / width
        else:
            scale = 1
        return scale

    def resize_image(self, scale):
        """Resize ảnh theo tỷ lệ scale."""
        if scale == 1:
           return self.image
        
        new_width = int(self.image.shape[1] * scale)
        new_height = int(self.image.shape[0] * scale)
        
        return cv2.resize(self.image,(new_width,new_height), interpolation=cv2.INTER_AREA)


    def display_image(self, scale):
        """Hiển thị ảnh với scale và trả về ảnh gốc."""
        resized_image = self.resize_image(scale)
        return resized_image
    
    def get_img(self):
        return self.image
    
    def set_img(self, img):
        self.image = img
        return self
  