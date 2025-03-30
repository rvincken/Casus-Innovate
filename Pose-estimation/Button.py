import cv2

class Button():
	def __init__(self, frame, text, x, y, width, height, color, hover_color, action=None):
		self.text = text
		self.x = x
		self.y = y
		self.width = width
		self.height = height
		self.color = color
		self.hover_color = hover_color
		self.action = action
		self.frame = frame
	def draw(self):
		cv2.rectangle(self.frame, (self.x, self.y), (self.x + self.width, self.y + self.height), self.color, -1)
		cv2.putText(self.frame, "Open Venster", (self.x, self.y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
	def is_hover(self, pos):
		if pos[0] > self.x and pos[0] < self.x + self.width:
			if pos[1] > self.y and pos[1] < self.y + self.height:
				return True
		return False
	def update(self, pos):
		if self.is_hover(pos):
			self.color = self.hover_color
		else:
			self.color = (255, 255, 255)