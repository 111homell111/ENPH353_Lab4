#!/usr/bin/env python3

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi

import cv2
import sys
import numpy as np

class My_App(QtWidgets.QMainWindow):

	def __init__(self):
		super(My_App, self).__init__()
		loadUi("./SIFT_app.ui", self)

		self._cam_id = 0
		self._cam_fps = 2
		self._is_cam_enabled = False
		self._is_template_loaded = False

		self.browse_button.clicked.connect(self.SLOT_browse_button)
		self.toggle_cam_button.clicked.connect(self.SLOT_toggle_camera)

		# Timer used to trigger the video processing
		self._timer = QtCore.QTimer(self)
		self._timer.timeout.connect(self.SLOT_query_camera)
		self._timer.setInterval(1 / self._cam_fps)
		self.template_image = None

	def SLOT_browse_button(self):
		dlg = QtWidgets.QFileDialog()
		dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
		if dlg.exec_():
			self.template_path = dlg.selectedFiles()[0]

		pixmap = QtGui.QPixmap(self.template_path)
		self.template_label.setPixmap(pixmap)
		self.template_image = cv2.imread(self.template_path, cv2.IMREAD_GRAYSCALE)
		print("Loaded template image file: " + self.template_path)

	# Converts the OpenCV frame to QPixmap for display
	def convert_cv_to_pixmap(self, cv_img):
		cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
		height, width, channel = cv_img.shape
		bytesPerLine = channel * width
		q_img = QtGui.QImage(cv_img.data, width, height, 
							 bytesPerLine, QtGui.QImage.Format_RGB888)
		return QtGui.QPixmap.fromImage(q_img)

	# Resizes the frame to fit inside QLabel while maintaining aspect ratio
	def resize_frame(self, frame):
		label_width = self.live_image_label.width()
		label_height = self.live_image_label.height()

		# Get the aspect ratio of the frame
		frame_height, frame_width = frame.shape[:2]
		aspect_ratio = frame_width / frame_height

		# Calculate the new width and height while maintaining the aspect ratio
		if frame_width > frame_height:
			new_width = label_width
			new_height = int(new_width / aspect_ratio)
			if new_height > label_height:
				new_height = label_height
				new_width = int(new_height * aspect_ratio)
		else:
			new_height = label_height
			new_width = int(new_height * aspect_ratio)
			if new_width > label_width:
				new_width = label_width
				new_height = int(new_width / aspect_ratio)

		return cv2.resize(frame, (new_width, new_height))

	def SLOT_toggle_camera(self):
		if self._is_cam_enabled:
			self._timer.stop()
			self._is_cam_enabled = False
			self.toggle_cam_button.setText("&Enable video")
		else:
			video_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.avi)")
			if video_path:
				self._video_device = cv2.VideoCapture(video_path)
				if not self._video_device.isOpened():
					print("Error: Could not open video.")
					return
				self._timer.start()
				self._is_cam_enabled = True
				self.toggle_cam_button.setText("&Disable video")

	def SLOT_query_camera(self):
		ret, frame = self._video_device.read()
		if not ret:
			print("End of video.")
			self._timer.stop()
			return

		# Resize the frame to match the QLabel size
		frame = self.resize_frame(frame)

		sift = cv2.SIFT_create()
		template_keypoints, template_descriptors = sift.detectAndCompute(self.template_image, None)
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		frame_keypoints, frame_descriptors = sift.detectAndCompute(gray_frame, None)
		index_params = dict(algorithm=0, trees=5)
		search_params = dict()
		flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

		# If the template is loaded, perform feature matching
		if template_descriptors is not None and frame_descriptors is not None:
			print('b')
			# Match descriptors using Brute-Force Matcher
			matches = flann.knnMatch(template_descriptors, frame_descriptors, k=2)

			# Apply ratio test to keep good matches
			good_matches = []
			for m, n in matches:
				if m.distance < 0.75 * n.distance:
					good_matches.append(m)

			# Get the matching keypoints for the homography calculation
			if len(good_matches) > 4:  # Homography needs at least 4 points
				print('c')
				src_pts = np.float32([template_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
				dst_pts = np.float32([frame_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

				# Find the homography matrix
				H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

				if H is not None:
					# Get the height and width of the template image
					h, w = self.template_image.shape

					# Define the corners of the template image
					template_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

					# Use the homography to project the corners of the template onto the video frame
					projected_corners = cv2.perspectiveTransform(template_corners, H)

					# Draw the projected corners on the frame
					frame = cv2.polylines(frame, [np.int32(projected_corners)], True, (0, 255, 0), 3, cv2.LINE_AA)

				# Draw matches on the frame
				matched_frame = cv2.drawMatches(self.template_image, template_keypoints, frame, frame_keypoints, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

				# Resize the matched frame to fit inside the QLabel
				matched_frame = self.resize_frame(matched_frame)
				pixmap = self.convert_cv_to_pixmap(matched_frame)
			else:
				pixmap = self.convert_cv_to_pixmap(frame)
		else:
			pixmap = self.convert_cv_to_pixmap(frame)
		self.live_image_label.setPixmap(pixmap)



if __name__ == "__main__":
	app = QtWidgets.QApplication(sys.argv)
	myApp = My_App()
	myApp.show()
	sys.exit(app.exec_())