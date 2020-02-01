from nuscenes.nuscenes import NuScenes
import numpy as np
import cv2

sensor = 'CAM_FRONT'
scene_id = 1
root_path = 'data/sets/nuscenes/'
def GetFilenameList(sensor='CAM_FRONT',scene_id=0):
	
	selected_scene = nusc.scene[scene_id]

	# Get the sensor data for the first sample of the selected scene 
	first_sample_token = selected_scene['first_sample_token']
	current_sample = nusc.get('sample', first_sample_token)
	sensor_data = nusc.get('sample_data', current_sample['data'][sensor])

	# Extract the file name containing the sensor data
	fileNameList = [sensor_data["filename"]]
	while not current_sample["next"] == "":
		current_sample = nusc.get('sample', current_sample["next"])
		sensor_data = nusc.get('sample_data', current_sample['data'][sensor])
		fileNameList.append(sensor_data["filename"])
	return fileNameList

if __name__ == '__main__':
	nusc = NuScenes(version='v1.0-mini', dataroot=root_path, verbose=True)

	fileNameList = GetFilenameList(sensor,scene_id)

	for filename in fileNameList:
		image = cv2.imread(root_path+filename)

		# Use Canny edge detection on the grayscale image
		gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)	
		gray = cv2.GaussianBlur(gray, (5, 5), 0)
		edges = cv2.Canny(gray,50,200)

		# Use region mask on the lower part of the image
		# to remove the lines outside the road
		height, width = gray.shape
		polygons = np.array([[(0, height-200), (width//2, height//2), (width, height-200)]])
		mask = np.zeros_like(gray)
		cv2.fillPoly(mask, polygons, 255)
		filtered_edges = cv2.bitwise_and(edges, mask)

		# hough = cv2.HoughLinesP(filtered_edges, 2, np.pi / 180, 100, np.array([]), minLineLength = 100, maxLineGap = 50)
		# print(hough)

		vis = cv2.resize(filtered_edges, (720, 480))  
		cv2.imshow("Video", vis)  
		cv2.waitKey(30) 
	cv2.destroyAllWindows()

