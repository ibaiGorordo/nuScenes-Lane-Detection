from nuscenes.nuscenes import NuScenes
import matplotlib.pyplot as plt
import numpy as np
import cv2

# sensor_names = ["CAM_FRONT","CAM_BACK"]
sensor_names = ["CAM_FRONT"]
scene_id = 3
root_path = 'data/sets/nuscenes/'
lower_limit = np.array([120,150,180])
upper_limit = np.array([220,220,220])
kernel = np.ones((15,15),np.uint8)

# Point order: Top-Left, Top-Right, Bottom-Left, Bottom-Right
input_points = np.array([[600,565],[1100,565],[1600,900],[70,900]], dtype = "float32")
def TransformImagePerspective(input_points,input_image):

	bottom_width = input_points[2,0] - input_points[3,0]
	height = input_points[2,1] - input_points[0,1]

	rectangle_points = np.array([
		[0, 0],
		[bottom_width - 1, 0],
		[bottom_width - 1, height - 1],
		[0, height - 1]], dtype = "float32")

	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(input_points, rectangle_points)

	transformed_image = cv2.warpPerspective(input_image, M, (bottom_width, height))
	return transformed_image

def GetFilenameList(sensor_list=['CAM_FRONT'],scene_id=0):

	# Get the first sample in the selected scene
	current_sample = nusc.get('sample', nusc.scene[scene_id]['first_sample_token'])
	
	# Extract the filenames for the samples in the scene
	filename_list = []
	while not current_sample["next"] == "":
		sample_file_list = []
		for sensor in sensor_list:
			sensor_data = nusc.get('sample_data', current_sample['data'][sensor])
			sample_file_list.append(sensor_data["filename"])

		filename_list.append(sample_file_list)

		# Update the current sample with the next sample
		current_sample = nusc.get('sample', current_sample["next"])
	return filename_list

def GetSensorCalibration(sensor_list=['CAM_FRONT'],scene_id=0):

	# Get the first sample in the selected scene
	current_sample = nusc.get('sample', nusc.scene[scene_id]['first_sample_token'])

	# Get the calibration data for each of the sensors
	calibration_data = []
	for sensor in sensor_list:
		sensor_data = nusc.get('sample_data', current_sample['data'][sensor])
		calibration_data.append(nusc.get("calibrated_sensor", sensor_data["calibrated_sensor_token"]))

	return calibration_data


if __name__ == '__main__':
	nusc = NuScenes(version='v1.0-mini', dataroot=root_path, verbose=True)

	sensor_filenames = GetFilenameList(sensor_names,scene_id)
	sensor_calibration_data = GetSensorCalibration(sensor_names,scene_id)

	for sample_filenames in sensor_filenames:
		for sensor_filename, sensor_calibration in zip(sample_filenames,sensor_calibration_data):
			image = cv2.imread(root_path+sensor_filename)
			intrinsic_matrix = np.array(sensor_calibration["camera_intrinsic"])

			image = cv2.undistort(image, intrinsic_matrix, None)
			image = TransformImagePerspective(input_points,image)
			cv2.imshow("Transformed", cv2.resize(image, (720, 480)))

			mask = cv2.inRange(image, lower_limit, upper_limit)
			mask = cv2.dilate(mask,kernel,iterations = 1)

			edges = cv2.Canny(mask,50,150)
			vis = cv2.resize(edges, (720, 480))  
			cv2.imshow("Edges", vis)
			cv2.waitKey(10) 

		
	cv2.destroyAllWindows()
