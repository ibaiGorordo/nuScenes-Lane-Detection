from nuscenes.nuscenes import NuScenes
import matplotlib.pyplot as plt
import numpy as np
import cv2

# sensor_names = ["CAM_FRONT","CAM_FRONT_RIGHT","CAM_BACK_RIGHT","CAM_BACK","CAM_BACK_LEFT","CAM_FRONT_LEFT"]
sensor_names = ["CAM_FRONT","CAM_FRONT_RIGHT"]
scene_id = 0
root_path = 'data/sets/nuscenes/'
image_overlap = 15/100 

def GetFilenameList(sensor_list='CAM_FRONT',scene_id=0):

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

def GetSensorCalibration(sensor_list='CAM_FRONT',scene_id=0):

	# Get the first sample in the selected scene
	current_sample = nusc.get('sample', nusc.scene[scene_id]['first_sample_token'])

	# Get the calibration data for each of the sensors
	calibration_data = []
	for sensor in sensor_list:
		sensor_data = nusc.get('sample_data', current_sample['data'][sensor])
		calibration_data.append(nusc.get("calibrated_sensor", sensor_data["calibrated_sensor_token"]))

	return calibration_data

def StichImages(img_left,img_right):
	gray_left = cv2.cvtColor(img_left,cv2.COLOR_BGR2GRAY)
	gray_right = cv2.cvtColor(img_right,cv2.COLOR_BGR2GRAY)
	height, width = gray_left.shape
	gray_left = gray_left[:,round(width*(1-image_overlap)):]
	gray_right = gray_right[:,:round(width*image_overlap)]

	# sift = cv2.ORB_create()
	sift = cv2.xfeatures2d.SIFT_create(edgeThreshold=10)

	# find key points
	kp_left, des_left = sift.detectAndCompute(gray_left,None)
	kp_right, des_right = sift.detectAndCompute(gray_right,None)

	# # Brute force match the descriptors
	# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	# matches = bf.match(des_left,des_right)
	# matches = sorted(matches, key = lambda x:x.distance)
	# matches = [match for match in matches if match.distance < 50]
	# src_pts = np.float32([ kp_right[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
	# dst_pts = np.float32([tuple(sum(x) for x in zip(kp_left[m.queryIdx].pt, (round(width*(1-image_overlap)),0))) for m in matches]).reshape(-1,1,2)

	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des_left,des_right, k=2)

	# Apply ratio test
	good = []
	for m in matches:
		if m[0].distance < 0.5*m[1].distance:
			good.append(m)
	matches = np.asarray(good)
	
	src_pts = np.float32([ kp_right[m.trainIdx].pt for m in matches[:,0]]).reshape(-1,1,2)
	dst_pts = np.float32([tuple(sum(x) for x in zip(kp_left[m.queryIdx].pt, (round(width*(1-image_overlap)),0))) for m in matches[:,0]]).reshape(-1,1,2)

	
	M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

	# Matrix obtained averaging the matrices of various good stiches
	M = np.array([[-2.76926139e-02,  8.82836032e-02,  1.39366897e+03],
 		[-2.94342047e-01,  9.48160747e-01,  2.81456705e+01],
 		[-6.24142082e-04,  1.45840918e-05,  1.00000000e+00]])

	# print(M)
	matchesMask = mask.ravel().tolist()
	
	good = [matches[i] for i in range(len(matches)) if matchesMask[i]==1]

	stiched_img = cv2.warpPerspective(img_right,M,(img_left.shape[1] + img_right.shape[1], img_left.shape[0]))
	stiched_img[0:img_left.shape[0], 0:img_left.shape[1]] = img_left
	# img3 = cv2.drawMatches(gray_left,kp_left,gray_right,kp_right,good,None)
	# vis = cv2.resize(img3, (720, 480))  
	# cv2.imshow("matches",vis)
	# cv2.waitKey(0) 

	return stiched_img

def trim(frame):
    #crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    #crop bottom
    elif not np.sum(frame[-1]):
        return trim(frame[:-2])
    #crop left
    elif not np.sum(frame[:,0]):
        return trim(frame[:,1:]) 
    #crop right
    elif not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])    
    return frame


if __name__ == '__main__':
	nusc = NuScenes(version='v1.0-mini', dataroot=root_path, verbose=True)

	sensor_filenames = GetFilenameList(sensor_names,scene_id)
	sensor_calibration_data = GetSensorCalibration(sensor_names,scene_id)

	for sample_filenames in sensor_filenames:
		images = []
		for sensor_filename, sensor_calibration in zip(sample_filenames,sensor_calibration_data):
			image = cv2.imread(root_path+sensor_filename)
			intrinsic_matrix = np.array(sensor_calibration["camera_intrinsic"])

			images.append(cv2.undistort(image, intrinsic_matrix, None))

		stiched_img = images[0]
		count = 0
		for image in images[1:]:
			try:
				stiched_img = StichImages(trim(stiched_img),image)
			except:
				print("No match")
			count += 1

		vis = cv2.resize(stiched_img, (720, 480))  
		cv2.imshow("stiched", vis)
		cv2.waitKey(10) 
	cv2.destroyAllWindows()
