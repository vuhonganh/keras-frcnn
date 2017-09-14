import cv2
import numpy as np

def get_data(input_path):
	found_bg = False
	all_imgs = {}

	classes_count = {}

	class_mapping = {}

	visualise = True
	cnt = 0
	cnt_2 = 0
	with open(input_path,'r') as f:

		print('Parsing annotation files')

		for line in f:
			line_split = line.strip().split(',')
			(filename,x1,y1,x2,y2,class_name) = line_split

			if class_name not in classes_count:
				classes_count[class_name] = 1
			else:
				classes_count[class_name] += 1

			if class_name not in class_mapping:
				if class_name == 'bg' and found_bg == False:
					print('Found class name with special name bg. Will be treated as a background region (this is usually for hard negative mining).')
					found_bg = True
				class_mapping[class_name] = len(class_mapping)

			if filename not in all_imgs:
				all_imgs[filename] = {}
				
				img = cv2.imread(filename)
				(rows,cols) = img.shape[:2]
				all_imgs[filename]['filepath'] = filename
				all_imgs[filename]['width'] = cols
				all_imgs[filename]['height'] = rows
				all_imgs[filename]['bboxes'] = []
				cnt += 1
				if cnt == 10:
					cnt = 0
					all_imgs[filename]['imageset'] = 'test'
				else:
					all_imgs[filename]['imageset'] = 'trainval'


			all_imgs[filename]['bboxes'].append({'class': class_name, 'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2)})
			#print(line)
			#print(all_imgs[filename]['bboxes'])

		# all_data = []
		# for key in all_imgs:
		# 	all_data.append(all_imgs[key])
		
		# # make sure the bg class is last in the list
		# if found_bg:
		# 	if class_mapping['bg'] != len(class_mapping) - 1:
		# 		key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping)-1][0]
		# 		val_to_switch = class_mapping['bg']
		# 		class_mapping['bg'] = len(class_mapping) - 1
		# 		class_mapping[key_to_switch] = val_to_switch

		all_data = list(all_imgs.values())
		#print(all_data[9])
		return all_data, classes_count, class_mapping


def write_test_file():
	test_imgs = 'test.txt'
	input_path = '../clean_bbox_v2.txt'
	all_data, _, _ = get_data(input_path)
	test_file_path = [all_data[i]['filepath'] for i in range(len(all_data)) if all_data[i]['imageset'] == 'test']
	test_file_path = sorted(test_file_path)
	with open(test_imgs, 'w') as f:
		for fp in test_file_path:
			f.write(fp + '\n')


def write_gt_test_file():
	test_imgs = 'gt_test.txt'
	input_path = '/data/hav16/imagenet/clean_bbox.txt'
	all_data, _, _ = get_data(input_path)
	print(all_data[9])
	# idx_test = [i for i in range(len(all_data)) if all_data[i]['imageset'] == 'test']
	# with open(test_imgs, 'w') as f:
	# 	for i in idx_test:
	# 		print(i, len(all_data[i]['bboxes']))
	# 		for bb in all_data[i]['bboxes']:
	# 			line = '%s,%d,%d,%d,%d,%s\n' % (all_data[i]['filepath'], bb['x1'], bb['x2'], bb['y1'], bb['y2'], bb['class'])
	# 			f.write(line)



	# test_file_path = [all_data[i]['filepath'] for i in range(len(all_data)) if all_data[i]['imageset'] == 'test']
	# test_file_path = sorted(test_file_path)
	# with open(test_imgs, 'w') as f:
	# 	for fp in test_file_path:
	# 		f.write(fp + '\n')


def write_test_imgs_to_dir():
	# write test image to a folder
	import shutil
	import os
	test_dir = './test_icl/'
	os.makedirs('test_icl', exist_ok=True)

	with open('test.txt', 'r') as f:
		for line in f:
			line = line.strip('\n')
			shutil.copy(line, test_dir)


if __name__ == '__main__':
	# train_imgs = 'train.txt'  # trainval however val is considered as train in this implementation
	#write_gt_test_file()
        write_test_file()
        write_test_imgs_to_dir()
