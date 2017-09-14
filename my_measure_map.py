import os
import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time
from keras_frcnn import config
import keras_frcnn.resnet as nn
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers
from keras_frcnn import data_generators
from sklearn.metrics import average_precision_score


def voc_ap(rec, prec, use_07_metric=False):
  """
  rec, prec: recall, precision should be in np array type
  ap = voc_ap(rec, prec, [use_07_metric])
  Compute VOC AP given precision and recall.
  If use_07_metric is true, uses the
  VOC 07 11 point method (default:False).
  """
  if use_07_metric:
    # 11 point metric
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
      if np.sum(rec >= t) == 0:
        p = 0
      else:
        p = np.max(prec[rec >= t])
      ap = ap + p / 11.
  else:
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
      mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
  return ap


def ground_truth_to_nparr(gt, f, class_name):
	al = []
	fx, fy = f
	for gtbb in gt:
		if gtbb['class'] == class_name:
			gt_x1 = gtbb['x1'] / fx
			gt_x2 = gtbb['x2'] / fx
			gt_y1 = gtbb['y1'] / fy
			gt_y2 = gtbb['y2'] / fy
			al.append(np.array([gt_x1, gt_x2, gt_y1, gt_y2]))
	return np.array(al)


def eval_each_img(pred, gt, f, overlap_thresh=0.5):
	# all the dict below is by class
	tp = {}  # true positive
	fp = {}  # false positive
	fn = {}  # false negative (undetected ground truth boxes)
	prob_fp = []  # proba of false positive
	fx, fy = f
	for gt_bbox in gt:
		gt_bbox['bbox_matched'] = False

	pred_probs = np.array([s['prob'] for s in pred])
	box_idx_sorted_by_prob = np.argsort(pred_probs)[::-1]

	for box_idx in box_idx_sorted_by_prob:
		pred_box = pred[box_idx]
		pred_class = pred_box['class']
		# init tp, fp, fn:
		if pred_class not in tp:
			tp[pred_class] = 0
		if pred_class not in fp:
			fp[pred_class] = 0
		if pred_class not in fn:
			fn[pred_class] = 0
		pred_x1 = pred_box['x1']
		pred_x2 = pred_box['x2']
		pred_y1 = pred_box['y1']
		pred_y2 = pred_box['y2']

		# find list of ground truth with matched class
		idx_gt_class_matched = [i for i in range(len(gt)) if gt[i]['class'] == pred_class]
		#print(box_idx)
		#print(idx_gt_class_matched)

		# compute IoU with GT_same_class and find the max
		if len(idx_gt_class_matched) > 0:  # this line is important: only consider ground truth of same class
			idx_max = -1
			cur_max_IoU = -np.inf
			for idx in idx_gt_class_matched:
				gt_box = gt[idx]
				gt_x1 = gt_box['x1'] / fx
				gt_x2 = gt_box['x2'] / fx
				gt_y1 = gt_box['y1'] / fy
				gt_y2 = gt_box['y2'] / fy
				cur_IoU = data_generators.iou((pred_x1, pred_y1, pred_x2, pred_y2), (gt_x1, gt_y1, gt_x2, gt_y2))
				if cur_max_IoU < cur_IoU:
					cur_max_IoU = cur_IoU
					idx_max = idx
			if cur_max_IoU >= overlap_thresh:
				if not gt[idx_max]['bbox_matched']:  # gt not detected previously
					tp[pred_class] += 1  # true positive
					gt[idx_max]['bbox_matched'] = True
				else:
					fp[pred_class] += 1
					prob_fp.append(pred_box['prob'])

			else:  # detect where no ground truth overthere -> FP
				fp[pred_class] += 1
				prob_fp.append(pred_box['prob'])

	# now deal with the false negative:
	gt_bbox_not_detected = [gt_bbox for gt_bbox in gt if not gt_bbox['bbox_matched']]
	for undetected_bb in gt_bbox_not_detected:
		# for VOC difficult:
		if 'difficult' in gt_bbox_not_detected:
			if gt_bbox_not_detected['difficult']:
				continue

		bb_class = undetected_bb['class']
		# init all to make sure the accumulation match
		if bb_class not in fn:
			fn[bb_class] = 0
		if bb_class not in tp:
			tp[bb_class] = 0
		if bb_class not in fp:
			fp[bb_class] = 0

		fn[bb_class] += 1

	return tp, fp, fn, prob_fp


def extract_non_nan(cum_tp_by_key, cum_fp_by_key, cum_fn_by_key):
	# find first idx that tp[idx] + fp[idx] > 0
	id_first_prec = -1
	id_first_rec = -1
	for i in range(len(cum_tp_by_key)):
		if cum_tp_by_key[i] + cum_fp_by_key[i] > 0:
			if id_first_prec == -1:  # only update the first time
				id_first_prec = i
		if cum_tp_by_key[i] + cum_fn_by_key[i] > 0:
			if id_first_rec == -1:
				id_first_rec = i
		if id_first_prec != -1 and id_first_rec != -1:
			break
	id_first = max(id_first_rec, id_first_prec)

	return cum_tp_by_key[id_first:], cum_fp_by_key[id_first:], cum_fn_by_key[id_first:]


def get_map(pred, gt, f):
	T = {}
	P = {}
	fx, fy = f

	for bbox in gt:
		bbox['bbox_matched'] = False

	pred_probs = np.array([s['prob'] for s in pred])
	box_idx_sorted_by_prob = np.argsort(pred_probs)[::-1]

	for box_idx in box_idx_sorted_by_prob:
		pred_box = pred[box_idx]
		pred_class = pred_box['class']
		pred_x1 = pred_box['x1']
		pred_x2 = pred_box['x2']
		pred_y1 = pred_box['y1']
		pred_y2 = pred_box['y2']
		pred_prob = pred_box['prob']
		if pred_class not in P:
			P[pred_class] = []
			T[pred_class] = []
		P[pred_class].append(pred_prob)
		found_match = False

		for gt_box in gt:
			gt_class = gt_box['class']
			gt_x1 = gt_box['x1']/fx
			gt_x2 = gt_box['x2']/fx
			gt_y1 = gt_box['y1']/fy
			gt_y2 = gt_box['y2']/fy
			gt_seen = gt_box['bbox_matched']
			if gt_class != pred_class:
				continue
			if gt_seen:
				continue
			iou = data_generators.iou((pred_x1, pred_y1, pred_x2, pred_y2), (gt_x1, gt_y1, gt_x2, gt_y2))
			if iou >= 0.5:
				found_match = True
				gt_box['bbox_matched'] = True
				break
			else:
				continue

		T[pred_class].append(int(found_match))

	for gt_box in gt:
		if not gt_box['bbox_matched']: #and not gt_box['difficult']:
			if gt_box['class'] not in P:
				P[gt_box['class']] = []
				T[gt_box['class']] = []

			T[gt_box['class']].append(1)
			P[gt_box['class']].append(0)

	#import pdb
	#pdb.set_trace()
	return T, P

sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("-p", "--path", dest="test_path", help="Path to test data.")
parser.add_option("-n", "--num_rois", dest="num_rois",
				help="Number of ROIs per iteration. Higher means more memory use.", default=32)
parser.add_option("--config_filename", dest="config_filename", help=
				"Location to read the metadata related to the training (generated when training).",
				default="config.pickle")
parser.add_option("-o", "--parser", dest="parser", help="Parser to use. One of simple or pascal_voc",
				default="pascal_voc"),

(options, args) = parser.parse_args()

if not options.test_path:   # if filename is not given
	parser.error('Error: path to test data must be specified. Pass --path to command line')


if options.parser == 'pascal_voc':
	from keras_frcnn.pascal_voc_parser import get_data
elif options.parser == 'simple':
	from keras_frcnn.simple_parser import get_data
else:
	raise ValueError("Command line option parser must be one of 'pascal_voc' or 'simple'")

config_output_filename = options.config_filename

with open(config_output_filename, 'rb') as f_in:
	C = pickle.load(f_in)

# turn off any data augmentation at test time
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

img_path = options.test_path


def format_img(img, C):
	img_min_side = float(C.im_size)
	(height,width,_) = img.shape
	
	if width <= height:
		f = img_min_side/width
		new_height = int(f * height)
		new_width = int(img_min_side)
	else:
		f = img_min_side/height
		new_width = int(f * width)
		new_height = int(img_min_side)
	fx = width/float(new_width)
	fy = height/float(new_height)
	img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
	#img = img[:, :, (2, 1, 0)]
	img = img.astype(np.float32)
	img[:, :, 0] -= C.img_channel_mean[0]
	img[:, :, 1] -= C.img_channel_mean[1]
	img[:, :, 2] -= C.img_channel_mean[2]
	img /= C.img_scaling_factor
	img = np.transpose(img, (2, 0, 1))
	img = np.expand_dims(img, axis=0)
	return img, fx, fy


class_mapping = C.class_mapping

if 'bg' not in class_mapping:
	class_mapping['bg'] = len(class_mapping)

class_mapping = {v: k for k, v in class_mapping.items()}
# print(class_mapping)
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
C.num_rois = int(options.num_rois)

if K.image_dim_ordering() == 'th':
	input_shape_img = (3, None, None)
	input_shape_features = (1024, None, None)
else:
	input_shape_img = (None, None, 3)
	input_shape_features = (None, None, 1024)


img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

model_rpn = Model(img_input, rpn_layers)
model_classifier_only = Model([feature_map_input, roi_input], classifier)

model_classifier = Model([feature_map_input, roi_input], classifier)

model_rpn.load_weights(C.model_path, by_name=True)
model_classifier.load_weights(C.model_path, by_name=True)

model_rpn.compile(optimizer='sgd', loss='mse')
model_classifier.compile(optimizer='sgd', loss='mse')

print(options.test_path)
all_imgs, _, _ = get_data(options.test_path)
print(len(all_imgs))
test_imgs = [s for s in all_imgs if s['imageset'] == 'test']


T = {}
P = {}
print(len(test_imgs))

TruePos = {}
FalsePos = {}
FalseNeg = {}
proba_false_pos = []


skip = 0


for idx, img_data in enumerate(test_imgs):
        #print('{}/{}'.format(idx,len(test_imgs)))
	st = time.time()
	filepath = img_data['filepath']

	img = cv2.imread(filepath)

	X, fx, fy = format_img(img, C)

	if K.image_dim_ordering() == 'tf':
		X = np.transpose(X, (0, 2, 3, 1))

	# get the feature maps and output from the RPN
	[Y1, Y2, F] = model_rpn.predict(X)

	R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.7)

	# convert from (x1,y1,x2,y2) to (x,y,w,h)
	R[:, 2] -= R[:, 0]
	R[:, 3] -= R[:, 1]

	# apply the spatial pyramid pooling to the proposed regions
	bboxes = {}
	probs = {}

	for jk in range(R.shape[0] // C.num_rois + 1):
		ROIs = np.expand_dims(R[C.num_rois * jk:C.num_rois * (jk + 1), :], axis=0)
		if ROIs.shape[1] == 0:
			break

		if jk == R.shape[0] // C.num_rois:
			# pad R
			curr_shape = ROIs.shape
			target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
			ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
			ROIs_padded[:, :curr_shape[1], :] = ROIs
			ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
			ROIs = ROIs_padded

		[P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

		for ii in range(P_cls.shape[1]):

			if np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
				continue

			cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

			if cls_name not in bboxes:
				bboxes[cls_name] = []
				probs[cls_name] = []

			(x, y, w, h) = ROIs[0, ii, :]

			cls_num = np.argmax(P_cls[0, ii, :])
			try:
				(tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
				tx /= C.classifier_regr_std[0]
				ty /= C.classifier_regr_std[1]
				tw /= C.classifier_regr_std[2]
				th /= C.classifier_regr_std[3]
				x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
			except:
				pass
			bboxes[cls_name].append([16 * x, 16 * y, 16 * (x + w), 16 * (y + h)])
			probs[cls_name].append(np.max(P_cls[0, ii, :]))

	all_dets = []

	for key in bboxes:
		bbox = np.array(bboxes[key])

		new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.7)
		for jk in range(new_boxes.shape[0]):
			if new_probs[jk] > 0.0:  # threshold for detection
				(x1, y1, x2, y2) = new_boxes[jk, :]
				det = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': key, 'prob': new_probs[jk]}
				all_dets.append(det)


	#print('Elapsed time = {}'.format(time.time() - st))

	# print('file image = ', img_data['filepath'])
	# print('all detections = ')
	# print(all_dets)
	# print()
    #
	# print('ground truth = ')
	# print(img_data['bboxes'])
	# print()


	t, p = get_map(all_dets, img_data['bboxes'], (fx, fy))

	tp, fp, fn, cur_prob_fp = eval_each_img(all_dets, img_data['bboxes'], (fx, fy))

	# print('TP = ', tp)
	# print('FP = ', fp)
	# print('FN = ', fn)
	# if len(cur_prob_fp):
	# 	print('cur mean of proba false positive = ', np.mean(cur_prob_fp))

	proba_false_pos += cur_prob_fp

	for key in tp.keys():
		if key not in TruePos:
			TruePos[key] = []
		TruePos[key] += [tp[key]]  # make tp[key] a list, TruePos extend that list to accumulate later
	for key in fp.keys():
		if key not in FalsePos:
			FalsePos[key] = []
		FalsePos[key] += [fp[key]]
	for key in fn.keys():
		if key not in FalseNeg:
			FalseNeg[key] = []
		FalseNeg[key] += [fn[key]]


	# print(TruePos)
	# print(FalsePos)
	# print(FalseNeg)
	# print(proba_false_pos)

	# skip += 1
	# if skip == 200:
	# 	break

	for key in t.keys():
		if key not in T:
			T[key] = []
			P[key] = []
		T[key].extend(t[key])
		P[key].extend(p[key])
	all_aps = []
	#for key in T.keys():
		#ap = average_precision_score(T[key], P[key])
		#print('{} AP: {}'.format(key, ap))
		#all_aps.append(ap)
	#print('mAP = {}'.format(np.mean(np.array(all_aps))))
	#print(T)
	#print(P)
K.clear_session()

recalls = {}
precisions = {}
all_ap = {}
for key in TruePos.keys():
	print(key)
	all_tp_by_key = np.cumsum(np.array(TruePos[key]))
	all_fp_by_key = np.cumsum(np.array(FalsePos[key]))
	all_fn_by_key = np.cumsum(np.array(FalseNeg[key]))
	print('true pos cummu')
	print(all_tp_by_key, end='\n\n')
	print('false pos cummu')
	print(all_fp_by_key, end='\n\n')
	print('false neg cummu')
	print(all_fn_by_key, end='\n\n')
	all_tp_by_key, all_fp_by_key, all_fn_by_key = extract_non_nan(all_tp_by_key, all_fp_by_key, all_fn_by_key)
	precisions[key] = all_tp_by_key / (all_tp_by_key + all_fp_by_key)
	recalls[key] = all_tp_by_key / (all_tp_by_key + all_fn_by_key)
	# print(precisions[key])
	# print(recalls[key])
	all_ap[key] = voc_ap(recalls[key], precisions[key], False)


print('min proba detection is ', min(proba_false_pos))

print('mAP = %f' % np.mean(list(all_ap.values())))

classStr = ''
apStr = ''

for key in all_ap:
	classStr += '%s & ' % key
	apStr += '%f & ' % all_ap[key]
	print('%12s %.3f %.3f %.3f' %(key, all_ap[key], precisions[key][-1], recalls[key][-1]))

print(classStr)
print(apStr)

#import matplotlib.pyplot as plt
#plt.hist(proba_false_pos, bins=20)
#plt.ylabel('Number of False Positive')
#plt.xlabel('Predicted Probability')
#plt.title('Distribution of predicted probability for False Positive case')
#plt.savefig('ProbaFP.png')
