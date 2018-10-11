import time
import numpy as np
import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix
from zmd_dataset import TSNDataSet
from models import TSN
from transforms import *
from ops import ConsensusModule

def test_models(fileAddr, vModality, frames, weights, gpus, save_scores):
# fileAddr: video address
# vModality: 'RGB' or 'Flow'
# frames: total frame number of the video
# weights: model weight
# gpus: gpu index
# save_scores: score address
	num_class = 11
	test_segments = 25     # frame number
	workers = 1              # gpu index
	test_crops = 10
	# save_scores = 'test_tsn/score_file_' + str.lower(vModality) + '_2'

	net = TSN(num_class, 1, vModality,
			  base_model='BNInception',
			  consensus_type='avg',
			  dropout=0.7)

	checkpoint = torch.load(weights)
	print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))

	base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
	net.load_state_dict(base_dict)

	cropping = torchvision.transforms.Compose([GroupOverSample(net.input_size, net.scale_size)])

	data_loader = torch.utils.data.DataLoader(
			TSNDataSet("", fileAddr, num_frames = frames, num_segments=test_segments,
					   new_length=1 if vModality == "RGB" else 5,
					   modality=vModality,
					   image_tmpl="img_{:05d}.jpg" if vModality in ['RGB', 'RGBDiff'] else 'flow_'+"{}_{:05d}.jpg",
					   test_mode=True,
					   transform=torchvision.transforms.Compose([
						   cropping,
						   Stack(roll='BNInception' == 'BNInception'),
						   ToTorchFormatTensor(div='BNInception' != 'BNInception'),
						   GroupNormalize(net.input_mean, net.input_std),
					   ])),
			batch_size=1, shuffle=False,
			num_workers=workers * 2, pin_memory=True)

	if gpus is not None:
		devices = [gpus[i] for i in range(workers)]
	else:
		devices = list(range(workers))


	net = torch.nn.DataParallel(net.cuda(devices[0]), device_ids=devices)
	net.eval()

	data_gen = enumerate(data_loader)

	total_num = len(data_loader.dataset)
	output = []


	def eval_video(video_data):
		i, data, label = video_data
		num_crop = test_crops

		if vModality == 'RGB':
			length = 3
		elif vModality == 'Flow':
			length = 10
		elif vModality == 'RGBDiff':
			length = 18
		else:
			raise ValueError("Unknown modality "+vModality)

		input_var = torch.autograd.Variable(data.view(-1, length, data.size(2), data.size(3)),
											volatile=True)
		rst = net(input_var).data.cpu().numpy().copy()
		return i, rst.reshape((num_crop, test_segments, num_class)).mean(axis=0).reshape(
			(test_segments, 1, num_class)
		), label[0]


	proc_start_time = time.time()
	max_num = len(data_loader.dataset)

	for i, (data, label) in data_gen:
		if i >= max_num:
			break
		rst = eval_video((i, data, label))
		output.append(rst[1:])
		cnt_time = time.time() - proc_start_time
		print('video {} done, total {}/{}, average {} sec/video'.format(i, i+1,
																		total_num,
																		float(cnt_time) / (i+1)))

	video_pred = [np.argmax(np.mean(x[0], axis=0)) for x in output]

	video_labels = [x[1] for x in output]


	cf = confusion_matrix(video_labels, video_pred).astype(float)

	cls_cnt = cf.sum(axis=1)
	cls_hit = np.diag(cf)

	cls_acc = cls_hit / cls_cnt
	
	print('Label: ' + str(video_pred))

	print('Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))

	if save_scores is not None:

		# reorder before saving
		name_list = fileAddr.strip().split()   ## video address

		order_dict = {e:i for i, e in enumerate(sorted(name_list))}

		reorder_output = [None] * len(output)
		reorder_label = [None] * len(output)

		for i in range(len(output)):
			idx = order_dict[name_list[i]]
			reorder_output[idx] = output[i]
			reorder_label[idx] = video_labels[i]

		np.savez(save_scores, scores=reorder_output, labels=reorder_label)

test_models('/data1/zhuhuihui/zhh/tsn_zhh/test_tsn/opencv_image/100_WritingOnBoard_011', 'Flow', '334', 'opencv_ucf101_bninception__flow_model_best.pth.tar',[1], 'score_file_flow_2_zmd_test')
