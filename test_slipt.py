'''
Args:
input_dir(abs_path):for example,/home/zi/input_dir
output_dir(abs_path):for example,/home/zi/output_dir
Usage:
python input_dir output_dir
Attention:
input_dir and outdir_dir must be different
'''
import os
import cv2
import sys
def slip_video(video_path,video_file,label):
    count = 0
    cap = cv2.VideoCapture(video_file)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    results = video_file + " " + str(num_frames) + " " + label + "\n"
    f.write(results)
    ret, pre_frame = cap.read()
    prvs = cv2.cvtColor(pre_frame, cv2.COLOR_BGR2GRAY)
    while True:
        flag, frame = cap.read()
        if flag:
            count = count+1
            rgb = video_path+"/"+"img_%05d.jpg" %(count)
            flow_x = video_path+"/"+"flow_x_%05d.jpg"%(count)
            flow_y = video_path+"/"+"flow_y_%05d.jpg"%(count)
            next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            cv2.normalize(flow, flow, 0, 255, cv2.NORM_MINMAX)
            prvs = next
            cv2.imwrite(rgb,frame)
            cv2.imwrite(flow_x,flow[:,:,0])
            cv2.imwrite(flow_y, flow[:,:,1])
        else:
            break

if __name__ == "__main__":
    out_dir = sys.argv[2]
    datasets_dir = sys.argv[1]
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    if not os.path.exists(datasets_dir):
        raise Exception("NO DIR")
    labelbs_txt = out_dir + "/" + "labels_txt"
    f = open(labelbs_txt, "w")
    video_dir = os.listdir(datasets_dir)
    for i in video_dir:
        video_abs_path = os.path.join(datasets_dir, i)
        if os.path.isdir(video_abs_path):
            video = os.listdir(video_abs_path)
            for j in range(0, len(video)):
                video_file = os.path.join(video_abs_path, video[j])
                label = i.split("_")[1]
                v_dir = video[j].split(".")[0]
                # video_path = os.path.join(video_abs_path,v_dir)
                video_path = os.path.join(out_dir, v_dir)
                if not os.path.exists(video_path):
                    os.mkdir(video_path)
                slip_video(video_path,video_file,label)
    f.close()


