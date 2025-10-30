import torch
import numpy as np
import time
import os
import render.plot_3d_global as plot_3d
import moviepy as mp
import matplotlib.pyplot as plt
## 绘制骨骼动画 
def render_motion(data,output_dir, method='fast'):
    fname = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(
        time.time())) + str(np.random.randint(10000, 99999))
    video_fname = fname + '.mp4'
    feats_fname = fname + '.npy'
    output_npy_path = os.path.join(output_dir, feats_fname)
    output_mp4_path = os.path.join(output_dir, video_fname)
    # if method == 'slow':
    #     if len(data.shape) == 4:
    #         data = data[0]
    #     data = data - data[0, 0]
    #     pose_generator = HybrIKJointsToRotmat()
    #     pose = pose_generator(data)
    #     pose = np.concatenate([
    #         pose,
    #         np.stack([np.stack([np.eye(3)] * pose.shape[0], 0)] * 2, 1)
    #     ], 1)
    #     shape = [768, 768]
    #     render = SMPLRender(cfg.RENDER.SMPL_MODEL_PATH)

    #     r = RRR.from_rotvec(np.array([np.pi, 0.0, 0.0]))
    #     pose[:, 0] = np.matmul(r.as_matrix().reshape(1, 3, 3), pose[:, 0])
    #     vid = []
    #     aroot = data[[0], 0]
    #     aroot[:, 1] = -aroot[:, 1]
    #     params = dict(pred_shape=np.zeros([1, 10]),
    #                   pred_root=aroot,
    #                   pred_pose=pose)
    #     render.init_renderer([shape[0], shape[1], 3], params)
    #     for i in range(data.shape[0]):
    #         renderImg = render.render(i)
    #         vid.append(renderImg)

    #     out = np.stack(vid, axis=0)
    #     output_gif_path = output_mp4_path[:-4] + '.gif'
    #     imageio.mimwrite(output_gif_path, out, duration=50)
    #     out_video = mp.VideoFileClip(output_gif_path)
    #     out_video.write_videofile(output_mp4_path)
    #     del out, render
    if method == 'fast':
        output_gif_path = output_mp4_path[:-4] + '.gif'
        if len(data.shape) == 3:
            data = data[None]
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        pose_vis = plot_3d.draw_to_batch(data, [''], [output_gif_path])
        out_video = mp.VideoFileClip(output_gif_path)
        out_video.write_videofile(output_mp4_path)
        del pose_vis

    return output_mp4_path, video_fname, output_npy_path, feats_fname


def plot_3d_point(data, output_dir, fnum):
    # get frame
    data = data[fnum]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 2], data[:, 0], data[:, 1])
    plt.show()
    plt.savefig(os.path.join(output_dir, '3d_point.png'))
def divide_3d_point(data, output_dir, fnum, name):
    # get frame
    data = data[fnum,np.newaxis,:]
    print(data.shape)
    np.save(os.path.join(output_dir, name), data)

if __name__=="__main__":
    # npy_file = r'C:\Users\Y9000P\Desktop\WorkingSpace\motionProcessTools\MotionProcessTools\data\npy_file\TCS01C01B24_joints.npy'
    # h3d_raw_npy_file = r'C:\Users\PC\Desktop\workingSpace\KPT2SkAnimEval\dataset\humanML3D\000000.npy'
    # h3d_hyb_npy_file = r'C:\Users\PC\Desktop\workingSpace\KPT2SkAnimEval\dataset\out\hyberik_humanml3d_reKpt\SMPLPoss_000000_kpt3d.npy'
    # h3d_smplify3d_npy_file = r'C:\Users\PC\Desktop\workingSpace\KPT2SkAnimEval\dataset\out\smplify3d_humanml3d_reKpt\SMPLPose_000000_kpt3d.npy'

    # out_dir_h3d_raw = r"C:\Users\PC\Desktop\workingSpace\KPT2SkAnimEval\view\raw"
    # out_dir_h3d_hyb = r"C:\Users\PC\Desktop\workingSpace\KPT2SkAnimEval\view\hybri"
    # out_dir_h3d_smpl = r"C:\Users\PC\Desktop\workingSpace\KPT2SkAnimEval\view\smpl3d"

    # data_npy_h3d_raw = np.load(h3d_raw_npy_file)
    # data_npy_h3d_hyb = np.load(h3d_hyb_npy_file)
    # data_npy_h3d_smpl = np.load(h3d_smplify3d_npy_file)
    # render_motion(data_npy_h3d_raw, out_dir_h3d_raw, method='fast')
    # render_motion(data_npy_h3d_hyb, out_dir_h3d_hyb, method='fast')
    # render_motion(data_npy_h3d_smpl, out_dir_h3d_smpl, method='fast')

    taichi_raw_npy_file = r"C:\Users\PC\Desktop\workingSpace\KPT2SkAnimEval\dataset\taichi\S01C01B01.npy"

    taichi_data_raw = np.load(taichi_raw_npy_file)
    out_path = r"C:\Users\PC\Desktop\workingSpace\KPT2SkAnimEval\view"

    render_motion(taichi_data_raw, out_path, method='fast')