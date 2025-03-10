# import glob
# import imageio
# # 生成 gif 格式的图片
# env_log_dir = 'logs/picture/picture20231018-171358'
# img_paths = glob.glob(env_log_dir + '/*.png')
# img_paths.sort(key=lambda x: int(x.split('.')[0].split('\\')[-1]))
# gif_images = []
# for path in img_paths:
#     gif_images.append(imageio.imread(path))
# imageio.mimsave(env_log_dir + '/all.gif', gif_images, fps=20)


import numpy as np
d2d_epoch = 'new_logs/d2d_num.npy'

# 提取x和y坐标
d2d_epoch_num = np.load(d2d_epoch)
print(d2d_epoch_num)