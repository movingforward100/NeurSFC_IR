import os
import numpy as np
from glob import glob
from PIL import Image
from sklearn.cluster import MiniBatchKMeans

def compute_centroids(image_folder, n_clusters=256, max_images=None, per_image_pixels=5000,
                      resize_to=None, seed=42):
    paths = sorted(glob(os.path.join(image_folder, '*.*')))
    rng = np.random.default_rng(seed)
    if max_images is not None:
        idx = rng.choice(len(paths), size=min(max_images, len(paths)), replace=False)
        paths = [paths[i] for i in idx]

    samples = []
    for p in paths:
        im = Image.open(p).convert('RGB')
        if resize_to is not None:
            im = im.resize((resize_to, resize_to), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.uint8).reshape(-1, 3)
        k = min(per_image_pixels, arr.shape[0])
        pick = rng.choice(arr.shape[0], size=k, replace=False)
        samples.append(arr[pick])
    X = np.concatenate(samples, axis=0).astype(np.float32)

    print(X.shape)

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, batch_size=4096)
    kmeans.fit(X)
    return kmeans.cluster_centers_.astype(np.float32)


root = 'dataset/GoPro'

input_folder = os.path.join(root, 'train', 'LQ_crops_64x64_skip_128')
save_path = os.path.join(root, 'train_crpo_64x64_skip_128_centroids.npy')

centroids = compute_centroids(
    input_folder,
    n_clusters=256,
    max_images=None,          
    per_image_pixels=200,   # 每图采样 200 像素, 根据数据集作调整。很多时候crop之后的图片数已经是100K+, 减少per_image_pixels，比如50, 否则机器卡死
    resize_to=None   
)

print(centroids.shape)
C = centroids
print("min/max per channel:", C.min(0), C.max(0))
lum = 0.299*C[:,0] + 0.587*C[:,1] + 0.114*C[:,2]   # 亮度
print("lum p5/median/p95:", np.percentile(lum,[5,50,95]))
print("暗部占比(lum<20):", (lum<20).mean())
print("是否有NaN:", np.isnan(C).any())

np.save(save_path, centroids)
