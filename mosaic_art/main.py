import argparse
import copy
import glob
import logging
import math
import os
import time

import cv2
import joblib
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def find(path: str, pattern: str) -> list:
    files = glob.glob(os.path.join(path, "{}".format(pattern)))
    files.sort()

    return files


# 目標画像のタイル化
def create_tile(img, height: int, width: int) -> list:
    w, h = img.size
    rect_dict = list()
    for y in tqdm(range(0, h // height + 1)):
        for x in range(0, w // width + 1):
            height2 = y * height
            width2 = x * width
            crap_img = img.crop((width2, height2, width2 + width, height2 + height))
            hist = calc_hist(crap_img)
            rect_dict.append((hist, width2, height2))
    return rect_dict


# ヒストグラムの計算(RGBそれぞれを計算)
def calc_hist(img, tile_factor=(1, 1)):
    color = ["r", "g" "b"]
    images = np.asarray(img)
    hist_list = [cv2.calcHist([images], [c[0]], None, [256], [0, 256]) for c in enumerate(color)]
    hist = np.array(hist_list)
    hist = hist.reshape(hist.shape[0] * hist.shape[1], 1)
    hist = np.tile(hist, tile_factor)

    return hist


def is_greanback(img, threshold=200):
    rgb = np.array(img)[0:40, -40:-1].mean(0).mean(0)
    if rgb[1] - rgb[0] > threshold or rgb[1] - rgb[2] > threshold:
        return True
    else:
        return False


def calc_parallel(path: str, resize: tuple):
    try:
        img = Image.open(path).resize(resize)

        if is_greanback(img):
            return (None, None)
        else:
            # path: [img, hist, use_counter]
            return (path, [img, calc_hist(img), 0])
    except Exception:
        return (None, None)


def __compare_gpu(idx: int, src: np.array, tar: np.array):
    tar = torch.from_numpy(tar).to(DEVICE)
    tar_dump = torch.tile(tar, (1, 1, len(src)))
    src = torch.from_numpy(np.hstack(src)).to(DEVICE).unsqueeze(dim=0)
    src_hists = torch.tile(src, (tar.size(0), 1, 1))

    return (idx, torch.sum(torch.minimum(tar_dump, src_hists), dim=1).cpu().numpy())


def compare(src: list, tar: list):
    num_split = len(tar) // 30
    src_split = np.array_split(src, num_split)
    tar_hists = np.stack(tar, axis=0)

    result = list()

    result_dict = dict()
    result = [__compare_gpu(i, src, tar_hists) for i, src in enumerate(tqdm(src_split))]
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    result_dict.update(result)
    result_sorted = sorted(result_dict.items(), key=lambda x: x[0])
    result_dict.clear()
    result_dict.update(result_sorted)

    return np.concatenate(list(result_dict.values()), axis=1)


def generate(n_jobs):
    src = [hist[1] for hist in src_hist_dict.values()]
    tar = [hist[0] for hist in tile_hist_ls]
    logger.info("Start compare...")
    result = compare(src, tar)

    choice_idx = np.random.choice(result.shape[0], result.shape[0], replace=False).tolist()

    _ = joblib.Parallel(n_jobs=n_jobs, require="sharedmem")(
        joblib.delayed(__generate_parallel)(idx, result) for idx in tqdm(choice_idx)
    )


def __generate_parallel(idx: int, result: np.array):
    delete_idx = copy.deepcopy(remove_idx)

    dump = np.delete(result[idx,], delete_idx).tolist()
    fname_list = np.delete(list(src_hist_dict.keys()), delete_idx)
    fname = fname_list[dump.index(np.max(dump))]

    target.paste(src_hist_dict[fname][0], (tile_hist_ls[idx][1], tile_hist_ls[idx][2]))
    src_hist_dict[fname][2] += 1
    if src_hist_dict[fname][2] == used_count:
        remove_idx.append(list(src_hist_dict.keys()).index(fname))


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", type=str, default="./target.png", help="ピクセルアートのリファレンス画像")
    parser.add_argument("-s", "--src_dir", type=str, default="./src", help="ソースディレクトリ")
    parser.add_argument("-o", "--output", type=str, default="./output.png", help="出力ファイル名")
    parser.add_argument("-j", "--jobs", type=int, default=-1, help="使用するCPU数")
    parser.add_argument("--src_find_pattern", type=str, default="**/*.png", help="ソース画像を検索する時のパターン")
    parser.add_argument("--used_count", type=int, default=3, help="1つの画像を何回まで使えるか")
    parser.add_argument("--zoom", type=float, default=3, help="目標画像の倍率")
    parser.add_argument("--resize", type=int, default=(60, 40), nargs="+", help="ソース画像のリサイズサイズ")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    args = parser.parse_args()

    global target
    global tile_hist_ls
    global src_hist_dict
    global remove_idx
    global used_count
    global DEVICE

    DEVICE = args.device

    used_count = args.used_count

    resize = tuple(args.resize)

    s = time.time()

    tar_img = Image.open(args.target)
    tar_img = tar_img.resize((math.floor(i * args.zoom) for i in tar_img.size))
    target = Image.new("RGB", tar_img.size, 255)

    logger.info("Start Tiling...")
    tile_hist_ls = create_tile(tar_img, resize[1], resize[0])
    logger.info("Finish Tiling! | time: {:4f}".format(time.time() - s))

    file_paths = find(args.src_dir, args.src_find_pattern)
    src_hist_dict = dict()
    sh = time.time()
    logger.info("Start Calc hist...")
    result = joblib.Parallel(n_jobs=args.jobs)(joblib.delayed(calc_parallel)(path, resize) for path in tqdm(file_paths))
    while True:
        try:
            result.remove((None, None))
        except:
            break

    src_hist_dict.update(result)
    logger.info("Finish Calc hist! | time: {:4f}".format(time.time() - sh))

    target = Image.new("RGB", tar_img.size, 255)
    remove_idx = list()

    sg = time.time()
    logger.info("Start Generate image...")
    generate(args.jobs)
    # map(gen_parallel, tiles)
    # for tar_tile in tqdm(tiles):
    #     gen_parallel(tar_tile)
    # _ = joblib.Parallel(n_jobs=args.jobs, require='sharedmem')(joblib.delayed(gen_parallel)(tar_tile) for tar_tile in tqdm(tiles))
    eg = time.time()
    logger.info("Finish Generate image! | time: {:4f}".format(eg - sg))

    target.save(args.output)
    logger.info("All Finish! | time: {:4f}".format(eg - s))


if __name__ == "__main__":
    main()
