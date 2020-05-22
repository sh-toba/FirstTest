import cv2, os, glob, json, datetime, shutil, re, sys
import numpy as np
import numpy.matlib
import pandas as pd
from pandas import DataFrame
from pandas import ExcelWriter
import openpyxl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
#from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy
from joblib import Parallel, delayed

from scipy.spatial import Delaunay, Voronoi
import scipy.ndimage

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import Utilities.utility as utl

class ColonyPickup:

    #TODO: 200521, ぼかして大津の2値化で対応可能ではないか？

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    SETTING_FILE = os.path.join(ROOT_DIR, "settings.json")

    INPUT_DIR = os.path.join(ROOT_DIR, "__input__")
    DONE_DIR = os.path.join(ROOT_DIR, "__done__")
    OUTPUT_DIR = os.path.join(ROOT_DIR, "__output__")
    MISC_DIR = "__misc__"
    MASK_FILE_POSTFIX = "_Mask.png"

    target_name = ""
    dst_root = OUTPUT_DIR
    dst_path = ""
    misc_path = ""
    colony_data = {
        "ColonyID":[],
        "ColonyArea":[],
        "BBOX":[],
        "IntensityMax":[],
        "IntensityMin":[],
        "IntensityAvg":[],
        "IntensityStd":[]
    }

    settings = {
        "Boarder": 0.95,
        "PaddingSize": 32,
        "ThresholdType": "Otsu",
        "BackGroundValue": 5,
        "KernelSizeBlur": 31,
        "KernelSizeDilate": 11,
        "MinimumColonyPixelSize": 40000,
        "ManualPickup": False,
        "ShowResult": False,
        "ShowTileSize": 3
    }

    def __init__(self):
        self.load_setting()
        self.dir_setting()

    def load_setting(self, show=True):
    
        if not os.path.exists(self.SETTING_FILE):
            with open(self.SETTING_FILE, 'w') as fw:
                json.dump(self.settings, fw, ensure_ascii=False, indent=4, sort_keys=False, separators=(',', ': '))
        else:
            with open(self.SETTING_FILE, 'r') as fr:
                self.settings = json.load(fr)
        
        print('\n=====[Now Settings]=====')
        print(json.dumps(self.settings, ensure_ascii=False, indent=4, sort_keys=False, separators=(',', ': ')))
        print('========================')

    def dir_setting(self):
        dir_list = [self.INPUT_DIR, self.DONE_DIR, self.OUTPUT_DIR]
        for dl in dir_list:
            if not os.path.exists(dl):
                os.makedirs(dl, exist_ok=True)

    def search_target(self):
        target_list = []
        for fl in glob.glob(os.path.join(self.INPUT_DIR, "*.tif")):
            target_list.append(os.path.splitext(os.path.basename(fl))[0])

        return target_list

    def set_target(self, target_name):

        self.target_name = target_name
        self.dst_path = os.path.join(self.dst_root, target_name)
        os.mkdir(self.dst_path)
        self.misc_path = os.path.join(self.dst_path, self.MISC_DIR)
        os.mkdir(self.misc_path)

        self.colony_data = {
            "ColonyID":[],
            "ColonyArea":[],
            "BBOX":[],
            "IntensityMax":[],
            "IntensityMin":[],
            "IntensityAvg":[],
            "IntensityStd":[]
        }

        return

    def preprocess(self, img):

        KERNEL_SIZE_BLUR = self.settings["KernelSizeBlur"]
        THRESHOLD_TYPE = self.settings["ThresholdType"]
        BACKGROUND_VALUE = self.settings["BackGroundValue"]
        KERNEL_SIZE_DILATE = self.settings["KernelSizeDilate"]
        SHOW_RESULT = self.settings["ShowResult"]

        h, w = img.shape[:2]
        #plt.imshow(img, cmap='gray')
        #plt.show()

        # ブラー
        print('>> Blur...')
        #img_blur = cv2.GaussianBlur(img_norm,(KERNEL_SIZE,KERNEL_SIZE),0)
        img_blur = cv2.blur(img, (KERNEL_SIZE_BLUR,KERNEL_SIZE_BLUR))
        # 二値化
        print('>> Threshold...')
        if THRESHOLD_TYPE == "Otsu":
            _, mask_org = cv2.threshold(img_blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        elif THRESHOLD_TYPE == "Manual":
            _, mask_org = cv2.threshold(img_blur,BACKGROUND_VALUE,255,cv2.THRESH_BINARY)
        # ごみ除去
        kernel = np.ones((KERNEL_SIZE_DILATE,KERNEL_SIZE_DILATE),np.uint8)
        #print('>> Closing...')
        #mask_close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        #print('>> Opening...')
        #mask_open = cv2.morphologyEx(mask_close, cv2.MORPH_OPEN, kernel)
        #print('>> Dilation...')
        #mask_dilate = cv2.dilate(mask_org,kernel,iterations = 1)
        print('>> Fill Hole...')
        mask_fill = utl.fill_hole(mask_org)

        # 確認表示
        if SHOW_RESULT:
            TILE_SIZE = self.settings["ShowTileSize"]

            tile_w = int(w/TILE_SIZE)
            tile_h = int(h/TILE_SIZE)
            tile_loc = round(TILE_SIZE/2)-1
            w_st = tile_w*tile_loc
            w_ed = w_st + tile_w
            h_st = tile_h*tile_loc
            h_ed = h_st + tile_h
            #utl.get_overlay(img_norm[h_st:h_ed, w_st:w_ed], mask_dilate[h_st:h_ed, w_st:w_ed])
            plt.subplot(231),plt.title('Original'),plt.imshow(img[h_st:h_ed, w_st:w_ed], cmap = 'gray'),plt.axis(False)
            plt.subplot(233),plt.title('Blur'),plt.imshow(img_blur[h_st:h_ed, w_st:w_ed], cmap = 'gray'),plt.axis(False)
            plt.subplot(234),plt.title('Threshold'),plt.imshow(mask_org[h_st:h_ed, w_st:w_ed], cmap = 'gray'),plt.axis(False)
            #plt.subplot(235),plt.title('Dilation'),plt.imshow(mask_dilate[h_st:h_ed, w_st:w_ed], cmap = 'gray'),plt.axis(False)
            plt.subplot(236),plt.title('Fill Hole'),plt.imshow(mask_fill[h_st:h_ed, w_st:w_ed], cmap = 'gray'),plt.axis(False)
            plt.show()

        return mask_fill

    def save_target_roi(self, savename, img):
        
        N = len(self.colony_data["ColonyID"])

        # オブジェクト情報を利用してラベリング結果を画面に表示
        for n in range(N):

            colony_id = self.colony_data["ColonyID"][n]

            # 各オブジェクトの外接矩形を赤枠で表示
            bbox = self.colony_data["BBOX"][n]
            x0 = bbox[0]
            y0 = bbox[1]
            x1 = bbox[2]
            y1 = bbox[3]
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), thickness=8)

            # 各オブジェクトのラベル番号と面積に黄文字で表示
            cv2.putText(img, "ID:" +str(colony_id), (x0, y1), cv2.FONT_HERSHEY_PLAIN, 12, (0, 255, 0), thickness=8)
        
            #cv2.imshow("color_src01", color_src01)
            cv2.imwrite(savename, img)

        return

    def extract_colony(self, img, mask):

        BOARDER = self.settings["Boarder"]
        PADDING_SIZE = self.settings["PaddingSize"]

        h, w = img.shape[:2]

        # クロップ
        trim_rate = (1 - BOARDER)/2
        crop_size = [int(w * BOARDER), int(h * BOARDER)]
        crop_w_st = int(w * trim_rate)
        crop_h_st = int(h * trim_rate)
        crop_w_ed = crop_w_st + crop_size[0]
        crop_h_ed = crop_h_st + crop_size[1]

        # 輪郭抽出
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        colony_index = 0
        for i, contour in enumerate(contours):

            print('Colony {}/{} | '.format(str(i+1).zfill(5), str(len(contours)).zfill(5)), end='')
        
            # 重心位置計算
            pts_array, grav = utl.cvtContourInfo(contour)

            # 単一コロニーマスクの作成
            tmp_colony_mask = utl.fill_contour(mask.shape, pts_array, 1)
            tmp_colony_mask = np.where((mask != 0) & (tmp_colony_mask != 0), 1, 0)
            (left, right, top, bottom) = utl.zero_trimming(tmp_colony_mask, grav)
            left -= PADDING_SIZE
            right += PADDING_SIZE
            top -= PADDING_SIZE
            bottom += PADDING_SIZE

            # コロニーサイズ
            colony_area = np.count_nonzero(tmp_colony_mask)

            # 境界条件
            bb_condition = np.array([
                crop_w_st <= left, left < crop_w_ed,
                crop_w_st <= right, right < crop_w_ed,
                crop_h_st <= top, top < crop_h_ed,
                crop_h_st <= bottom, bottom < crop_h_ed
                ])
            if not np.all(bb_condition):
                print('limit of area. colony {} is skipped.'.format(str(i+1).zfill(5)))
                continue

            colony = deepcopy(img[top:bottom+1, left:right+1])
            colony_mask = deepcopy(tmp_colony_mask[top:bottom+1, left:right+1])
            colony_mask = np.array(colony_mask, dtype=np.uint8)

            colony_index += 1

            fname_prefix = '{}_Colony{}'.format(self.target_name, str(colony_index).zfill(4))

            a = colony[colony_mask!=0]
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.hist(a, bins=256, density=True)
            ax.set_title('Colony{}'.format(str(colony_index).zfill(4)))
            ax.set_xlabel('intensity')
            ax.set_ylabel('freq')
            #plt.show()
            plt.savefig(os.path.join(self.misc_path, '{}_Hist.png'.format(fname_prefix)))

            self.colony_data["ColonyID"].append(colony_index)
            self.colony_data["ColonyArea"].append(colony_area)
            self.colony_data["BBOX"].append((left, top, right, bottom))
            self.colony_data["IntensityMax"].append(np.max(a))
            self.colony_data["IntensityMin"].append(np.min(a))
            self.colony_data["IntensityAvg"].append(np.average(a))
            self.colony_data["IntensityStd"].append(np.std(a))
            
            filename = os.path.join(self.dst_path, '{}.tif'.format(fname_prefix))
            cv2.imwrite(filename, colony)
            filename = os.path.join(self.dst_path, '{}_Mask.png'.format(fname_prefix))
            cv2.imwrite(filename, colony_mask*255)

            print("")

            #plt.imshow(utl.get_overlay(colony, colony_mask))

        if colony_index > 0:
            savename = os.path.join(self.misc_path, '{}_Label.png'.format(self.target_name))
            color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(color_img, (crop_w_st, crop_h_st), (crop_w_ed, crop_h_ed), (0, 0, 255), thickness=8)
            self.save_target_roi(savename, color_img)
            data_pd = pd.DataFrame(self.colony_data)
            data_pd.to_csv(os.path.join(self.misc_path, '{}_ColonyList.csv'.format(self.target_name)), index = False)

        return

    def pickup_colony(self):

        COLONY_AREA_MIN = self.settings["MinimumColonyPixelSize"]

        # 画像読み込み
        print('>> Load Image...', self.target_name)
        img = cv2.imread(os.path.join(self.INPUT_DIR, self.target_name + ".tif"), -1)
        if img is None:
            print('Failed to load.')
            return

        if img.ndim == 3:
            b = img[:,:,0]
            g = img[:,:,1]
            r = img[:,:,2]
            img = b + g + r

        # 8bit変換
        if img.dtype != 'uint8':
            print('>> Convert 8bit...')
            img = utl.normalization_byImageJ(img)
        else:
            img = deepcopy(img)
        
        # 前処理
        mask = self.preprocess(img)

        # 作成マスクの保存
        filename = os.path.join(self.misc_path, '{}_Mask0.png'.format(self.target_name))
        cv2.imwrite(filename, mask)

        # 小領域の除去
        print('>> Remove Small Area...')
        #mask_crop = remove_small_colony(mask_crop, area_thres=COLONY_AREA_MIN)
        _, mask_label = cv2.connectedComponents(mask)
        label, count = np.unique(mask_label,return_counts=True)
        area_count = len(label)
        for l, c in zip(label, count):
            if c <= COLONY_AREA_MIN:
                print('  ({}/{}) remove label.'.format(l, area_count))
                mask[mask_label == l] = 0

        filename = os.path.join(self.misc_path, '{}_Mask2_Refine.png'.format(self.target_name))
        cv2.imwrite(filename, mask)

        # コロニー画像抽出
        self.extract_colony(img, mask)

        return

    def run(self):

        # ターゲット画像の検索
        target_list = []
        while(True):
            target_list = self.search_target()
            if len(target_list) == 0:
                print('>> !!! Not Found. the target !!! | Path: ', self.INPUT_DIR)
            else:
                print(">> Found targets. followings: ")
                print("  ", target_list)

            print(">> Please push any keys [ 'y': start pickup | 'q': quit | other keys: re-search target]")
            key = input()
            if key == 'y':
                break
            if key == 'q':
                return

        # 出力フォルダの作成
        self.dst_root = os.path.join(self.OUTPUT_DIR, utl.now())
        os.mkdir(self.dst_root)

        # 設定ファイルのコピー
        shutil.copy(self.SETTING_FILE, os.path.join(self.dst_root, "settings.json"))

        # コロニー画像のピックアップ
        for tl in target_list:
            self.set_target(tl)
            self.pickup_colony()

        return

if __name__ == "__main__":

    cp = ColonyPickup()
    cp.run()