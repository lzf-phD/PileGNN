import pandas as pd
import numpy as np
import torch
from PIL import Image
import os
import torchvision.transforms as transforms

# ------------------------Input----------------------------
txt_name = "Q1ZK123.txt"
parameter_path = r"../parameters_table.xlsx"   #excel(*xlsx*) parameters table
#------------------------------------------------------------
edge_label_path = '../data_EB/test_B'
edge_pred_path = '../result/pile_EB'
label_path = os.path.join(edge_label_path, txt_name)
pred_path = os.path.join(edge_pred_path, txt_name)
# ------------------------reading data----------------------------
label_lines = np.loadtxt(label_path, dtype=str, encoding="utf-8").tolist()

def get_trans_label(label_lines):
    txt = list(map(float, label_lines.split(',')))
    return txt
#y = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.55, 0]

y_real = get_trans_label(label_lines)
y = np.loadtxt(pred_path, dtype=float, encoding="utf-8").tolist()

data = pd.read_excel(parameter_path, header=None, sheet_name=0, dtype=str, engine='openpyxl')
data_origin = pd.read_excel(parameter_path, header=None, sheet_name=0, dtype=str, engine='openpyxl')

first_col = data.iloc[:87, 0].astype(str).tolist()
image_name = txt_name.replace('_qik.txt', '')
image_name = txt_name.replace('_k.txt', '')
image_name = txt_name.replace('.txt', '')
co_efficient_v = 1024/90
co_efficient_h = 64/14.098

fill_row = list(range(6, 13)) #6-12 255
silt_row = list(range(13, 25))#13-24 205
clay_row = list(range(25, 28))+list(range(56, 62))#25-27+56-61 155
sand_row = list(range(28, 56))#28-55 105
rock_row = list(range(62, 74))#62-73 55

for col in range(2000):
    gama_list = []
    fa0_list = []
    qik_list = []
    frk_list = []
    hi_list = []
    category = []
    green_label = []
    zk_col = col+9
    thickness_col = data.iloc[:80,zk_col].astype(str).tolist()
    origin_col = data_origin.iloc[:81, zk_col].astype(str).tolist()
    zk_name = origin_col[-1]
    if thickness_col[0] != 'nan' and zk_name == image_name:
        top_elevation = -float(thickness_col[-6])
        K = thickness_col[-2]
        pile_diameter = float(thickness_col[-4])
        pile_type = 'end-bearing pile'
        zk_name = origin_col[-1]
        for i in range(5, 73):
            zk_value = data.iloc[i, zk_col]
            if pd.notna(zk_value) and zk_value != 'nan':
                gama_list.append(data.iloc[i, 3])
                fa0_list.append(data.iloc[i, 5])
                qik_list.append(data.iloc[i, 7])
                frk_list.append(data.iloc[i, 8])
                hi_list.append(pd.to_numeric(zk_value, errors='coerce'))
                if i + 1 in fill_row:
                    green_label.append([235, 244, 229])
                elif i + 1 in silt_row:
                    green_label.append([191, 221, 173])
                elif i + 1 in clay_row:
                    green_label.append([158, 203, 127])
                elif i + 1 in sand_row:
                    green_label.append([84, 130, 53])
                elif i + 1 in rock_row:
                    green_label.append([56, 87, 35])
        soil_pixel_horizon = 64
        soil_pixel_vertical = [round(co_efficient_v*hi) for hi in hi_list]
        pile_pixel_horizon = 0
        if pile_diameter == 1:
            pile_pixel_horizon = 23
        elif pile_diameter == 1.2:
            pile_pixel_horizon = 28
        elif pile_diameter == 1.5:
            pile_pixel_horizon = 34
        elif pile_diameter == 1.8:
            pile_pixel_horizon = 41
        elif pile_diameter == 2:
            pile_pixel_horizon = 46
        elif pile_diameter == 2.5:
            pile_pixel_horizon = 57
        else:
            pile_pixel_horizon = round(4.5511*pile_diameter*5)
        image_source = torch.ones((3, 1024, 512))
        image_label = torch.ones((3, 1024, 512))
        image_cond = torch.ones((3, 1024, 512))
        image_real = torch.ones((3, 1024, 512))
        if len(hi_list) == len(y):
            pile_length = round(sum(a * b for a, b in zip(y, hi_list)),1)
            real_length = round(sum(a * b for a, b in zip(y_real, hi_list)), 1)
        else:
            break
        H_sum = [0]
        for i in range(len(hi_list)):
            H_add = soil_pixel_vertical[i]+H_sum[-1]
            H_sum.append(H_add)
        H0 = 512-H_sum[-1]/2-1
        for i in range(len(hi_list)):
          soil_green_R = green_label[i][0]
          soil_green_G = green_label[i][1]
          soil_green_B = green_label[i][2]
          mask_RGB = 255-i*20
          y2 = H0+H_sum[i]
          y1 = H0+H_sum[i+1]
          image_source[0, int(y2):int(y1), 214:278] = soil_green_R/255
          image_source[1, int(y2):int(y1), 214:278] = soil_green_G/255
          image_source[2, int(y2):int(y1), 214:278] = soil_green_B/255
          image_label[0, int(y2):int(y1), 214:278] = soil_green_R/255
          image_label[1, int(y2):int(y1), 214:278] = soil_green_G/255
          image_label[2, int(y2):int(y1), 214:278] = soil_green_B/255
          image_real[0, int(y2):int(y1), 214:278] = soil_green_R/255
          image_real[1, int(y2):int(y1), 214:278] = soil_green_G/255
          image_real[2, int(y2):int(y1), 214:278] = soil_green_B/255
          image_cond[0, int(y2):int(y1), 214:278] = 0
          image_cond[1, int(y2):int(y1), 214:278] = mask_RGB / 255
          image_cond[2, int(y2):int(y1), 214:278] = 0
        tip_elevation = - pile_length
        tip_real = - real_length
        pile_x1 = int(278)
        pile_x2 = int(278+pile_pixel_horizon)
        pile_y1 = int(H0-round(top_elevation*co_efficient_v))
        pile_y2 = int(H0-round(tip_elevation*co_efficient_v))
        pile_y2_2 = int(H0 - round(tip_real * co_efficient_v))
        soil_tip = int(H0+H_sum[-1])
        #train_A
        image_source[0,pile_y1:soil_tip:,pile_x1:pile_x2] = 132/255
        image_source[1,pile_y1:soil_tip,pile_x1:pile_x2] = 132/255
        image_source[2,pile_y1:soil_tip,pile_x1:pile_x2] = 132/255
        #train_B
        if pile_type=='end-bearing pile':
            image_label[0,pile_y1:pile_y2,pile_x1:pile_x2] = 1
            image_label[1,pile_y1:pile_y2,pile_x1:pile_x2] = 0
            image_label[2,pile_y1:pile_y2,pile_x1:pile_x2] = 0
            image_label[0,pile_y2:soil_tip,pile_x1:pile_x2] = 132/255
            image_label[1,pile_y2:soil_tip,pile_x1:pile_x2] = 132/255
            image_label[2,pile_y2:soil_tip,pile_x1:pile_x2] = 132/255

            image_real[0, pile_y1:pile_y2_2, pile_x1:pile_x2] = 1
            image_real[1, pile_y1:pile_y2_2, pile_x1:pile_x2] = 0
            image_real[2, pile_y1:pile_y2_2, pile_x1:pile_x2] = 0
            image_real[0, pile_y2_2:soil_tip, pile_x1:pile_x2] = 132 / 255
            image_real[1, pile_y2_2:soil_tip, pile_x1:pile_x2] = 132 / 255
            image_real[2, pile_y2_2:soil_tip, pile_x1:pile_x2] = 132 / 255
        else:
            image_label[0, pile_y1:pile_y2, pile_x1:pile_x2] = 0
            image_label[1, pile_y1:pile_y2, pile_x1:pile_x2] = 0
            image_label[2, pile_y1:pile_y2, pile_x1:pile_x2] = 1
            image_label[0, pile_y2:soil_tip, pile_x1:pile_x2] = 132 / 255
            image_label[1, pile_y2:soil_tip, pile_x1:pile_x2] = 132 / 255
            image_label[2, pile_y2:soil_tip, pile_x1:pile_x2] = 132 / 255

            image_real[0, pile_y1:pile_y2_2, pile_x1:pile_x2] = 0
            image_real[1, pile_y1:pile_y2_2, pile_x1:pile_x2] = 0
            image_real[2, pile_y1:pile_y2_2, pile_x1:pile_x2] = 1
            image_real[0, pile_y2_2:soil_tip, pile_x1:pile_x2] = 132 / 255
            image_real[1, pile_y2_2:soil_tip, pile_x1:pile_x2] = 132 / 255
            image_real[2, pile_y2_2:soil_tip, pile_x1:pile_x2] = 132 / 255
        #cond/train
        image_cond[0,pile_y1:soil_tip,pile_x1:pile_x2] = 132/255
        image_cond[1,pile_y1:soil_tip,pile_x1:pile_x2] = 132/255
        image_cond[2,pile_y1:soil_tip,pile_x1:pile_x2] = 132/255
        #-------------------------------------printing----------------------------------
        to_pil = transforms.ToPILImage()
        image_source = to_pil(image_source)
        image_source.save(f"../result/PNG/{image_name}_input.png")
        image_label = to_pil(image_label)
        image_label.save(f"../result/PNG/{image_name}_syn.png")
        image_real = to_pil(image_real)
        image_real.save(f"../result/PNG/{image_name}_real.png")
        image_cond = to_pil(image_cond)
        image_cond.save(f"../result/PNG/{image_name}_mask.png")
        print(f"{image_name} print finished")
        print(y)
        print(y_real)
        break
