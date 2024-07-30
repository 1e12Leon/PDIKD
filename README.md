<div align="center">

## Progressive and Domain-invariant Knowledge Distillation for UAV-based Object Detection

[Liang Yao (姚亮)](https://multimodality.group/author/%E5%A7%9A%E4%BA%AE/) 
<img src="assets/hhu_logo.png" alt="Logo" width="15">, &nbsp; &nbsp; 
[Fan Liu (刘凡)](https://multimodality.group/author/%E5%88%98%E5%87%A1/) ✉ 
<img src="assets/hhu_logo.png" alt="Logo" width="15">, &nbsp; &nbsp;
[Chuanyi Zhang (张传一)](https://ai.hhu.edu.cn/2023/0809/c17670a264073/page.htm) 
<img src="assets/hhu_logo.png" alt="Logo" width="15">, &nbsp; &nbsp; 

[Zhiquan Ou (欧志权)](https://multimodality.group/author/%E6%AC%A7%E5%BF%97%E6%9D%83/) 
<img src="assets/hhu_logo.png" alt="Logo" width="15">, &nbsp; &nbsp; 
[Ting Wu (吴婷)](https://multimodality.group/author/%E5%90%B4%E5%A9%B7/) 
<img src="assets/hhu_logo.png" alt="Logo" width="15">, &nbsp; &nbsp; 
[Jun Zhou (周峻)](https://experts.griffith.edu.au/7205-jun-zhou) 
<img src="assets/griffith_logo.png" alt="Logo" width="15">

</div>

![Fig2_4](https://github.com/user-attachments/assets/26aceea6-d277-4468-ba3d-39f6e268cc37)

### News

- **2024/07/29**: We propose a Progressive and Domain-invariant Knowledge Distillation method for UAV-OD. Codes and models will be open-sourced at this repository.

### Experimental Results

| Dataset   | Method                   | Publication  | mAP   | AP_{0.5} | AP_{0.75} | AP_S  | AP_M  | AP_L  |  
|-----------|--------------------------|--------------|-------|----------|-----------|-------|-------|-------|  
|           | YOLOv7-L (T)             | CVPR2023     | 6.94  | 30.26    | **16.89** | 8.17  | **26.90** | **42.41** |  
|           | YOLOv7-Tiny (S)          | CVPR2023     | 11.62 | 21.95    | 11.25     | 4.71  | 18.39 | 32.60 |  
| VisDrone  | FitNets                  | ICLR2015     | 12.87 | 24.71    | 12.34     | 5.21  | 20.62 | 34.77 |  
|           | BCKD                     | ICCV2023     | 16.08 | 30.71    | 15.81     | 8.81 | 24.90 | 26.60 |  
|           | CrossKD                  | CVPR2024     | 14.96 | 29.22    | 14.11     | 8.79  | 23.77 | 24.54 |  
|           | *Ours*                   | -            | **17.07** | **31.92** | 16.77  | **9.56** | 25.90 | 38.98 |  
|-----------|--------------------------|--------------|-------|----------|-----------|-------|-------|-------|  
|           | YOLOv7-L (T)             | CVPR2023     | 33.66 | 64.01    | 30.36     | 18.44 | 42.19 | **40.97** |  
|           | YOLOv7-Tiny (S)          | CVPR2023     | 30.21 | 58.41    | 27.78     | 15.66 | 37.88 | 36.68 |  
| SynDrone  | FitNets                  | ICLR2015     | 32.12 | 61.73    | 29.32     | 16.82 | 40.21 | 39.38 |  
|           | BCKD                     | ICCV2023     | 32.66 | 63.12    | 29.95     | 21.52 | 40.52 | 31.93 |  
|           | CrossKD                  | CVPR2024     | 31.79 | 61.85    | 28.46     | 17.84 | 39.53 | 30.88 |  
|           | *Ours*                   | -            | **35.12** | **65.09** | **33.30** | **22.27** | **43.08** | 36.51 |

### Citation


### Contact
Please Contact [yaoliang@hhu.edu.cn](yaoliang@hhu.edu.cn)
