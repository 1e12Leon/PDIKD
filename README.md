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

<table>  
    <thead>  
        <tr>  
            <th>Dataset</th>  
            <th>Method</th>  
            <th>Publication</th>  
            <th>mAP</th>  
            <th>AP_{0.5}</th>  
            <th>AP_{0.75}</th>  
            <th>AP_S</th>  
            <th>AP_M</th>  
            <th>AP_L</th>  
        </tr>  
    </thead>  
    <tbody>  
        <tr>  
            <td></td>  
            <td>YOLOv7-L (T)</td>  
            <td>CVPR2023</td>  
            <td>6.94</td>  
            <td>30.26</td>  
            <td><strong>16.89</strong></td>  
            <td>8.17</td>  
            <td><strong>26.90</strong></td>  
            <td><strong>42.41</strong></td>  
        </tr>  
        <tr>  
            <td></td>  
            <td>YOLOv7-Tiny (S)</td>  
            <td>CVPR2023</td>  
            <td>11.62</td>  
            <td>21.95</td>  
            <td>11.25</td>  
            <td>4.71</td>  
            <td>18.39</td>  
            <td>32.60</td>  
        </tr>  
        <tr>  
            <td>VisDrone</td>  
            <td>FitNets</td>  
            <td>ICLR2015</td>  
            <td>12.87</td>  
            <td>24.71</td>  
            <td>12.34</td>  
            <td>5.21</td>  
            <td>20.62</td>  
            <td>34.77</td>  
        </tr>  
        <tr>  
            <td></td>  
            <td>BCKD</td>  
            <td>ICCV2023</td>  
            <td>16.08</td>  
            <td>30.71</td>  
            <td>15.81</td>  
            <td>8.81</td>  
            <td>24.90</td>  
            <td>26.60</td>  
        </tr>  
        <tr>  
            <td></td>  
            <td>CrossKD</td>  
            <td>CVPR2024</td>  
            <td>14.96</td>  
            <td>29.22</td>  
            <td>14.11</td>  
            <td>8.79</td>  
            <td>23.77</td>  
            <td>24.54</td>  
        </tr>  
        <tr>  
            <td></td>  
            <td><em>Ours</em></td>  
            <td>-</td>  
            <td><strong>17.07</strong></td>  
            <td><strong>31.92</strong></td>  
            <td>16.77</td>  
            <td><strong>9.56</strong></td>  
            <td>25.90</td>  
            <td>38.98</td>  
        </tr>  
        <tr>  
            <td></td>  
            <td>YOLOv7-L (T)</td>  
            <td>CVPR2023</td>  
            <td>33.66</td>  
            <td>64.01</td>  
            <td>30.36</td>  
            <td>18.44</td>  
            <td>42.19</td>  
            <td><strong>40.97</strong></td>  
        </tr>  
        <tr>  
            <td></td>  
            <td>YOLOv7-Tiny (S)</td>  
            <td>CVPR2023</td>  
            <td>30.21</td>  
            <td>58.41</td>  
            <td>27.78</td>  
            <td>15.66</td>  
            <td>37.88</td>  
            <td>36.68</td>  
        </tr>  
        <tr>  
            <td>SynDrone</td>  
            <td>FitNets</td>  
            <td>ICLR2015</td>  
            <td>32.12</td>  
            <td>61.73</td>  
            <td>29.32</td>  
            <td>16.82</td>  
            <td>40.21</td>  
            <td>39.38</td>  
        </tr>  
        <tr>  
            <td></td>  
            <td>BCKD</td>  
            <td>ICCV2023</td>  
            <td>32.66</td>  
            <td>63.12</td>  
            <td>29.95</td>  
            <td>21.52</td>  
            <td>40.52</td>  
            <td>31.93</td>  
        </tr>  
        <tr>  
            <td></td>  
            <td>CrossKD</td>  
            <td>CVPR2024</td>  
            <td>31.79</td>  
            <td>61.85</td>  
            <td>28.46</td>  
            <td>17.84</td>  
            <td>39.53</td>  
            <td>30.88</td>  
        </tr>  
        <tr>  
            <td></td>  
            <td><em>Ours</em></td>  
            <td>-</td>  
            <td><strong>35.12</strong></td>  
            <td><strong>65.09</strong></td>  
            <td><strong>33.30</strong></td>  
            <td><strong>22.27</strong></td>  
            <td><strong>43.08</strong></td>  
            <td>36.51</td>  
        </tr>  
    </tbody>  
</table>

### Citation


### Contact
Please Contact yaoliang@hhu.edu.cn
