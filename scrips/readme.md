# 使用说明

## 执行步骤
- 获取ROS Bag的数据，更改*scrips/script_config.json*文件的**DATA_FILE**参数，及*scrips/log_LF/export_bag_to_csv.py*的数据包路径
```
cd scrips/log_LF/
python3 export_bag_to_csv.py
```
- 查看所有腿的足端力曲线

根据数据的时间长短，查看前后数据的力阈值，更改相关py文件的时间段，相关曲线图在*scrips/log_data/"DATA_FILE"/Force/fig/*文件夹下
```
cd scrips/log_force/
python3 plot_foot_force_all_0_15.py
```

- 修改*build_contact_state_labels.py*的阈值，构建基于足端力阈值的标签
```
cd scrips/log_force/
python3 build_contact_state_labels.py
```

- 裁切最后的csv时间段
```
cd scrips/log_Gait_GRF_Label/
python3 crop_force_label_by_time.py
```

- 最后的数据汇总在，*scrips/log_data/"DATA_FILE"/* 文件夹下
```
data.csv -- 原始数据
Gait_GRF/data_label_with_**_**.csv -- 基于足端力阈值的标签
Gait_GRF/data_label_with_force.csv -- 基于足端力阈值的标签(同一个阈值)
Gait_GRF/data_label_with_gait.csv -- 基于步态的标签(足端的z方向离机身的距离)
```