# Lab4 AdaBoost人脸检测实验报告编译说明

## 📁 文件说明

### LaTeX源文件
- `Lab4_AdaBoost_Face_Detection_Report.tex` - 实验报告LaTeX源文件

### 必要文件
- `IEEEtran.cls` - IEEE期刊格式模板
- `SCUT.png` - 华工Logo
- `training_curves.png` - 训练曲线图

## 🚀 编译方法

### 方法1: 使用pdflatex (推荐)
```bash
# 在Lab4目录下执行
pdflatex Lab4_AdaBoost_Face_Detection_Report.tex
pdflatex Lab4_AdaBoost_Face_Detection_Report.tex  # 第二次编译以更新引用
```

### 方法2: 使用xelatex
```bash
xelatex Lab4_AdaBoost_Face_Detection_Report.tex
xelatex Lab4_AdaBoost_Face_Detection_Report.tex
```

## 📊 报告内容概述

### 实验成果
- ✅ 完整实现AdaBoost人脸检测系统
- ✅ 使用Extended Yale B数据集 + 自建非人脸数据
- ✅ Haar特征提取 (2000个特征)
- ✅ 与OpenCV CascadeClassifier性能对比

### 关键结果
- **F1分数**: 93.46% (超越OpenCV的90.29%)
- **召回率**: 100% (无漏检)
- **精确率**: 87.72% (7个假正例)
- **训练收敛**: 7轮达到完美训练精度

### 技术亮点
- 快速收敛训练算法
- 完整的评估指标分析
- 详细的性能对比表格
- 训练过程可视化

## 🎯 适用场景
- 机器学习课程实验报告
- 计算机视觉项目文档
- AdaBoost算法实现参考
- 人脸检测技术研究

---
*注: 编译前请确保已安装TeX Live或MiKTeX等LaTeX发行版*