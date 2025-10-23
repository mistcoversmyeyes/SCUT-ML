#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lab2 分类算法实现模块
实现逻辑回归、线性SVM和核SVM分类器
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

class Lab2Classifier:
    """Lab2分类算法实现类"""

    def __init__(self):
        self.models = {}
        self.best_params = {}
        self.training_history = {}

    def train_logistic_regression(self, X_train, y_train, param_grid=None, cv=5):
        """
        训练逻辑回归分类器

        Args:
            X_train: 训练特征
            y_train: 训练标签
            param_grid: 参数网格
            cv: 交叉验证折数

        Returns:
            训练好的模型和最佳参数
        """
        print("🔧 训练逻辑回归分类器...")

        if param_grid is None:
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l2'],
                'solver': ['liblinear', 'lbfgs'],
                'max_iter': [1000]
            }

        # 基础模型
        base_model = LogisticRegression(random_state=42)

        # 网格搜索
        grid_search = GridSearchCV(
            base_model, param_grid, cv=cv,
            scoring='accuracy', n_jobs=-1, verbose=1
        )

        # 训练
        grid_search.fit(X_train, y_train)

        # 保存结果
        self.models['logistic_regression'] = grid_search.best_estimator_
        self.best_params['logistic_regression'] = grid_search.best_params_
        self.training_history['logistic_regression'] = {
            'cv_scores': grid_search.cv_results_,
            'best_score': grid_search.best_score_
        }

        print(f"  ✅ 最佳参数: {grid_search.best_params_}")
        print(f"  ✅ 交叉验证最佳准确率: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_, grid_search.best_params_

    def train_linear_svm(self, X_train, y_train, param_grid=None, cv=5):
        """
        训练线性SVM分类器

        Args:
            X_train: 训练特征
            y_train: 训练标签
            param_grid: 参数网格
            cv: 交叉验证折数

        Returns:
            训练好的模型和最佳参数
        """
        print("🔧 训练线性SVM分类器...")

        if param_grid is None:
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'kernel': ['linear'],
                'probability': [True]
            }

        # 基础模型
        base_model = SVC(random_state=42)

        # 网格搜索
        grid_search = GridSearchCV(
            base_model, param_grid, cv=cv,
            scoring='accuracy', n_jobs=-1, verbose=1
        )

        # 训练
        grid_search.fit(X_train, y_train)

        # 保存结果
        self.models['linear_svm'] = grid_search.best_estimator_
        self.best_params['linear_svm'] = grid_search.best_params_
        self.training_history['linear_svm'] = {
            'cv_scores': grid_search.cv_results_,
            'best_score': grid_search.best_score_
        }

        print(f"  ✅ 最佳参数: {grid_search.best_params_}")
        print(f"  ✅ 交叉验证最佳准确率: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_, grid_search.best_params_

    def train_rbf_svm(self, X_train, y_train, param_grid=None, cv=5):
        """
        训练RBF核SVM分类器

        Args:
            X_train: 训练特征
            y_train: 训练标签
            param_grid: 参数网格
            cv: 交叉验证折数

        Returns:
            训练好的模型和最佳参数
        """
        print("🔧 训练RBF核SVM分类器...")

        if param_grid is None:
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf'],
                'probability': [True]
            }

        # 基础模型
        base_model = SVC(random_state=42)

        # 网格搜索
        grid_search = GridSearchCV(
            base_model, param_grid, cv=cv,
            scoring='accuracy', n_jobs=-1, verbose=1
        )

        # 训练
        grid_search.fit(X_train, y_train)

        # 保存结果
        self.models['rbf_svm'] = grid_search.best_estimator_
        self.best_params['rbf_svm'] = grid_search.best_params_
        self.training_history['rbf_svm'] = {
            'cv_scores': grid_search.cv_results_,
            'best_score': grid_search.best_score_
        }

        print(f"  ✅ 最佳参数: {grid_search.best_params_}")
        print(f"  ✅ 交叉验证最佳准确率: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_, grid_search.best_params_

    def evaluate_model(self, model, X_test, y_test, model_name, dataset_name):
        """
        评估模型性能

        Args:
            model: 训练好的模型
            X_test: 测试特征
            y_test: 测试标签
            model_name: 模型名称
            dataset_name: 数据集名称

        Returns:
            评估结果字典
        """
        print(f"📊 评估 {model_name} 在 {dataset_name} 上的性能...")

        # 预测
        y_pred = model.predict(X_test)
        y_prob = None

        # 对于二分类，获取概率预测
        if len(np.unique(y_test)) == 2:
            y_prob = model.predict_proba(X_test)[:, 1]

        # 基本指标
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # 生成分类报告
        report = classification_report(y_test, y_pred, output_dict=True)

        print(f"  ✅ 准确率: {accuracy:.4f}")

        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': report,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'y_true': y_test
        }

    def plot_confusion_matrix(self, cm, class_names, model_name, dataset_name):
        """
        绘制混淆矩阵

        Args:
            cm: 混淆矩阵
            class_names: 类别名称列表
            model_name: 模型名称
            dataset_name: 数据集名称
        """
        plt.figure(figsize=(8, 6))

        # 使用热力图显示混淆矩阵
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)

        plt.title(f'{model_name} - {dataset_name} 混淆矩阵')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')

        # 保存图片
        filename = f"{model_name}_{dataset_name}_confusion_matrix.png"
        filepath = f"lab2/outputs/{filename}"
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.show()

        print(f"  📈 混淆矩阵已保存到: {filepath}")

    def plot_roc_curve(self, y_true, y_prob, model_name, dataset_name):
        """
        绘制ROC曲线（仅用于二分类）

        Args:
            y_true: 真实标签
            y_prob: 预测概率
            model_name: 模型名称
            dataset_name: 数据集名称
        """
        if y_prob is None:
            print("  ⚠️  跳过ROC曲线绘制（多分类任务）")
            return

        # 计算ROC曲线
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        # 绘制ROC曲线
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'{model_name} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率')
        plt.ylabel('真阳性率')
        plt.title(f'{model_name} - {dataset_name} ROC曲线')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        # 保存图片
        filename = f"{model_name}_{dataset_name}_roc_curve.png"
        filepath = f"lab2/outputs/{filename}"
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.show()

        print(f"  📈 ROC曲线已保存到: {filepath}")
        print(f"  ✅ AUC值: {roc_auc:.4f}")

    def compare_models(self, results_dict, dataset_name):
        """
        比较不同模型的性能

        Args:
            results_dict: 包含所有模型结果的字典
            dataset_name: 数据集名称
        """
        print(f"📊 比较各模型在 {dataset_name} 上的性能...")

        # 提取准确率
        models = list(results_dict.keys())
        accuracies = [results_dict[model]['accuracy'] for model in models]

        # 绘制柱状图
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, accuracies, color=['skyblue', 'lightgreen', 'lightcoral'])

        # 添加数值标签
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=12)

        plt.title(f'各模型在 {dataset_name} 上的准确率对比')
        plt.xlabel('模型')
        plt.ylabel('准确率')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)

        # 保存图片
        filename = f"{dataset_name}_model_comparison.png"
        filepath = f"lab2/outputs/{filename}"
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.show()

        print(f"  📈 模型对比图已保存到: {filepath}")

        # 打印详细比较结果
        print("\n📋 详细性能对比:")
        print("-" * 60)
        print(f"{'模型':<15} {'准确率':<10} {'最佳参数':<30}")
        print("-" * 60)

        for model in models:
            accuracy = results_dict[model]['accuracy']
            if model in self.best_params:
                params = str(self.best_params[model])
                if len(params) > 25:
                    params = params[:25] + "..."
            else:
                params = "N/A"
            print(f"{model:<15} {accuracy:<10.4f} {params:<30}")

    def run_full_experiment(self, X_train, X_test, y_train, y_test, dataset_name):
        """
        运行完整的实验流程

        Args:
            X_train, X_test: 训练和测试特征
            y_train, y_test: 训练和测试标签
            dataset_name: 数据集名称

        Returns:
            所有模型的评估结果
        """
        print(f"\n{'='*20} {dataset_name} 分类实验 {'='*20}")

        results = {}

        # 1. 训练逻辑回归
        print("\n1️⃣ 逻辑回归")
        lr_model, lr_params = self.train_logistic_regression(X_train, y_train)
        results['logistic_regression'] = self.evaluate_model(
            lr_model, X_test, y_test, '逻辑回归', dataset_name
        )

        # 2. 训练线性SVM
        print("\n2️⃣ 线性SVM")
        svm_linear_model, svm_linear_params = self.train_linear_svm(X_train, y_train)
        results['linear_svm'] = self.evaluate_model(
            svm_linear_model, X_test, y_test, '线性SVM', dataset_name
        )

        # 3. 训练RBF核SVM
        print("\n3️⃣ RBF核SVM")
        svm_rbf_model, svm_rbf_params = self.train_rbf_svm(X_train, y_train)
        results['rbf_svm'] = self.evaluate_model(
            svm_rbf_model, X_test, y_test, 'RBF核SVM', dataset_name
        )

        # 4. 生成可视化结果
        print("\n📈 生成可视化结果...")

        # 确定类别名称
        unique_classes = np.unique(y_test)
        if len(unique_classes) == 2:
            class_names = ['负类', '正类']
        else:
            class_names = [f'类别 {i}' for i in unique_classes]

        # 为每个模型生成混淆矩阵和ROC曲线
        for model_name, result in results.items():
            self.plot_confusion_matrix(
                result['confusion_matrix'], class_names, model_name, dataset_name
            )
            self.plot_roc_curve(
                result['y_true'], result['y_prob'], model_name, dataset_name
            )

        # 5. 模型对比
        self.compare_models(results, dataset_name)

        return results

    def get_model_summary(self):
        """获取所有模型的训练摘要"""
        print("\n" + "="*60)
        print("📊 模型训练摘要")
        print("="*60)

        for model_name, history in self.training_history.items():
            print(f"\n{model_name}:")
            print(f"  最佳交叉验证准确率: {history['best_score']:.4f}")
            print(f"  最佳参数: {self.best_params[model_name]}")

if __name__ == "__main__":
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    print("Lab2 分类算法实现模块")
    print("请使用主实验脚本运行完整实验")