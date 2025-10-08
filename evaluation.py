#!/usr/bin/env python
# -*- coding: utf-8 -*-

import statistics
import numpy as np
from sklearn import svm
from sklearn import linear_model
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc,precision_recall_curve,roc_auc_score
from sklearn.metrics import precision_score,recall_score,f1_score
from collections import Counter
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from ELM.elm import ELMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def get_normalized_inner_product_score(vector1, vector2):
	return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

def get_sigmoid_score(vector1, vector2):
	return sigmoid(np.dot(vector1, vector2))

def get_average_score(vector1, vector2):
	return (vector1 + vector2)/2

def get_hadamard_score(vector1, vector2):
	return np.multiply(vector1, vector2)

def get_l1_score(vector1, vector2):
	return np.abs(vector1 - vector2)

def get_l2_score(vector1, vector2):
	return np.square(vector1 - vector2)

def get_link_score(embeds, node1, node2, score_type):
	if score_type not in ["cosine", "sigmoid", "hadamard", "average", "l1", "l2"]:
		raise NotImplementedError
	vector_dimension = embeds.shape[1]
	try:
		vector1 = embeds[node1]
		vector2 = embeds[node2]
	except Exception as e:
		if score_type in ["cosine", "sigmoid"]:
			return 0
		elif score_type in ["hadamard", "average", "l1", "l2"]:
			return np.zeros(vector_dimension)

	if score_type == "cosine":
		score = get_normalized_inner_product_score(vector1, vector2)
	elif score_type == "sigmoid":
		score = get_sigmoid_score(vector1, vector2)
	elif score_type == "hadamard":
		score = get_hadamard_score(vector1, vector2)
	elif score_type == "average":
		score = get_average_score(vector1, vector2)
	elif score_type == "l1":
		score = get_l1_score(vector1, vector2)
	elif score_type == "l2":
		score = get_l2_score(vector1, vector2)

	return score

def get_links_scores(embeds, links, score_type):
	features = []
	num_links = 0
	for l in links:
		num_links = num_links + 1
		node1, node2 = l[0], l[1]
		f = get_link_score(embeds, node1, node2, score_type)
		features.append(f)
	return features

def evaluate_classifier(embeds, train_pos_edges, train_neg_edges, test_pos_edges, test_neg_edges, score_type):
	train_pos_feats = np.array(get_links_scores(embeds, train_pos_edges, score_type))
	train_neg_feats = np.array(get_links_scores(embeds, train_neg_edges, score_type))
	train_pos_labels = np.ones(train_pos_feats.shape[0])
	train_neg_labels = np.zeros(train_neg_feats.shape[0])
	train_data = np.concatenate((train_pos_feats, train_neg_feats), axis=0)
	train_labels = np.append(train_pos_labels, train_neg_labels)

	test_pos_feats = np.array(get_links_scores(embeds, test_pos_edges, score_type))
	test_neg_feats = np.array(get_links_scores(embeds, test_neg_edges, score_type))
	test_pos_labels = np.ones(test_pos_feats.shape[0])
	test_neg_labels = np.zeros(test_neg_feats.shape[0])
	test_data = np.concatenate((test_pos_feats, test_neg_feats), axis=0)
	test_labels = np.append(test_pos_labels, test_neg_labels)

	logistic_regression = linear_model.LogisticRegression()
	logistic_regression.fit(train_data, train_labels)

	test_predict_prob = logistic_regression.predict_proba(test_data)
	test_predict = logistic_regression.predict(test_data)
	# print(test_predict.shape, test_predict_prob.shape)

	auroc = roc_auc_score(test_labels, test_predict_prob[:, 1])
	precisions, recalls, _ = precision_recall_curve(test_labels, test_predict_prob[:, 1])
	auprc = auc(recalls, precisions)
	return auroc, auprc


def node_classification(embed, labels_dict,id_to_index):
	# 创建一个与embeds相同长度的标签数组
	num_samples = embed.shape[0]
	labels = np.full(num_samples, -1)  # 初始化为-1或其他值，表示未标记

	# 填充标签数组
	for key, value in labels_dict.items():
		if key in id_to_index:
			index = id_to_index[key]
			if index < num_samples:  # 确保索引在范围内
				labels[index] = value-1

	# 确保没有未标记的样本
	if np.any(labels == -1):
		print("Warning: Some samples are not labeled.")
		unlabelled_indices = np.where(labels == -1)[0]
		print(f"Unlabeled sample indices: {unlabelled_indices}")
	else:
		print("All samples are labeled.")

	# 检查标签是否在有效范围内
	if np.any(labels < 0) or np.any(labels >= 3):  # 假设有 3 个类
		print("Error: Labels should be in the range [0, 3).")


	X_train, X_test, y_train, y_test = train_test_split(embed, labels, test_size=0.3, random_state=42)
	# 构建 LightGBM 数据集
	# 创建 LightGBM 数据集
	train_data = lgb.Dataset(X_train, label=y_train)
	test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

	# 设置 LightGBM 参数
	params = {
		'objective': 'multiclass',  # 多类别分类
		'num_class': 3,  # 类别数
		'metric': 'multi_logloss',  # 评价指标
		'boosting_type': 'gbdt',  # 使用 GBDT
		'learning_rate': 0.1,  # 学习率
		'num_leaves': 31,  # 叶子数
		'max_depth': -1,  # 最大深度
	}

	# 训练模型
	num_round = 100
	bst = lgb.train(params, train_data, num_round)

	# 预测
	y_pred_proba = bst.predict(X_test,pred_proba=True)
	y_pred = bst.predict(X_test)
	y_pred_max = np.argmax(y_pred, axis=1)  # 选择概率最大的类

    #SVM
	# clf = SGDClassifier(loss='squared_hinge', alpha=0.0001, max_iter=500, shuffle=True, n_jobs=1,
	# 	class_weight="balanced", verbose=False, tol=None, random_state=12345)
	# clf.fit(X_train, y_train)
	# test_pred_y = clf.predict(X_test)
    #ELM
	# clf = ELMClassifier(n_hidden=500, activation_func='hardlim', alpha=1.0, random_state=0)
	# clf.fit(X_train, y_train)
	# test_pred_y = clf.predict(X_test)
    #RF
	# clf = make_pipeline(StandardScaler(), RandomForestClassifier())
	# clf.fit(X_train, y_train)
	# test_pred_y = clf.predict(X_test)
    #MLP
	# clf = MLPClassifier()
	# clf.fit(X_train, y_train)
	# test_pred_y = clf.predict(X_test)

	test_micro_f1 = f1_score(y_test, y_pred_max, average="micro")
	print("### micro_F1 = %f" % test_micro_f1)

	test_macro_f1 = f1_score(y_test, y_pred_max, average="macro")
	print("### macro_F1 = %f" % test_macro_f1)


	acc = accuracy_score(y_test, y_pred_max)


	auc = roc_auc_score(y_test, y_pred_proba,multi_class='ovr')


	f1 = f1_score(y_test, y_pred_max,average='macro')
	precision = precision_score(y_test, y_pred_max,average='macro')
	recall = recall_score(y_test, y_pred_max,average='macro')

	return acc,auc,f1,precision,recall

