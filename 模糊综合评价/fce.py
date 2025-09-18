import numpy as np

def normalize(w):
	"""
	归一化权重向量（保证和为1）
	输入: w 可为 list 或 1d np.array
	输出: 归一化后的 np.array
	"""
	w = np.array(w, dtype=float)
	s = w.sum()
	if s == 0:
		raise ValueError("权重和为0，无法归一化")
	return w / s

def compose_one_level(weights, R):
	"""
	对单个一级指标下的二级指标进行模糊综合评价
	weights: 二级指标权重（长度 n）
	R: 二级评价矩阵，shape = (n, m) ，m 为评价等级数
	返回: 1xm 的行向量（np.array）
	运算: Bi = weights (1xn) @ R (n xm) -> (1 xm)
	"""
	w = normalize(weights)
	Rm = np.array(R, dtype=float)
	if Rm.shape[0] != w.shape[0]:
		raise ValueError("二级权重长度与评价矩阵行数不匹配")
	return w.dot(Rm)

def multi_level_evaluation(A, A_list, R_list, score_vector=None):
	"""
	多级模糊综合评价
	A: 一级权重，长度 p
	A_list: 长度 p 的列表，每项为对应一级指标的二级权重向量
	R_list: 长度 p 的列表，每项为对应一级指标的二级评价矩阵
	score_vector: 可选，长度为等级数的评分向量，用于解模糊
	返回: (B_total, score) 其中 B_total 为等级分布向量（长度 m），score 为标量（若未提供score_vector则为 None）
	"""
	A = normalize(A)
	p = len(A)
	if not (len(A_list) == len(R_list) == p):
		raise ValueError("A、A_list、R_list 长度必须一致")
	# 计算每个一级指标下的二级模糊评价向量 Bi（长度 m）
	B_rows = []
	for ai, Ri in zip(A_list, R_list):
		Bi = compose_one_level(ai, Ri)  # 1 x m
		B_rows.append(Bi)
	B_matrix = np.vstack(B_rows)  # shape p x m
	# 一级加权合成
	B_total = A.dot(B_matrix)  # 1 x m
	B_total = np.array(B_total, dtype=float)
	# 解模糊（若提供评分向量）
	score = None
	if score_vector is not None:
		S = np.array(score_vector, dtype=float)
		if S.shape[0] != B_total.shape[0]:
			raise ValueError("评分向量长度与等级数不匹配")
		score = B_total.dot(S)
	return B_total, score

# 示例（请把下面的示例 Ri 替换为从图片表格提取的真实数据）
if __name__ == "__main__":
	# 一级权重（图片示例）
	A = [0.4, 0.3, 0.2, 0.1]

	# 二级权重示例（对应 A1..A4，来自图片）
	A1 = [0.2, 0.3, 0.3, 0.2]
	A2 = [0.3, 0.2, 0.1, 0.2, 0.2]
	A3 = [0.1, 0.2, 0.3, 0.2, 0.2]
	A4 = [0.3, 0.2, 0.2, 0.3]

	# 评价矩阵，严格按图片表格填写
	R1 = [
		[0.8, 0.15, 0.05, 0.0, 0.0],
		[0.2, 0.6, 0.1, 0.1, 0.0],
		[0.5, 0.4, 0.1, 0.0, 0.0],
		[0.1, 0.3, 0.5, 0.05, 0.05],
	]
	R2 = [
		[0.3, 0.5, 0.15, 0.05, 0.0],
		[0.2, 0.2, 0.4, 0.1, 0.1],
		[0.4, 0.4, 0.1, 0.1, 0.0],
		[0.1, 0.3, 0.3, 0.2, 0.1],
		[0.3, 0.2, 0.2, 0.2, 0.1],
	]
	R3 = [
		[0.1, 0.3, 0.5, 0.1, 0.0],
		[0.2, 0.3, 0.3, 0.1, 0.1],
		[0.2, 0.3, 0.35, 0.15, 0.0],
		[0.1, 0.3, 0.4, 0.1, 0.1],
		[0.1, 0.4, 0.3, 0.1, 0.1],
	]
	R4 = [
		[0.3, 0.4, 0.2, 0.1, 0.0],
		[0.1, 0.4, 0.3, 0.1, 0.1],
		[0.2, 0.3, 0.4, 0.1, 0.0],
		[0.4, 0.3, 0.2, 0.1, 0.0],
	]

	# 调用多级模糊综合评价
	B_total, score = multi_level_evaluation(A, [A1, A2, A3, A4], [R1, R2, R3, R4], score_vector=[5,4,3,2,1])

	print("最终等级分布（从优秀到差）:", np.round(B_total, 4))
	print("综合评分:", None if score is None else round(float(score), 4))
