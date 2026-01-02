"""
GDN 评估模块
用于计算异常检测的各项评估指标
"""
from util.data import *
from util.debug_logger import get_logger
import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score


def get_full_err_scores(test_result, val_result):
    """
    计算所有特征的误差分数
    
    Args:
        test_result: 测试结果 [predictions, ground_truth, labels]
        val_result: 验证结果 [predictions, ground_truth, labels]
    
    Returns:
        all_scores: 所有特征的误差分数 [feature_num, sample_num]
        all_normals: 正常分数分布
    """
    logger = get_logger()
    
    np_test_result = np.array(test_result)
    np_val_result = np.array(val_result)

    all_scores =  None
    all_normals = None
    feature_num = np_test_result.shape[-1]

    labels = np_test_result[2, :, 0].tolist()
    
    logger.log_section("误差分数计算", icon='📈')
    logger.log("特征数量", feature_num)
    logger.log("测试样本数", np_test_result.shape[1])

    for i in range(feature_num):
        test_re_list = np_test_result[:2,:,i]
        val_re_list = np_val_result[:2,:,i]

        scores = get_err_scores(test_re_list, val_re_list)
        normal_dist = get_err_scores(val_re_list, val_re_list)
        
        # 打印前几个特征的误差信息
        if i < 3:
            logger.log(f"特征{i}误差", f"范围 [{scores.min():.4f}, {scores.max():.4f}], 均值 {scores.mean():.4f}")

        if all_scores is None:
            all_scores = scores
            all_normals = normal_dist
        else:
            all_scores = np.vstack((
                all_scores,
                scores
            ))
            all_normals = np.vstack((
                all_normals,
                normal_dist
            ))
    
    logger.log("误差分数矩阵", f"shape={all_scores.shape}")

    return all_scores, all_normals


def get_final_err_scores(test_result, val_result):
    """获取最终误差分数（取各特征最大值）"""
    full_scores, all_normals = get_full_err_scores(test_result, val_result, return_normal_scores=True)

    all_scores = np.max(full_scores, axis=0)

    return all_scores



def get_err_scores(test_res, val_res):
    """
    计算单个特征的误差分数
    
    Args:
        test_res: [predictions, ground_truth]
        val_res: [predictions, ground_truth]
    
    Returns:
        smoothed_err_scores: 平滑后的误差分数
    """
    test_predict, test_gt = test_res
    val_predict, val_gt = val_res

    n_err_mid, n_err_iqr = get_err_median_and_iqr(test_predict, test_gt)

    test_delta = np.abs(np.subtract(
                        np.array(test_predict).astype(np.float64), 
                        np.array(test_gt).astype(np.float64)
                    ))
    epsilon=1e-2

    err_scores = (test_delta - n_err_mid) / ( np.abs(n_err_iqr) +epsilon)

    smoothed_err_scores = np.zeros(err_scores.shape)
    before_num = 3
    for i in range(before_num, len(err_scores)):
        smoothed_err_scores[i] = np.mean(err_scores[i-before_num:i+1])

    
    return smoothed_err_scores



def get_loss(predict, gt):
    """计算MSE损失"""
    return eval_mseloss(predict, gt)

def get_f1_scores(total_err_scores, gt_labels, topk=1):
    """计算F1分数"""
    print('total_err_scores', total_err_scores.shape)
    # remove the highest and lowest score at each timestep
    total_features = total_err_scores.shape[0]

    # topk_indices = np.argpartition(total_err_scores, range(total_features-1-topk, total_features-1), axis=0)[-topk-1:-1]
    topk_indices = np.argpartition(total_err_scores, range(total_features-topk-1, total_features), axis=0)[-topk:]
    
    topk_indices = np.transpose(topk_indices)

    total_topk_err_scores = []
    topk_err_score_map=[]
    # topk_anomaly_sensors = []

    for i, indexs in enumerate(topk_indices):
       
        sum_score = sum( score for k, score in enumerate(sorted([total_err_scores[index, i] for j, index in enumerate(indexs)])) )

        total_topk_err_scores.append(sum_score)

    final_topk_fmeas = eval_scores(total_topk_err_scores, gt_labels, 400)

    return final_topk_fmeas

def get_val_performance_data(total_err_scores, normal_scores, gt_labels, topk=1):
    """
    使用验证集阈值计算性能
    
    Returns:
        f1, precision, recall, auc_score, threshold
    """
    logger = get_logger()
    
    total_features = total_err_scores.shape[0]

    topk_indices = np.argpartition(total_err_scores, range(total_features-topk-1, total_features), axis=0)[-topk:]

    total_topk_err_scores = []
    topk_err_score_map=[]

    total_topk_err_scores = np.sum(np.take_along_axis(total_err_scores, topk_indices, axis=0), axis=0)

    thresold = np.max(normal_scores)
    
    logger.log_subsection("验证集阈值评估", icon='📊')
    logger.log("阈值来源", "验证集最大误差分数")
    logger.log("阈值", f"{thresold:.4f}")

    pred_labels = np.zeros(len(total_topk_err_scores))
    pred_labels[total_topk_err_scores > thresold] = 1

    for i in range(len(pred_labels)):
        pred_labels[i] = int(pred_labels[i])
        gt_labels[i] = int(gt_labels[i])

    pre = precision_score(gt_labels, pred_labels)
    rec = recall_score(gt_labels, pred_labels)

    f1 = f1_score(gt_labels, pred_labels)


    auc_score = roc_auc_score(gt_labels, total_topk_err_scores)
    
    # 打印详细评估结果
    n_pred_anomaly = int(pred_labels.sum())
    n_gt_anomaly = int(sum(gt_labels))
    logger.log("预测异常数", f"{n_pred_anomaly} / {len(pred_labels)}")
    logger.log("真实异常数", f"{n_gt_anomaly} / {len(gt_labels)}")

    return f1, pre, rec, auc_score, thresold


def get_best_performance_data(total_err_scores, gt_labels, topk=1):
    """
    搜索最优阈值计算最佳性能
    
    Returns:
        best_f1, precision, recall, auc_score, threshold
    """
    logger = get_logger()

    total_features = total_err_scores.shape[0]

    # topk_indices = np.argpartition(total_err_scores, range(total_features-1-topk, total_features-1), axis=0)[-topk-1:-1]
    topk_indices = np.argpartition(total_err_scores, range(total_features-topk-1, total_features), axis=0)[-topk:]

    total_topk_err_scores = []
    topk_err_score_map=[]

    total_topk_err_scores = np.sum(np.take_along_axis(total_err_scores, topk_indices, axis=0), axis=0)
    
    logger.log_subsection("最优阈值搜索", icon='🔍')
    logger.log("搜索步数", 400)

    final_topk_fmeas ,thresolds = eval_scores(total_topk_err_scores, gt_labels, 400, return_thresold=True)

    th_i = final_topk_fmeas.index(max(final_topk_fmeas))
    thresold = thresolds[th_i]
    
    logger.log("最优阈值位置", f"第 {th_i + 1} / 400 步")
    logger.log("最优阈值", f"{thresold:.4f}")

    pred_labels = np.zeros(len(total_topk_err_scores))
    pred_labels[total_topk_err_scores > thresold] = 1

    for i in range(len(pred_labels)):
        pred_labels[i] = int(pred_labels[i])
        gt_labels[i] = int(gt_labels[i])

    pre = precision_score(gt_labels, pred_labels)
    rec = recall_score(gt_labels, pred_labels)

    auc_score = roc_auc_score(gt_labels, total_topk_err_scores)
    
    # 打印异常分数分布
    normal_mask = np.array(gt_labels) == 0
    anomaly_mask = np.array(gt_labels) == 1
    
    if sum(normal_mask) > 0 and sum(anomaly_mask) > 0:
        normal_scores = total_topk_err_scores[normal_mask]
        anomaly_scores = total_topk_err_scores[anomaly_mask]
        
        logger.log_subsection("异常分数分布", icon='📉')
        logger.log("正常样本分数", f"mean={normal_scores.mean():.4f}, std={normal_scores.std():.4f}, max={normal_scores.max():.4f}")
        logger.log("异常样本分数", f"mean={anomaly_scores.mean():.4f}, std={anomaly_scores.std():.4f}, max={anomaly_scores.max():.4f}")
        
        # 计算分离度
        separation = (anomaly_scores.mean() - normal_scores.mean()) / (normal_scores.std() + 1e-8)
        logger.log("分离度", f"{separation:.4f} (越大越好)")

    return max(final_topk_fmeas), pre, rec, auc_score, thresold
