import numpy as np
import pickle as pkl
from Bio import SeqIO
import os
import scipy.io as sio
import json
import pickle
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from operator import itemgetter
import torch
from transformers import AutoTokenizer, AutoModel
N_ITER = 5
N_BEST_MODLES = 3
USE_BAYES_OPT = False
# USE_BAYES_OPT = True  
USE_EMBEDDINGS = True  # 是否使用预训练嵌入
# USE_EMBEDDINGS = False  # 是否使用预训练嵌入
EMBEDDING_MODEL = 'esm'  # 可选 'esm' 或 'protbert'

def read_fasta(file):
    """读取FASTA文件"""
    sequences = []
    for record in SeqIO.parse(file, "fasta"):
        sequences.append(str(record.seq))
    return sequences

def CalculateMatrix(data, order, k):
    """计算位置特异性矩阵"""
    if k == 1:
        matrix = np.zeros((len(data[0]), 4))
        for i in range(len(data[0]) - 1):
            for j in range(len(data)):
                matrix[i][order[data[j][i:i+1]]] += 1
    elif k == 2:
        matrix = np.zeros((len(data[0]) - 1, 16))
        for i in range(len(data[0]) - 2):
            for j in range(len(data)):
                matrix[i][order[data[j][i:i+2]]] += 1
    else:
        matrix = np.zeros((len(data[0]) - 2, 64))
        for i in range(len(data[0]) - 2):
            for j in range(len(data)):
                matrix[i][order[data[j][i:i+3]]] += 1           
    return matrix

def test_PSP(train_datapath, test_data, k, is_training=False):
    """提取PSP特征"""
    # 验证输入序列
    valid_nucleotides = set('ACGU')
    if not is_training:
        if not all(n in valid_nucleotides for n in test_data.upper()):
            raise ValueError(f"Invalid sequence: {test_data}. Sequence should only contain A, C, G, U")
        test_data = test_data.upper()  # 转换为大写
    
    sequences = read_fasta(train_datapath)
    train_positive = []
    train_negative = []
    
    for i in range(int(len(sequences)/2)):
        train_positive.append(str(sequences[i]))
    for j in range(int(len(sequences)/2), len(sequences)):
        train_negative.append(str(sequences[j]))
    
    train_p_num = len(train_positive)
    train_n_num = len(train_negative)
    
    nucleotides = ['A', 'C', 'G', 'U']
    
    if k == 1:
        nuc = [n1 for n1 in nucleotides]
        order = {nuc[i]: i for i in range(len(nuc))}
        
        matrix_po = CalculateMatrix(train_positive, order, 1)
        matrix_ne = CalculateMatrix(train_negative, order, 1)
        
        F1 = matrix_po/train_p_num
        F2 = matrix_ne/train_n_num       
        F = F1 - F2
        
        if is_training:
            test_seq = read_fasta(test_data)
            test = []
            for pos in test_seq:
                test.append(str(pos))
            test_num = len(test)
            test_l = len(test[0])
            
            code = []
            for sequence in test:
                for j in range(len(sequence)):
                    number = F[j][order[sequence[j:j+1]]]
                    code.append(number)
            code = np.array(code)
            code = code.reshape((test_num, test_l))
        else:
            code = []
            for j in range(len(test_data)):
                number = F[j][order[test_data[j:j+1]]]
                code.append(number)
            code = np.array(code)
            
    elif k == 2:
        dnuc = [n1 + n2 for n1 in nucleotides for n2 in nucleotides]
        order = {dnuc[i]: i for i in range(len(dnuc))}
        
        matrix_po = CalculateMatrix(train_positive, order, 2)
        matrix_ne = CalculateMatrix(train_negative, order, 2)
        
        F1 = matrix_po/train_p_num
        F2 = matrix_ne/train_n_num       
        F = F1 - F2
        
        if is_training:
            test_seq = read_fasta(test_data)
            test = []
            for pos in test_seq:
                test.append(str(pos))
            test_num = len(test)
            test_l = len(test[0])
            
            code = []
            for sequence in test:
                for j in range(len(sequence)-1):
                    number = F[j][order[sequence[j:j+2]]]
                    code.append(number)
            code = np.array(code)
            code = code.reshape((test_num, test_l-1))
        else:
            code = []
            for j in range(len(test_data)-1):
                number = F[j][order[test_data[j:j+2]]]
                code.append(number)
            code = np.array(code)
            
    else:
        tnuc = [n1 + n2 + n3 for n1 in nucleotides for n2 in nucleotides for n3 in nucleotides]
        order = {tnuc[i]: i for i in range(len(tnuc))}
        
        matrix_po = CalculateMatrix(train_positive, order, 3)
        matrix_ne = CalculateMatrix(train_negative, order, 3)
        
        F1 = matrix_po/train_p_num
        F2 = matrix_ne/train_n_num       
        F = F1 - F2
        
        if is_training:
            test_seq = read_fasta(test_data)
            test = []
            for pos in test_seq:
                test.append(str(pos))
            test_num = len(test)
            test_l = len(test[0])
            
            code = []
            for sequence in test:
                for j in range(len(sequence)-2):
                    number = F[j][order[sequence[j:j+3]]]
                    code.append(number)
            code = np.array(code)
            code = code.reshape((test_num, test_l-2))
        else:
            code = []
            for j in range(len(test_data)-2):
                number = F[j][order[test_data[j:j+3]]]
                code.append(number)
            code = np.array(code)
        
    return code

def Kmer(sequences, is_training=False):
    """提取Kmer特征"""
    AA = 'ACGU'
    AADict = {AA[i]: i for i in range(len(AA))}
    
    if is_training:
        Kmer_feature = []
        for sequence in sequences:
            kmer1 = [0] * 4
            for j in range(len(sequence)):
                kmer1[AADict[sequence[j]]] += 1
            if sum(kmer1) != 0:
                kmer1 = [i/sum(kmer1) for i in kmer1]
                
            kmer2 = [0] * 16
            for j in range(len(sequence)-2+1):
                kmer2[AADict[sequence[j]]*4 + AADict[sequence[j+1]]] += 1
            if sum(kmer2) != 0:
                kmer2 = [i/sum(kmer2) for i in kmer2]
                
            kmer3 = [0] * 64
            for j in range(len(sequence)-3+1):
                kmer3[AADict[sequence[j]]*16 + AADict[sequence[j+1]]*4 + AADict[sequence[j+2]]] += 1
            if sum(kmer3) != 0:
                kmer3 = [i/sum(kmer3) for i in kmer3]
                
            kmer = kmer1 + kmer2 + kmer3
            Kmer_feature.append(kmer)
        return np.array(Kmer_feature)
    else:
        sequence = sequences
        kmer1 = [0] * 4
        for j in range(len(sequence)):
            kmer1[AADict[sequence[j]]] += 1
        if sum(kmer1) != 0:
            kmer1 = [i/sum(kmer1) for i in kmer1]
            
        kmer2 = [0] * 16
        for j in range(len(sequence)-2+1):
            kmer2[AADict[sequence[j]]*4 + AADict[sequence[j+1]]] += 1
        if sum(kmer2) != 0:
            kmer2 = [i/sum(kmer2) for i in kmer2]
            
        kmer3 = [0] * 64
        for j in range(len(sequence)-3+1):
            kmer3[AADict[sequence[j]]*16 + AADict[sequence[j+1]]*4 + AADict[sequence[j+2]]] += 1
        if sum(kmer3) != 0:
            kmer3 = [i/sum(kmer3) for i in kmer3]
            
        return np.array(kmer1 + kmer2 + kmer3)

def PCPseDNC(sequences, is_training=False):
    """提取PCPseDNC特征"""
    with open('Phychepro.data', 'rb') as f:
        myPropertyValue = pkl.load(f)
        
    lamadaValue = 2
    weight = 0.1
    
    myDiIndex = {
        'AA': 0, 'AC': 1, 'AG': 2, 'AU': 3,
        'CA': 4, 'CC': 5, 'CG': 6, 'CU': 7,
        'GA': 8, 'GC': 9, 'GG': 10, 'GU': 11,
        'UA': 12, 'UC': 13, 'UG': 14, 'UU': 15
    }
    
    myPropertyName = ['Base stacking', 'Protein induced deformability', 'B-DNA twist', 'A-philicity', 
                     'Propeller twist', 'Duplex stability:(freeenergy)', 'DNA denaturation', 'Bending stiffness',
                     'Protein DNA twist', 'Aida_BA_transition', 'Breslauer_dG', 'Breslauer_dH',
                     'Electron_interaction', 'Hartman_trans_free_energy', 'Helix-Coil_transition',
                     'Lisser_BZ_transition', 'Polar_interaction', 'SantaLucia_dG', 'SantaLucia_dS',
                     'Sarai_flexibility', 'Stability', 'Sugimoto_dG', 'Sugimoto_dH', 'Sugimoto_dS',
                     'Duplex tability(disruptenergy)', 'Stabilising energy of Z-DNA', 'Breslauer_dS',
                     'Ivanov_BA_transition', 'SantaLucia_dH', 'Stacking_energy', 'Watson-Crick_interaction',
                     'Dinucleotide GC Content', 'Twist', 'Tilt', 'Roll', 'Shift', 'Slide', 'Rise']
    
    if is_training:
        PCPseDNC_feature = []
        for sequence in sequences:
            dipeptideFrequency = {}
            for i in range(len(sequence)-1):
                dipeptide = sequence[i:i+2]
                dipeptideFrequency[dipeptide] = dipeptideFrequency.get(dipeptide, 0) + 1
                
            thetaArray = []
            for tmpLamada in range(lamadaValue):
                theta = 0
                for i in range(len(sequence)-tmpLamada-2):
                    for p in myPropertyName:
                        theta += (float(myPropertyValue[p][myDiIndex[sequence[i:i+2]]]) - 
                                float(myPropertyValue[p][myDiIndex[sequence[i+tmpLamada+1:i+tmpLamada+3]]])) ** 2
                thetaArray.append(theta/(len(sequence)-tmpLamada-2))
            
            code = []
            for pair in sorted(myDiIndex.keys()):
                if pair in dipeptideFrequency:
                    code.append(dipeptideFrequency[pair]/(len(sequence)-1)/(1 + weight * sum(thetaArray)))
                else:
                    code.append(0)
                    
            for k in range(lamadaValue):
                code.append((weight * thetaArray[k])/(1 + weight * sum(thetaArray)))
                
            PCPseDNC_feature.append(code)
        return np.array(PCPseDNC_feature)
    else:
        sequence = sequences
        dipeptideFrequency = {}
        for i in range(len(sequence)-1):
            dipeptide = sequence[i:i+2]
            dipeptideFrequency[dipeptide] = dipeptideFrequency.get(dipeptide, 0) + 1
            
        thetaArray = []
        for tmpLamada in range(lamadaValue):
            theta = 0
            for i in range(len(sequence)-tmpLamada-2):
                for p in myPropertyName:
                    theta += (float(myPropertyValue[p][myDiIndex[sequence[i:i+2]]]) - 
                            float(myPropertyValue[p][myDiIndex[sequence[i+tmpLamada+1:i+tmpLamada+3]]])) ** 2
            thetaArray.append(theta/(len(sequence)-tmpLamada-2))
        
        code = []
        for pair in sorted(myDiIndex.keys()):
            if pair in dipeptideFrequency:
                code.append(dipeptideFrequency[pair]/(len(sequence)-1)/(1 + weight * sum(thetaArray)))
            else:
                code.append(0)
                
        for k in range(lamadaValue):
            code.append((weight * thetaArray[k])/(1 + weight * sum(thetaArray)))
            
        return np.array(code)

def TriNcleotideComposition(sequence, base):
    """计算三核苷酸组成"""
    trincleotides = [nn1 + nn2 + nn3 for nn1 in base for nn2 in base for nn3 in base]
    tnc_dict = {}
    for triN in trincleotides:
        tnc_dict[triN] = 0
    for i in range(len(sequence) - 2):
        tnc_dict[sequence[i:i + 3]] += 1
    for key in tnc_dict:
        tnc_dict[key] /= (len(sequence) - 2)
    return tnc_dict

def PseEIIP(sequences, is_training=False):
    """提取PseEIIP特征"""
    base = 'ACGU'
    EIIP_dict = {
        'A': 0.1260,
        'C': 0.1340,
        'G': 0.0806,
        'U': 0.1335
    }
    
    trincleotides = [n1 + n2 + n3 for n1 in base for n2 in base for n3 in base]
    EIIPxyz = {}
    for triN in trincleotides:
        EIIPxyz[triN] = EIIP_dict[triN[0]] + EIIP_dict[triN[1]] + EIIP_dict[triN[2]]
    
    if is_training:
        PseEIIP_feature = []
        for sequence in sequences:
            trinc_freq = TriNcleotideComposition(sequence, base)
            feature_vector = [EIIPxyz[triN] * trinc_freq[triN] for triN in trincleotides]
            PseEIIP_feature.append(feature_vector)
        return np.array(PseEIIP_feature)
    else:
        sequence = sequences
        trinc_freq = TriNcleotideComposition(sequence, base)
        feature_vector = [EIIPxyz[triN] * trinc_freq[triN] for triN in trincleotides]
        return np.array(feature_vector)



def extract_features(data, is_training=False, file_type='fasta'):
    """整合所有特征，包括预训练嵌入
    
    Args:
        data: 数据路径(训练模式)或单个序列(预测模式)
        is_training: 是否为训练模式
        file_type: 文件类型，可选 'fasta' 或 'txt'
    """

    if file_type == 'fasta':
        sequences = read_fasta(data)
    elif file_type == 'txt':
        sequences, _ = load_dataset(data)
    else:
        raise ValueError("file_type must be 'fasta' or 'txt'")
        
    # 提取基本特征
    Kmer1 = Kmer(sequences, is_training=True)
    PCPseDNC1 = PCPseDNC(sequences, is_training=True)
    PseEIIP1 = PseEIIP(sequences, is_training=True)
    
    # 如果启用了预训练嵌入，则添加嵌入特征
    if USE_EMBEDDINGS:
        print(f"\n使用预训练语言模型嵌入: {EMBEDDING_MODEL}")
        # 根据是否为训练模式加载对应的嵌入
        if is_training:
            emb_file = f"features/train_embeddings_{EMBEDDING_MODEL}.npy"
        else:
            emb_file = f"features/test_embeddings_{EMBEDDING_MODEL}.npy"
        
        print(f"加载嵌入文件: {emb_file}")
        embeddings = np.load(emb_file)
        
        # 打印维度信息以便确认
        print(f"基本特征数量: {len(sequences)}")
        print(f"加载的嵌入数量: {len(embeddings)}")
        assert len(sequences) == len(embeddings), "特征数量与嵌入数量不匹配！"
        
        feature_vector = np.concatenate((Kmer1, PCPseDNC1, PseEIIP1, embeddings), axis=1)
        
        # 记录特征维度信息（包括语言模型嵌入）
        feature_info = {
            'Kmer': {'start': 0, 'end': Kmer1.shape[1]},
            'PCPseDNC': {'start': Kmer1.shape[1], 'end': Kmer1.shape[1] + PCPseDNC1.shape[1]},
            'PseEIIP': {'start': Kmer1.shape[1] + PCPseDNC1.shape[1], 
                        'end': Kmer1.shape[1] + PCPseDNC1.shape[1] + PseEIIP1.shape[1]},
            'LM_Embeddings': {'start': Kmer1.shape[1] + PCPseDNC1.shape[1] + PseEIIP1.shape[1], 
                            'end': feature_vector.shape[1],
                            'model': EMBEDDING_MODEL}
        }
    else:
        feature_vector = np.concatenate((Kmer1, PCPseDNC1, PseEIIP1), axis=1)
        
        # 记录基本特征维度信息
        feature_info = {
            'Kmer': {'start': 0, 'end': Kmer1.shape[1]},
            'PCPseDNC': {'start': Kmer1.shape[1], 'end': Kmer1.shape[1] + PCPseDNC1.shape[1]},
            'PseEIIP': {'start': Kmer1.shape[1] + PCPseDNC1.shape[1], 'end': feature_vector.shape[1]}
        }
    
    return feature_vector, feature_info
    # else:
    #     sequence = data
    #     Kmer1 = Kmer(sequence, is_training=False)
    #     PCPseDNC1 = PCPseDNC(sequence, is_training=False)
    #     PseEIIP1 = PseEIIP(sequence, is_training=False)
        
    #     if USE_EMBEDDINGS:
    #         print(f"\n使用预训练语言模型嵌入: {EMBEDDING_MODEL}")
    #         # 加载测试集的预训练嵌入
    #         emb_file = f"features/test_embeddings_{EMBEDDING_MODEL}.npy"
    #         print(f"加载嵌入文件: {emb_file}")
    #         embeddings = np.load(emb_file)
    #         # 注意：这里需要根据序列在测试集中的位置获取对应的嵌入
    #         feature_vector = np.concatenate((Kmer1, PCPseDNC1, PseEIIP1, embeddings[0]))
    #     else:
    #         feature_vector = np.concatenate((Kmer1, PCPseDNC1, PseEIIP1))
            
    #     return feature_vector

def load_dataset(file_path):
    """读取数据集
    
    Args:
        file_path: 数据集文件路径
        
    Returns:
        sequences: 序列列表
        labels: 标签列表
    """
    sequences = []
    labels = []
    
    with open(file_path, 'r') as f:
        for line in f:
            # 去除行末换行符并按逗号分割
            seq, label = line.strip().split(',')
            sequences.append(seq)
            labels.append(int(label))
            
    return sequences, labels

def save_features(features, labels, feature_info, save_dir):
    """保存特征和特征信息
    
    Args:
        features: 特征矩阵
        labels: 标签列表
        feature_info: 特征维度信息字典
        save_dir: 保存目录
    """
    # 保存特征矩阵和标签
    with open(f"{save_dir}/features.pkl", 'wb') as f:
        pickle.dump({
            'features': features,
            'labels': labels
        }, f)
    
    # 保存特征维度信息
    with open(f"{save_dir}/feature_info.json", 'w') as f:
        json.dump(feature_info, f, indent=4)

def load_features(save_dir):
    """加载保存的特征
    
    Args:
        save_dir: 保存目录
        
    Returns:
        features: 特征矩阵
        labels: 标签列表
        feature_info: 特征维度信息字典
    """
    # 加载特征矩阵和标签
    with open(f"{save_dir}/features.pkl", 'rb') as f:
        data = pickle.load(f)
        features = data['features']
        labels = data['labels']
    
    # 加载特征维度信息
    with open(f"{save_dir}/feature_info.json", 'r') as f:
        feature_info = json.load(f)
        
    return features, labels, feature_info

def train_base_models(X_train, y_train, X_test, y_test, save_dir="features"):
    """训练基础版本的五种分类器"""
    # 创建模型保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 检查是否存在已保存的模型和评估结果
    if os.path.exists(f"{save_dir}/base_models.pkl") and os.path.exists(f"{save_dir}/base_results.json"):
        print("加载已保存的基础模型和评估结果...")
        # 加载模型
        with open(f"{save_dir}/base_models.pkl", 'rb') as f:
            models = pickle.load(f)
        # 加载评估结果
        with open(f"{save_dir}/base_results.json", 'r') as f:
            results = json.load(f)
            
        print("\n已保存的基础模型评估结果:")
        for name, metrics in results.items():
            print(f"\n{name} 评估结果:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
                
        return models, results
    
    # 如果没有保存的模型和结果，训练新模型
    print("训练新的基础模型...")
    models = {
        'SVM': SVC(probability=True, random_state=42),
        'GBDT': GradientBoostingClassifier(random_state=42),
        'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42),
        'LightGBM': LGBMClassifier(random_state=42, verbose=-1),
        'ExtraTree': ExtraTreesClassifier(random_state=42)
    }
    
    results = {}
    
    # 训练和评估每个模型
    for name, model in models.items():
        try:
            print(f"\n训练 {name}...")
            model.fit(X_train, y_train)
            
            # 预测
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            # 计算评估指标
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'auc': roc_auc_score(y_test, y_prob)
            }
            
            results[name] = metrics
            
            print(f"{name} 评估结果:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
                
        except Exception as e:
            print(f"{name} 训练失败: {str(e)}")
            results[name] = {'error': str(e)}
    
    # 保存模型和评估结果
    with open(f"{save_dir}/base_models.pkl", 'wb') as f:
        pickle.dump(models, f)
    
    with open(f"{save_dir}/base_results.json", 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\n基础模型和评估结果已保存到 {save_dir}")
    
    return models, results

def optimize_svm(X, y):
    """SVM贝叶斯优化"""
    def svm_cv(C, gamma):
        val = cross_val_score(
            SVC(C=10**C, gamma=10**gamma, probability=True, random_state=42),
            X, y, scoring='roc_auc', cv=3
        ).mean()
        return val
    
    optimizer = BayesianOptimization(
        f=svm_cv,
        pbounds={
            'C': (-3, 3),      # 10^-3 到 10^3
            'gamma': (-3, 3)   # 10^-3 到 10^3
        },
        random_state=42,
        verbose=2
    )
    optimizer.maximize(init_points=5, n_iter=N_ITER)
    
    return {
        'C': 10**optimizer.max['params']['C'],
        'gamma': 10**optimizer.max['params']['gamma']
    }

def optimize_gbdt(X, y):
    """GBDT贝叶斯优化"""
    def gbdt_cv(n_estimators, learning_rate, max_depth, min_samples_split, subsample):
        val = cross_val_score(
            GradientBoostingClassifier(
                n_estimators=int(n_estimators),
                learning_rate=learning_rate,
                max_depth=int(max_depth),
                min_samples_split=int(min_samples_split),
                subsample=subsample,
                random_state=42
            ),
            X, y, scoring='roc_auc', cv=3
        ).mean()
        return val
    
    optimizer = BayesianOptimization(
        f=gbdt_cv,
        pbounds={
            'n_estimators': (50, 500),
            'learning_rate': (0.01, 0.3),
            'max_depth': (3, 12),
            'min_samples_split': (2, 20),
            'subsample': (0.6, 1.0)
        },
        random_state=42,
        verbose=2
    )
    optimizer.maximize(init_points=5, n_iter=N_ITER)
    
    return {k: int(v) if k in ['n_estimators', 'max_depth', 'min_samples_split'] 
            else v for k, v in optimizer.max['params'].items()}

def optimize_xgb(X, y):
    """XGBoost贝叶斯优化"""
    def xgb_cv(n_estimators, learning_rate, max_depth, min_child_weight, 
               subsample, colsample_bytree, gamma, reg_alpha, reg_lambda):
        val = cross_val_score(
            XGBClassifier(
                n_estimators=int(n_estimators),
                learning_rate=learning_rate,
                max_depth=int(max_depth),
                min_child_weight=min_child_weight,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                gamma=gamma,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False
            ),
            X, y, scoring='roc_auc', cv=3
        ).mean()
        return val
    
    optimizer = BayesianOptimization(
        f=xgb_cv,
        pbounds={
            'n_estimators': (100, 800),      # 增加树的数量范围
            'learning_rate': (0.001, 0.3),   # 扩大学习率范围
            'max_depth': (3, 15),            # 扩大深度范围
            'min_child_weight': (1, 10),     # 调整最小子节点权重
            'subsample': (0.5, 1.0),         # 调整样本采样比例
            'colsample_bytree': (0.5, 1.0),  # 调整特征采样比例
            'gamma': (0, 10),                # 增加gamma范围
            'reg_alpha': (0, 10),            # 添加L1正则化
            'reg_lambda': (0, 10)            # 添加L2正则化
        },
        random_state=42,
        verbose=2
    )
    optimizer.maximize(init_points=10, n_iter=N_ITER)  # 增加初始点数量
    
    return {k: int(v) if k in ['n_estimators', 'max_depth'] 
            else v for k, v in optimizer.max['params'].items()}

def optimize_lgb(X, y):
    """LightGBM贝叶斯优化"""
    def lgb_cv(n_estimators, learning_rate, max_depth, num_leaves, subsample, colsample_bytree, min_child_samples):
        val = cross_val_score(
            LGBMClassifier(
                n_estimators=int(n_estimators),
                learning_rate=learning_rate,
                max_depth=int(max_depth),
                num_leaves=int(num_leaves),
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                min_child_samples=int(min_child_samples),
                random_state=42,
                verbose=-1
            ),
            X, y, scoring='roc_auc', cv=3
        ).mean()
        return val
    
    optimizer = BayesianOptimization(
        f=lgb_cv,
        pbounds={
            'n_estimators': (50, 500),
            'learning_rate': (0.01, 0.3),
            'max_depth': (3, 12),
            'num_leaves': (20, 200),
            'subsample': (0.6, 1.0),
            'colsample_bytree': (0.6, 1.0),
            'min_child_samples': (10, 100)
        },
        random_state=42,
        verbose=2
    )
    optimizer.maximize(init_points=5, n_iter=N_ITER)
    
    return {k: int(v) if k in ['n_estimators', 'max_depth', 'num_leaves', 'min_child_samples'] 
            else v for k, v in optimizer.max['params'].items()}

def optimize_et(X, y):
    """ExtraTree贝叶斯优化"""
    def et_cv(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, criterion, bootstrap):
        val = cross_val_score(
            ExtraTreesClassifier(
                n_estimators=int(n_estimators),
                max_depth=int(max_depth),
                min_samples_split=int(min_samples_split),
                min_samples_leaf=int(min_samples_leaf),
                max_features=max_features,
                criterion='gini' if criterion < 0.5 else 'entropy',
                bootstrap=bool(round(bootstrap)),  # 将浮点数转换为布尔值
                random_state=42
            ),
            X, y, scoring='roc_auc', cv=3
        ).mean()
        return val
    
    optimizer = BayesianOptimization(
        f=et_cv,
        pbounds={
            'n_estimators': (50, 500),
            'max_depth': (3, 12),
            'min_samples_split': (2, 20),
            'min_samples_leaf': (1, 10),
            'max_features': (0.5, 1.0),
            'criterion': (0, 1),  # 0: gini, 1: entropy
            'bootstrap': (0, 1)   # 0: False, 1: True
        },
        random_state=42,
        verbose=2
    )
    optimizer.maximize(init_points=5, n_iter=N_ITER)
    
    params = optimizer.max['params']
    return {
        'n_estimators': int(params['n_estimators']),
        'max_depth': int(params['max_depth']),
        'min_samples_split': int(params['min_samples_split']),
        'min_samples_leaf': int(params['min_samples_leaf']),
        'max_features': params['max_features'],
        'criterion': 'gini' if params['criterion'] < 0.5 else 'entropy',
        'bootstrap': bool(round(params['bootstrap']))
    }

def train_optimized_models(X_train, y_train, X_test, y_test, save_dir):
    """训练优化后的模型"""
    # 检查是否存在已保存的优化结果
    if os.path.exists(f"{save_dir}/bayes_opt_results.json"):
        print("加载已保存的贝叶斯优化结果...")
        with open(f"{save_dir}/bayes_opt_results.json", 'r') as f:
            params = json.load(f)
            svm_params = params['SVM']
            gbdt_params = params['GBDT']
            xgb_params = params['XGBoost']
            lgb_params = params['LightGBM']
            et_params = params['ExtraTree']
    else:
        if USE_BAYES_OPT:
            print("开始贝叶斯优化...")
            
            # 优化每个模型的参数
            print("\n优化SVM...")
            svm_params = optimize_svm(X_train, y_train)
            print("\n优化GBDT...")
            gbdt_params = optimize_gbdt(X_train, y_train)
            print("\n优化XGBoost...")
            xgb_params = optimize_xgb(X_train, y_train)
            print("\n优化LightGBM...")
            lgb_params = optimize_lgb(X_train, y_train)
            print("\n优化ExtraTree...")
            et_params = optimize_et(X_train, y_train)
        else:
            print("使用默认参数...")
            # 使用默认参数
            svm_params = {'C': 1.0, 'gamma': 'scale'}
            gbdt_params = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3}
            xgb_params = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3}
            lgb_params = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3}
            et_params = {'n_estimators': 100, 'max_depth': None}
        
        # 保存参数
        params = {
            'SVM': svm_params,
            'GBDT': gbdt_params,
            'XGBoost': xgb_params,
            'LightGBM': lgb_params,
            'ExtraTree': et_params
        }
        with open(f"{save_dir}/bayes_opt_results.json", 'w') as f:
            json.dump(params, f, indent=4)
    
    # 使用优化后的参数创建并训练模型
    print("\n使用优化参数训练模型...")
    optimized_models = {
        'SVM': SVC(probability=True, random_state=42, **svm_params),
        'GBDT': GradientBoostingClassifier(random_state=42, **gbdt_params),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss', **xgb_params),
        'LightGBM': LGBMClassifier(random_state=42, verbose=-1, **lgb_params),
        'ExtraTree': ExtraTreesClassifier(random_state=42, **et_params)
    }
    
    results = {}
    # 训练和评估每个模型
    for name, model in optimized_models.items():
        try:
            print(f"\n训练优化后的 {name}...")
            print(f"最优参数: {params[name]}")
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'auc': roc_auc_score(y_test, y_prob),
                'best_params': params[name]
            }
            
            results[name] = metrics
            
            print(f"{name} 优化后评估结果:")
            for metric, value in metrics.items():
                if metric != 'best_params':
                    print(f"{metric}: {value:.4f}")
                    
        except Exception as e:
            print(f"{name} 训练失败: {str(e)}")
            results[name] = {'error': str(e)}
    
    return results, optimized_models  # 返回结果和训练好的模型

def ensemble_models(X_train, y_train, X_test, y_test, optimized_models, n_best_models=3):
    """集成学习，使用多种集成方法"""
    # 获取每个模型的准确率并排序
    model_accuracies = {
        name: model.score(X_test, y_test) 
        for name, model in optimized_models.items()
    }
    sorted_models = sorted(model_accuracies.items(), key=itemgetter(1), reverse=True)
    
    # 选择前n个最佳模型
    selected_models = {
        name: optimized_models[name] 
        for name, _ in sorted_models[:n_best_models]
    }
    print(f"\n选择的模型: {list(selected_models.keys())}")
    
    # 生成基础预测概率
    train_probs = np.column_stack([
        model.predict_proba(X_train)[:, 1] 
        for model in selected_models.values()
    ])
    test_probs = np.column_stack([
        model.predict_proba(X_test)[:, 1] 
        for model in selected_models.values()
    ])
    
    # 修改集成方法的参数设置
    ensemble_methods = {
        'LogisticRegression': LogisticRegression(random_state=42),
        'ExtraTreesClassifier': ExtraTreesClassifier(n_estimators=200, random_state=42),
        'SVC': SVC(probability=True, random_state=42),
        'XGBClassifier': XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        ),
        'LGBMClassifier': LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
    }
    
    ensemble_results = {}
    
    # 对每种集成方法进行训练和评估
    for method_name, ensemble_model in ensemble_methods.items():
        print(f"\n训练 {method_name} 集成...")
        try:
            # 训练集成模型
            ensemble_model.fit(train_probs, y_train)
            
            # 预测
            y_pred = ensemble_model.predict(test_probs)
            y_prob = ensemble_model.predict_proba(test_probs)[:, 1]
            
            # 计算评估指标
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'auc': roc_auc_score(y_test, y_prob)
            }
            
            ensemble_results[method_name] = metrics
            
            print(f"{method_name} 集成评估结果:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
                
        except Exception as e:
            print(f"{method_name} 集成失败: {str(e)}")
            ensemble_results[method_name] = {'error': str(e)}
    
    return ensemble_results, list(selected_models.keys()), ensemble_methods, train_probs, y_train

def evaluate_best_model(best_model, X_test, y_test, model_name, save_dir):
    """评估最佳模型在测试集上的表现
    
    Args:
        best_model: 最佳模型
        X_test: 测试特征
        y_test: 测试标签
        model_name: 模型名称
        save_dir: 保存目录
    """
    print(f"\n评估最佳模型 ({model_name}) 在测试集上的表现...")
    
    # 预测
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    
    # 计算评估指标
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_prob)
    }
    
    # 打印结果
    print("\n测试集评估结果:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # # 保存结果
    # with open(f"{save_dir}/best_model_test_results.json", 'w') as f:
    #     json.dump({
    #         'model_name': model_name,
    #         'metrics': metrics
    #     }, f, indent=4)
    
    return metrics

def main():
    import os
    from sklearn.model_selection import train_test_split
    
    # 设置保存目录
    save_dir = "features"
    os.makedirs(save_dir, exist_ok=True)
    
    # 检查是否已有保存的特征
    if os.path.exists(f"{save_dir}/features.pkl"):
        print("加载已保存的特征...")
        features, labels, feature_info = load_features(save_dir)
        print(f"特征维度: {features.shape}")
        print("特征信息:")
        print(json.dumps(feature_info, indent=2))
    else:
        print("生成新特征...")
        # 加载训练集
        sequences, labels = load_dataset("data/Human/train_m5C_201.txt")
        
        # 提取特征
        features, feature_info = extract_features("data/Human/train_m5C_201.txt", 
                                                is_training=True, 
                                                file_type='txt')
        
        # 保存特征
        save_features(features, labels, feature_info, save_dir)
        print(f"特征已保存到 {save_dir} 目录")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print("\n开始训练基础模型...")
    base_models, base_results = train_base_models(X_train, y_train, X_test, y_test)
    
    # 获取模型对象（根据是否使用贝叶斯优化）
    if USE_BAYES_OPT:
        print("\n开始训练优化后的模型...")
        optimized_results, models_for_ensemble = train_optimized_models(X_train, y_train, X_test, y_test, save_dir)
    else:
        print("\n检查是否存在已保存的基础模型...")
        # 检查是否存在已保存的基础模型
        if os.path.exists(f"{save_dir}/base_models.pkl"):
            print("加载已保存的基础模型...")
            with open(f"{save_dir}/base_models.pkl", 'rb') as f:
                models_for_ensemble = pickle.load(f)
        else:
            print("创建并训练新的基础模型...")
            models_for_ensemble = base_models  # 使用已训练的基础模型
    
    # 进行集成学习
    print("\n开始集成学习...")
    ensemble_results, selected_models, ensemble_methods, train_probs, y_train = ensemble_models(
        X_train, y_train, X_test, y_test, models_for_ensemble, n_best_models=N_BEST_MODLES
    )
    
    # 选择最佳模型
    best_ensemble_method = max(ensemble_results.items(), 
                             key=lambda x: x[1]['auc'] if 'error' not in x[1] else -1)[0]
    print(f"\n最佳集成方法: {best_ensemble_method}")
    
    # 加载真实测试集
    print("\n加载真实测试集...")
    test_sequences, test_labels = load_dataset("data/Human/test_m5C_201.txt")
    
    # 提取测试集特征
    if os.path.exists(f"{save_dir}/test_features.pkl"):
        print("加载已保存的测试集特征...")
        test_features = pickle.load(open(f"{save_dir}/test_features.pkl", 'rb'))
    else:
        print("生成测试集特征...")
        test_features, _ = extract_features("data/Human/test_m5C_201.txt", 
                                          is_training=False, 
                                          file_type='txt')
        pickle.dump(test_features, open(f"{save_dir}/test_features.pkl", 'wb'))
    
    # 获取选定模型的预测概率
    test_probs = np.column_stack([
        models_for_ensemble[model_name].predict_proba(test_features)[:, 1]
        for model_name in selected_models
    ])
    
    # 使用最佳集成方法进行最终预测
    best_ensemble = ensemble_methods[best_ensemble_method]
    best_ensemble.fit(train_probs, y_train)  # 使用之前的训练数据
    
    # 评估最佳模型
    final_results = evaluate_best_model(best_ensemble, test_probs, test_labels, 
                                      best_ensemble_method, save_dir)
    
    # 保存所有结果
    results = {
        'base_models': {
            name: {
                metric: value 
                for metric, value in metrics.items() 
                if isinstance(value, (int, float, str, bool))  # 只保存基本类型的指标
            }
            for name, metrics in base_results.items()
        },
        'ensemble_results': {
            name: {
                metric: value 
                for metric, value in metrics.items() 
                if isinstance(value, (int, float, str, bool))
            }
            for name, metrics in ensemble_results.items()
        },
        'selected_models': selected_models
    }
    
    # 只有在使用贝叶斯优化时才添加优化结果
    if USE_BAYES_OPT:
        results['optimized_models'] = {
            name: {
                metric: value 
                for metric, value in metrics.items() 
                if isinstance(value, (int, float, str, bool)) or metric == 'best_params'
            }
            for name, metrics in optimized_results.items()
        }
    
    # 保存结果为JSON
    with open(f"{save_dir}/model_comparison_results.json", 'w') as f:
        json.dump(results, f, indent=4)
        
    # 打印最终对比结果
    print("\n最终性能对比:")
    if USE_BAYES_OPT:
        print("\n1. 基础模型 vs 优化后模型:")
        for model in base_results.keys():
            print(f"\n{model}:")
            if 'error' not in base_results[model] and 'error' not in optimized_results[model]:
                for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
                    base = base_results[model][metric]
                    opt = optimized_results[model][metric]
                    improvement = (opt - base) / base * 100
                    print(f"{metric}: {base:.4f} -> {opt:.4f} (提升: {improvement:.2f}%)")
    
    print("\n2. 集成学习结果:")
    for method, metrics in ensemble_results.items():
        if 'error' not in metrics:
            print(f"\n{method}:")
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):  # 只打印数值型指标
                    print(f"{metric}: {value:.4f}")
                
    print(f"\n3. 当前设置为集成前{N_BEST_MODLES}个模型!")


if __name__ == "__main__":
    main()
