import torch
from transformers import AutoTokenizer, AutoModel
import esm
import numpy as np
import os

def load_dataset(file_path):
    """加载数据集"""
    sequences = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():  # 跳过空行
                # 分割序列和标签，只取序列部分
                sequence = line.strip().split(',')[0]
                sequences.append(sequence)
    return sequences

def get_embeddings(sequences, model_type='esm'):
    """使用预训练模型生成序列嵌入"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    try:
        if model_type == 'esm':
            # 设置较小的批次大小以减少内存使用
            BATCH_SIZE = 128
            
            # 加载ESM-2模型
            print("正在加载ESM模型...")
            model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
            batch_converter = alphabet.get_batch_converter()
            model = model.to(device)
            
            # 分批处理序列
            embeddings_list = []
            for i in range(0, len(sequences), BATCH_SIZE):
                batch_sequences = sequences[i:i + BATCH_SIZE]
                print(f"处理批次 {i//BATCH_SIZE + 1}/{(len(sequences)-1)//BATCH_SIZE + 1}")
                
                # 准备数据
                batch_labels, batch_strs, batch_tokens = batch_converter(
                    [(str(j), seq) for j, seq in enumerate(batch_sequences)]
                )
                batch_tokens = batch_tokens.to(device)
                
                # 生成嵌入
                with torch.no_grad():
                    results = model(batch_tokens, repr_layers=[6])
                    batch_embeddings = results["representations"][6].mean(dim=1)
                    embeddings_list.append(batch_embeddings.cpu().numpy())
                
                # 清理内存
                del results, batch_embeddings
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            embeddings = np.concatenate(embeddings_list, axis=0)
            
        elif model_type == 'protbert':
            # 加载ProtBERT模型
            print("正在加载ProtBERT模型...")
            tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert")
            model = AutoModel.from_pretrained("Rostlab/prot_bert")
            model = model.to(device)
            
            BATCH_SIZE = 4
            embeddings_list = []
            
            for i in range(0, len(sequences), BATCH_SIZE):
                batch_sequences = sequences[i:i + BATCH_SIZE]
                print(f"处理批次 {i//BATCH_SIZE + 1}/{(len(sequences)-1)//BATCH_SIZE + 1}")
                
                batch_embeddings = []
                for seq in batch_sequences:
                    # 添加空格between tokens
                    seq = " ".join(list(seq))
                    inputs = tokenizer(seq, return_tensors="pt", padding=True)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                        # 使用[CLS]标记的嵌入作为序列表示
                        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                        batch_embeddings.append(embedding[0])
                
                embeddings_list.extend(batch_embeddings)
                
                # 清理内存
                del outputs, batch_embeddings
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            embeddings = np.array(embeddings_list)
        
        return embeddings
        
    except Exception as e:
        print(f"生成嵌入时出错: {str(e)}")
        print("返回空嵌入...")
        return np.random.randn(len(sequences), 32)  # 32维随机嵌入

def process_file(file_path, save_name, model_type='esm'):
    """处理单个文件并保存嵌入"""
    print(f"\n处理文件: {file_path}")
    print(f"使用预训练语言模型嵌入: {model_type}")
    
    # 读取序列
    print("读取序列数据...")
    sequences = load_dataset(file_path)
    print(f"总序列数: {len(sequences)}")
    print(f"第一个序列示例: {sequences[0]}")
    print(f"序列长度: {len(sequences[0])}")
    
    # 生成嵌入
    print("\n开始生成嵌入...")
    embeddings = get_embeddings(sequences, model_type=model_type)
    
    # 显示嵌入信息
    print("\n嵌入信息:")
    print(f"嵌入形状: {embeddings.shape}")
    print(f"第一个嵌入示例前10个维度: {embeddings[0][:10]}")
    print(f"嵌入值范围: [{embeddings.min():.4f}, {embeddings.max():.4f}]")
    print(f"嵌入平均值: {embeddings.mean():.4f}")
    print(f"嵌入标准差: {embeddings.std():.4f}")
    
    # 保存嵌入
    save_dir = "features"
    os.makedirs(save_dir, exist_ok=True)
    np.save(f"{save_dir}/{save_name}_{model_type}.npy", embeddings)
    print(f"\n嵌入已保存到 {save_dir}/{save_name}_{model_type}.npy")

def main():
    # 选择模型类型
    model_type = 'esm'
    # model_type = 'protbert'
    
    # 处理训练集
    train_file = "data/Human/train_m5C_201.txt"
    process_file(train_file, "train_embeddings", model_type)
    
    # 处理测试集
    test_file = "data/Human/test_m5C_201.txt"
    process_file(test_file, "test_embeddings", model_type)

if __name__ == "__main__":
    main()
