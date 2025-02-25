import json
import random
import os


def convert_txt_to_json(input_file, train_output_file, valid_output_file, valid_ratio=0.2):
    """
    将原始txt文件转换为指定格式的JSON文件，并划分训练集和验证集
    
    参数:
    input_file (str): 输入的txt文件路径
    train_output_file (str): 训练集输出文件路径
    valid_output_file (str): 验证集输出文件路径
    valid_ratio (float): 验证集所占比例，默认0.2
    """
    # 读取原始数据
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 转换数据格式
    converted_data = []
    for line in lines:
        try:
            # 解析原始JSON数据
            data = json.loads(line.strip())

            # 构建新的格式
            question_content = data.get("问题内容", "")
            answer = data.get("答案", "")

            # 从答案中提取出主要内容（这里假设需要用[]包裹起来）
            # 你可能需要根据实际情况调整答案的提取逻辑
            answer_content = answer.split("，")[0] + "。" if "，" in answer else answer

            new_data = {
                "context": f"Instruction: 你现在是一个很厉害的阅读理解器，严格按照人类指令进行回答。\nInput: 以下是客户的问题，请进行回答,不要胡编乱造\n\n{question_content}\nAnswer: ",
                "target": f"['{answer_content}']"
            }

            converted_data.append(new_data)
        except json.JSONDecodeError:
            print(f"警告: 无法解析行: {line}")
            continue

    # 打乱数据以便随机划分
    random.shuffle(converted_data)

    # 计算验证集大小
    valid_size = int(len(converted_data) * valid_ratio)
    train_size = len(converted_data) - valid_size

    # 划分数据集
    train_data = converted_data[:train_size]
    valid_data = converted_data[train_size:]

    # 写入训练集
    with open(train_output_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # 写入验证集
    with open(valid_output_file, 'w', encoding='utf-8') as f:
        for item in valid_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"转换完成！总数据: {len(converted_data)}, 训练集: {len(train_data)}, 验证集: {len(valid_data)}")
    print(f"训练集保存至: {train_output_file}")
    print(f"验证集保存至: {valid_output_file}")


if __name__ == "__main__":
    # 设置文件路径
    input_file = r"D:\ecommerce_logistics_RAG\dataset\dataset.txt"  # 输入的原始文件名，请根据实际情况修改
    train_output_file = r"/dataset/ChatGLM_6B/train_data.json"
    valid_output_file = r"/dataset/ChatGLM_6B/valid_data.json"

    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 输入文件 '{input_file}' 不存在!")
    else:
        # 执行转换
        convert_txt_to_json(input_file, train_output_file, valid_output_file, valid_ratio=0.2)
