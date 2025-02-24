"""
Description: 模型训练
    
-*- Encoding: UTF-8 -*-
@File     ：train.py.py
@Author   ：King Songtao
@Time     ：2025/2/24 下午7:51
@Contact  ：king.songtao@gmail.com
"""
import os
from datetime import datetime

import torch
import transformers
from transformers import AutoModelForCausalLM
from configs.log_config import get_logger
from configs.train_config import TrainConfig
from full_fine_tuning.dataloader import get_dataloader

param = TrainConfig()
logger = get_logger("train")


def calculate_acc(logit, labels, ignore_index=-100):
    logit = logit[:, :-1, :].contiguous().view(-1, logit.size(-1))
    labels = labels[:, 1:].contiguous().view(-1)

    _, logit = logit.max(dim=-1)
    non_pad_mask = labels.ne(ignore_index)

    n_correct = logit.eq(labels).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return n_correct, n_word


def train_epoch(model, optimizer, scheduler, train_dataloader, epoch):
    # 指定模型进入训练模式
    model.train()

    # 指定ignore_index,对其不计算梯度
    ignore_index = param.ignore_index
    epoch_start_time = datetime.now()  # 模型开始训练时间节点
    total_loss = 0
    epoch_correct_num = 0  # 每个epoch中，output计算正确的token的数量
    epoch_total_num = 0  # 每个epoch中，output的token总数

    for batch_index, (input_ids, labels) in enumerate(train_dataloader):

        # 将训练数据拉入GPU
        input_ids = input_ids.to(param.device)
        labels = labels.to(param.device)

        outputs = model.forward(input_ids=input_ids, labels=labels)
        logits = outputs.logits
        loss = outputs.loss
        loss = loss.mean()

        # 统计该batch的预测token的正确数与总数
        batch_correct_num, batch_total_num = calculate_acc(logits, labels, ignore_index)
        batch_acc = batch_correct_num / batch_total_num

        # 统计该epoch的预测token的正确数与总数
        epoch_correct_num += batch_correct_num
        epoch_total_num += batch_total_num

        total_loss += loss.item()

        if param.gradient_accumulation_steps > 1:
            loss = loss / param.gradient_accumulation_steps

        loss.backward()
        # 应用梯度累计策略
        torch.nn.utils.clip_grad_norm_(model.parameters(), param.max_grad_norm)

        # 进行一定step的梯度累积后，更新参数
        if (batch_index + 1) % param.gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if (batch_index + 1) % param.loss_step == 0:
            logger.info(f"第{batch_index + 1}个batch | 第{epoch + 1}个epoch | loss:{loss.item()}, 该batch准确率：{batch_acc}学利率：{param.learning_rate}")

        del input_ids, outputs

    # 记录当前epoch的平均损失之与准确率
    epoch_mean_loss = total_loss / len(train_dataloader)
    epoch_acc = epoch_correct_num / epoch_total_num
    logger.info(
        f"第{epoch + 1}个epoch | 平均loss:{epoch_mean_loss}, 平均准确率:{epoch_acc}, 训练时间:{datetime.now() - epoch_start_time}")

    if epoch % 10 == 0 or epoch == param.epochs:
        logger.info(f"第{epoch + 1}个epoch | 保存模型...")
        model_path = os.path.join(param.save_model_path, 'bj_epoch{}'.format(epoch + 1))
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        model.save_pretrained(model_path)
        epoch_finish_time = datetime.now()
        logger.success(f"第{epoch + 1}轮次训练结束。总耗时：{epoch_finish_time - epoch_start_time}")

    return epoch_mean_loss


def validate_epoch(model, validate_dataloader, epoch):
    logger.info(f"第{epoch + 1}个epoch | 开始验证...")
    model.eval()
    epoch_start_time = datetime.now()
    total_loss = 0

    with torch.no_grad():
        for batch_index, (input_ids, labels) in enumerate(validate_dataloader):
            input_ids = input_ids.to(param.device)
            labels = labels.to(param.device)
            outputs = model.forward(input_ids=input_ids, labels=labels)

            logits = outputs.logits
            loss = outputs.loss
            loss = loss.mean()

            total_loss += loss.item()
            del input_ids, outputs

        # 记录当前epoch的平均loss
        epoch_mean_loss = total_loss / len(validate_dataloader)
        logger.info(
            f"第{epoch + 1}个验证轮次 | 平均loss:{epoch_mean_loss}")
        epoch_finish_time = datetime.now()
        logger.success(f"第{epoch + 1}轮次验证结束。总耗时：{epoch_finish_time - epoch_start_time}")

        return epoch_mean_loss


def train(model, train_dataloader, valid_dataloader):
    # 计算模型训练完毕，一共迭代多少步()
    t_total = len(train_dataloader) // param.gradient_accumulation_steps * param.epochs

    try:
        # 定义优化器
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=param.learning_rate,
            eps=param.eps,
            weight_decay=0.01
        )

        # 指定学习率预热
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(t_total * param.warm_up_ratio),
            num_training_steps=t_total
        )
        logger.info("优化器初始化成功...")
    except Exception as e:
        logger.error(f"优化器初始化失败，错误信息：{e}")
        raise e
    best_val_loss = param.init_val_loss
    tran_losses, validate_losses = [], []

    logger.info("开始训练...")
    for epoch in range(param.epochs):
        # =================模型训练=================
        train_loss = train_epoch(model, optimizer, scheduler, train_dataloader, epoch)
        tran_losses.append(train_loss)
        # =================模型验证=================
        validate_loss = validate_epoch(model, valid_dataloader, epoch)
        validate_losses.append(validate_loss)

        # =================保存最佳模型=================
        if validate_loss < best_val_loss:
            best_val_loss = validate_loss
            logger.success(f"第{epoch + 1}轮次，炼出最佳模型，保存中...")
            model_path = os.path.join(param.save_model_path, 'min_ppl_model_bj'.format(epoch + 1))
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            model.save_pretrained(model_path)


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # 1. 加载模型
    try:
        model = AutoModelForCausalLM.from_pretrained(param.qwen2, trust_remote_code=True).half().cuda()
        logger.info("模型加载成功...")
    except Exception as e:
        logger.error(f"模型加载失败，错误信息：{e}")
        raise e
    # 2. 加载数据
    try:
        logger.info("开始加载数据...")
        train_dataloader, valid_dataloader = get_dataloader(param.pkl_data_path)
        logger.info(f"数据加载成功:训练集总数:{len(train_dataloader)},验证集总数:{len(valid_dataloader)}")
    except Exception as e:
        logger.error(f"数据加载失败，错误信息：{e}")
        raise e
    # 3. 开始训练
    train(model, train_dataloader, valid_dataloader)


if __name__ == '__main__':
    main()
