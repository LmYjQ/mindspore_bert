# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""
Bert evaluation script.
"""

import os
import argparse
from mindspore import context
from mindspore import log as logger
import mindspore.dataset as de
import mindspore.dataset.transforms.c_transforms as C
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.evaluation_config import cfg, bert_net_cfg
from src.utils import BertNER, BertCLS
from src.CRF import postprocess
from src.finetune_config import tag_to_index
import moxing as mox
import numpy as np
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from src.evaluation_config import cfg
import json

access_key = 'BT1ISWESQZEPX9DEGL9N'
secret_key = 'rvcT7qK6WTM39pDZSnu2EzkXgqrAxrfQ7F32Sayi'
mox.file.set_auth(ak=access_key, sk=secret_key, server="obs.cn-north-4.myhuaweicloud.com")


class Accuracy():
    '''
    calculate accuracy
    '''

    def __init__(self):
        self.acc_num = 0
        self.total_num = 0

    def update(self, logits, labels):
        labels = labels.asnumpy()
        labels = np.reshape(labels, -1)
        logits = logits.asnumpy()
        logit_id = np.argmax(logits, axis=-1)
        self.acc_num += np.sum(labels == logit_id)
        self.total_num += len(labels)
        print("=========================accuracy is ", self.acc_num / self.total_num)
        return logit_id.tolist()

class F1():
    '''
    calculate F1 score
    '''

    def __init__(self):
        self.TP = 0
        self.FP = 0
        self.FN = 0

    def update(self, logits, labels):
        '''
        update F1 score
        '''
        labels = labels.asnumpy()
        labels = np.reshape(labels, -1)
        if cfg.use_crf:
            backpointers, best_tag_id = logits
            best_path = postprocess(backpointers, best_tag_id)
            logit_id = []
            for ele in best_path:
                logit_id.extend(ele)
        else:
            logits = logits.asnumpy()
            logit_id = np.argmax(logits, axis=-1)
            logit_id = np.reshape(logit_id, -1)
        pos_eva = np.isin(logit_id, [i for i in range(1, cfg.num_labels)])
        pos_label = np.isin(labels, [i for i in range(1, cfg.num_labels)])
        self.TP += np.sum(pos_eva & pos_label)
        self.FP += np.sum(pos_eva & (~pos_label))
        self.FN += np.sum((~pos_eva) & pos_label)


def get_dataset(batch_size=1, repeat_count=1, distribute_file=''):
    '''
    get dataset
    '''
    _ = distribute_file

    ds = de.TFRecordDataset([cfg.data_file], cfg.schema_file, columns_list=["input_ids", "input_mask",
                                                                            "segment_ids", "label_ids"])
    type_cast_op = C.TypeCast(mstype.int32)
    ds = ds.map(input_columns="segment_ids", operations=type_cast_op)
    ds = ds.map(input_columns="input_mask", operations=type_cast_op)
    ds = ds.map(input_columns="input_ids", operations=type_cast_op)
    ds = ds.map(input_columns="label_ids", operations=type_cast_op)
    ds = ds.repeat(repeat_count)

    # apply shuffle operation
    buffer_size = 960
    ds = ds.shuffle(buffer_size=buffer_size)

    # apply batch operations
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds


def bert_predict(Evaluation):
    '''
    prediction function
    '''
    target = args_opt.device_target
    if target == "Ascend":
        devid = int(os.getenv('DEVICE_ID'))
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=devid)
    elif target == "GPU":
        context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
        if bert_net_cfg.compute_type != mstype.float32:
            logger.warning('GPU only support fp32 temporarily, run with fp32.')
            bert_net_cfg.compute_type = mstype.float32
    else:
        raise Exception("Target error, GPU or Ascend is supported.")
    dataset = get_dataset(bert_net_cfg.batch_size, 1)
    if cfg.use_crf:
        net_for_pretraining = Evaluation(bert_net_cfg, False, num_labels=len(tag_to_index), use_crf=True,
                                         tag_to_index=tag_to_index, dropout_prob=0.0)
    else:
        net_for_pretraining = Evaluation(bert_net_cfg, False, num_labels)
    net_for_pretraining.set_train(False)
    param_dict = load_checkpoint(cfg.finetune_ckpt)
    load_param_into_net(net_for_pretraining, param_dict)
    model = Model(net_for_pretraining)
    return model, dataset


def test_eval():
    '''
    evaluation function
    '''
    task_type = BertCLS
    model, dataset = bert_predict(task_type)
    labels = [] # 预测结果
    if cfg.clue_benchmark:
        print('不用这个')
    else:
        callback = Accuracy()
        columns_list = ["input_ids", "input_mask", "segment_ids", "label_ids"]
        for data in dataset.create_dict_iterator():
            input_data = []
            for i in columns_list:
                input_data.append(Tensor(data[i]))
            input_ids, input_mask, token_type_id, label_ids = input_data
            logits = model.predict(input_ids, input_mask, token_type_id, label_ids)
            preds = callback.update(logits, label_ids)
            labels.extend(preds)
        lines = [json.dumps({"label": res}, ensure_ascii=False) for res in labels]
        open(pred_file, "w").write("\n".join(lines))

        print("==============================================================")
        if cfg.task == "NER":
            print("Precision {:.6f} ".format(callback.TP / (callback.TP + callback.FP)))
            print("Recall {:.6f} ".format(callback.TP / (callback.TP + callback.FN)))
            print("F1 {:.6f} ".format(2 * callback.TP / (2 * callback.TP + callback.FP + callback.FN)))
        else:
            print("acc_num {} , total_num {}, accuracy {:.6f}".format(callback.acc_num, callback.total_num,
                                                                      callback.acc_num / callback.total_num))
        print("==============================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bert eval')
    parser.add_argument('--device_target', type=str, default='Ascend', help='Device target')
    parser.add_argument('--train_url', type=str, default='aaa', help='兼容modelarts')
    parser.add_argument('--data_url', type=str, default='bbb', help='兼容modelarts')
    args_opt = parser.parse_args()
    num_labels = cfg.num_labels
    print('*********ls cache',os.listdir('/cache'))

    mox.file.copy_parallel('obs://imdb-lyq/checkpoint', './checkpoint')
    mox.file.copy_parallel('obs://imdb-lyq/data', './tnews_data')
    pred_file = "test_predict.json"
    test_eval()
    mox.file.copy(pred_file, 'obs://imdb-lyq/data/'+pred_file)
