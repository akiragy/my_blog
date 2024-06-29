# 【缝合向】PyTorch + Redis + Elasticsearch + Feast + Triton + Flask构建推荐系统，倒排/向量召回 + DeepFM排序 + 推理引擎 + 后端服务

以推荐系统中最经典的MovieLens数据集为例，介绍推荐系统从离线到在线的全流程，所有操作都可以在一台笔记本上完成，虽然缝合的东西多，但所有东西都被封在了Conda和Docker里，不会对本地环境有任何伤害。

- Conda环境安装pandas和PyTorch模拟工业界的HDFS -> Spark -> GPU集群的离线模型训练。
- Conda环境安装Flask模拟工业界的Spring推荐后端。
- Docker环境安装Redis + Elasticsearch + Feast Feature Store + Triton Inference Server四个组件，用本机localhost调用Docker来模拟工业界的推荐后端RPC调用各个组件。其中，Redis用于存储召回所需的user标签和向量，Elasticsearch用于构建召回所需的item标签和向量索引，Feast用于存储排序所需的user和item特征，Triton用作排序所需的实时打分引擎。

整个推荐系统的架构图如下：

![alt text](images/blog5_1.png)

所有代码整理在了[github](https://github.com/akiragy/recsys_pipeline)仓库里，如果想直接运行代码的话，可以去看github的readme文件。

下面将分离线、离线到在线、在线三个阶段来介绍召回和排序模块的开发部署流程。

## 1 离线

### 1.1 预处理

https://github.com/akiragy/recsys_pipeline/blob/master/offline/preprocess/s1_data_split.py

https://github.com/akiragy/recsys_pipeline/blob/master/offline/preprocess/s2_term_trans.py

[MovieLens-1M](https://grouplens.org/datasets/movielens/1m/)共有6040个user对3952个item的1000209个评分，每个评分在1到5之间，还提供了每个评分的时间戳。user额外提供了gender、age、occupation、zip-code这4个特征，item额外提供了title和genres这2个特征。在构建推荐系统之前，我们先对数据进行一些预处理

- 标记：搜广推中隐式反馈推荐系统（点击/未点击）远远多于显式反馈推荐系统（评分），所以我们将ml数据集转化为隐式反馈：将评分大于3的视为正样本，评分小于等于3的视为负样本，共有575281正样本，424928负样本。
- 样本：我们将每个user的打分行为按时间戳排序，每个user最后10个评分的item用作online评估（60400个），对于剩余的样本，每个user的前80%评分的item用作offline训练集（754233个），后20%评分的item用作offline测试集（185576个）。
- 特征：工业界一般会对深度学习框架进行改造，加入hashtable来存储sparse特征的embedding，这样就不需要预先定义embedding table的shape，但这部分比较超纲了，这里还是使用pytorch原生提供的固定shape的embedding，所以要对sparse特征进行label encoder，将其转换为从0开始的自然数序列。

此外，值得注意的是s2_term_trans.py文件中的point-in-time joins，即在生产离线训练样本（imp_term.pkl）时，要使用在行为发生之时或之前且离现在最近的的特征，若使用之后的会出现特征泄露，之前但离现在较远的会产生线上线下不一致。而在特征上线时，要使用最新的特征（user_term.pkl和item_term.pkl）。

### 1.2 召回

#### 1.2.1 user和item标签生产

https://github.com/akiragy/recsys_pipeline/blob/master/offline/recall/s1_term_recall.py

term召回使用user过去一段时间窗口内行为过的item所对应的genre_list，即使用user偏好和item的genre匹配，待后续导入Redis和ES。

#### 1.2.2 user和item向量生产

https://github.com/akiragy/recsys_pipeline/blob/master/offline/recall/s2_vector_recall.py

vector召回使用FM，仅使用userid和itemid两个特征，AUC为0.8081。训练完成后解析出user和item的embedding，待后续导入Redis和ES。

### 1.3 排序

#### 1.3.1 user和item特征生产

https://github.com/akiragy/recsys_pipeline/blob/master/offline/rank/s1_feature_engi.py

#### 1.3.2 模型训练

https://github.com/akiragy/recsys_pipeline/blob/master/offline/rank/s2_model_train.py

模型使用DeepFM，AUC为0.8206。[pytorch-fm](https://github.com/rixwew/pytorch-fm)是一个极简的FM系列算法包，但有两个缺点：1.只支持sparse，不支持dense特征；2.所有sparse特征只能设为同样的维度，违反“id embedding应该维度高一点，side info embedding应该维度低一点”这一直觉。因此，没有直接安装使用pytorch-fm，而是基于源码进行了一些修改，使得既支持dense特征，又支持id和side info特征具有不同的embedding维度。此外，观察到deep embedding部分会使得模型效果下降，因此删掉了这一部分，所以模型结构本质是sparse FM + dense MLP，不算严格的DeepFM。

值得注意的是，线上的样本特征是从Feast里实时抽取的，所以需要一个线上线下一致的特征meta定义。

```python
# 来自data_exchange_center/parse_feat_meta.py
from data_exchange_center.constants import *


def get_feature_def(feat_meta):
    sparse_id_dims = []
    sparse_side_dims = []
    user_sparse, item_sparse, user_dense, item_dense = feat_meta["sparse"]["user"], feat_meta["sparse"]["item"], \
                                                       feat_meta["dense"]["user"], feat_meta["dense"]["item"]
    # parse id features
    sparse_id_dims.append(user_sparse[USERID][1])
    sparse_id_dims.append(item_sparse[ITEMID][1])
    sparse_id_feat = [USERID, ITEMID]
    del user_sparse[USERID], item_sparse[ITEMID]

    # parse other sparse features
    for col in user_sparse:
        sparse_side_dims.append(user_sparse[col][1])
    for col in item_sparse:
        sparse_side_dims.append(item_sparse[col][1])
    sparse_side_feat = list(user_sparse.keys()) + list(item_sparse.keys())

    # parse dense features
    dense_dim = len(user_dense) + len(item_dense)
    dense_feat = list(user_dense.keys()) + list(item_dense.keys())
    # print(f"sparse_id_dims: {sparse_id_dims}, sparse_side_dims: {sparse_side_dims}, dense_dim: {dense_dim}, sparse_id_feat: {sparse_id_feat}, sparse_side_feat: {sparse_side_feat}, dense_feat: {dense_feat}")
    return sparse_id_dims, sparse_side_dims, dense_dim, sparse_id_feat, sparse_side_feat, dense_feat
```

## 2 离线到在线

### 2.1 环境安装

为了不对本地环境产生影响，Redis、Elasticsearch、Feast、Triton都是以docker的形式使用，这一步要先下载镜像（需要确保本地装了[docker](https://www.docker.com/)）。

```bash
docker pull redis:6.0.0
docker pull elasticsearch:8.8.0
docker pull feastdev/feature-server:0.31.0
docker pull nvcr.io/nvidia/tritonserver:20.12-py3
```

### 2.2 召回

#### 2.2.1 user标签和向量进Redis

大名鼎鼎的[Redis](https://redis.io/) 就不用介绍了，我们使用它来存储召回所需的user信息。

启动Redis容器，-d参数保持后台执行。

```bash
docker run --name redis -p 6379:6379 -d redis:6.0.0
```

将user的term、vector、filter导入Redis，其中term和vector产出自1.2部分，filter是user历史行为过的item，再推荐时要过滤掉这些。

代码在导入数据后还会以某个用户为例进行数据校验，如果有下图输出，代表校验成功。

https://github.com/akiragy/recsys_pipeline/blob/master/offline_to_online/recall/s1_user_to_redis.py

![alt text](images/blog5_2.png)

注意：本地环境应该conda install redis-py，不要conda install redis，否则安装的是redis数据库，我们已经在docker里安装了数据库，conda里装的应该是与数据库建立连接的client。

#### 2.2.2 item标签和向量进ES

[Elasticsearch](https://www.elastic.co/) 的大名也是无人不知，我们使用它来为item构建倒排索引和向量索引。它最初被用于搜索领域，最原始的用法是用word去检索doc，如果我们将一个item视为一篇doc，它的标签（如电影类别）视为word，就可以借助ES来根据标签检索item，这就是倒排索引的概念，因此Elasticsearch也常被用于推荐系统的term召回模块。对于向量召回，经典工具是facebook开源的faiss，但是为了方便整合，我们在这里使用Elasticsearch提供的向量检索功能，Elasticsearch自版本7开始支持向量检索，版本8开始支持近似KNN检索算法，这里我们安装的是8及以后的版本，因为精确KNN检索的性能几乎不可能满足线上使用。

启动ES容器，-it参数进入内部终端。

```bash
docker run --name es8 -p 9200:9200 -it elasticsearch:8.8.0
```

复制下图位置的密码，粘贴为data_exchange_center/constants.py文件中ES_KEY的值，因为ES从版本8开始要强制进行密码验证。

![alt text](images/blog5_3.png)

粘贴完成后，ctrl+C（或command+C）退出内部终端，此时容器也会停止运行，所以我们需要重新启动ES容器并保持后台执行。

```bash
docker start es8
```

将item的term导入构建标签索引，vector导入构建向量索引，工业界为了性能和灵活性会将两个索引拆分，但这里简单起见我们将索引合二为一。

代码在导入数据后还会以某个term和vector为例进行数据校验，如果有下图输出，代表校验成功。

https://github.com/akiragy/recsys_pipeline/blob/master/offline_to_online/recall/s2_item_to_es.py


![alt text](images/blog5_4.png)

### 2.3 排序

#### 2.3.1 user和item特征进Feast

[Feast](https://feast.dev/) 是第一个开源的特征仓库，有历史里程碑意义，它有离线和在线两部分功能，离线部分主要提供的功能就是point-in-time joins，因为我们在pandas中自己处理了pit，所以没必要再使用Feast的离线功能，只使用它的线上特征服务即可。关于为什么不使用Feast进行pit，是因为feast本身只有特征存储的能力，不具备特征工程的能力（最新版本支持一部分简单变换），各大公司还是喜欢自研功能更强大的特征引擎，所以不需要花太多时间去研究它的离线用法，还是用更通用的pandas或者spark处理完特征，只把它当作一个离在线特征同步器来使用吧。

将特征文件由csv转为parquet格式以满足Feast的要求。

https://github.com/akiragy/recsys_pipeline/blob/master/offline_to_online/rank/s1_feature_to_feast.py

新启动一个终端（称为终端2），先切换到预先建好的仓库目录下，再启动docker，-it进入容器内部终端，-v将当前目录的特征配置和数据同步至docker内。

```bash
cd data_exchange_center/online/feast
docker run --rm --name feast-server --entrypoint "bash" -v $(pwd):/home/hs -p 6566:6566 -it feastdev/feature-server:0.31.0
```

下面的命令在docker容器内部终端执行。

```bash
# 进入同步目录
cd /home/hs/ml/feature_repo
# 根据配置文件初始化特征仓库（读取parquet建立db）
feast apply
# 特征由离线同步至在线（默认sqlite，可以切换Redis等）
feast materialize-incremental "$(date +'%Y-%m-%d %H:%M:%S')"
# 启动特征服务
feast serve -h 0.0.0.0 -p 6566
```

全部执行完成后，会有下图输出。

![alt text](images/blog5_5.png)

回到终端1输入以下命令，测试Feast是否已正常服务，如果成功会返回一串json。


```bash
curl -X POST \
  "http://localhost:6566/get-online-features" \
  -d '{
    "feature_service": "ml_item",
    "entities": {
      "itemid": [1,3,5]
    }
  }'
```

注意：因为Feast的conda包安装失败了，用pip装又容易破坏全conda的环境，所以这一步没有在本地装Feast。这么做也是有理由的：与Redis和Elasticsearch不同的是，Feast并不区分server端和client端，前文所谓的Feast Server本质就是把Feast本体封装进了docker而已，所以只要在docker中装了之后，本地conda环境不装也没问题，用http请求docker即可。

#### 2.3.2 PyTorch转ONNX进Triton

[Triton](https://developer.nvidia.com/triton-inference-server) 全称是Triton Inference Server，是Nvidia开源的全模型serving引擎，支持TensorFlow、PyTorch、ONNX和其他各种模型，虽然是N家的产品，但是也可以使用cpu进行serving，所以请放心使用。业界更通用的方案是TensorFlow -> SavedModel -> TF Serving，但Triton因为不会被绑定在一家平台上，个人非常看好它的前景，所以这里使用的是PyTorch -> ONNX -> Triton Server的方案。

将pytorch模型导出为onnx格式。

https://github.com/akiragy/recsys_pipeline/blob/master/offline_to_online/rank/s2_model_to_triton.py

新启动一个终端（称为终端3），先切换到预先建好的仓库目录下，再启动docker，-v将当前目录的模型同步至docker内，会有下图输出，8000是http接口，8001是grpc接口。

```bash
cd data_exchange_center/online/triton
docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 -v $(pwd)/:/models/ nvcr.io/nvidia/tritonserver:20.12-py3 tritonserver --model-repository=/models/
```

![alt text](images/blog5_6.png)

Triton要求每个模型仓库都有一个config.pbtxt声明模型signature，这里使用的配置如下：

```
name: "ml_rec"
platform: "onnxruntime_onnx"
max_batch_size: 128
input [
{
    name: "ml_input",
    data_type: TYPE_FP32,
    dims: [59]
}
]
output [
{
    name: "ml_output",
    data_type: TYPE_FP32,
    dims: [1]
    reshape: { shape: [ ] }
}
]
default_model_filename: "model"
```

回到终端1运行以下命令，测试模型线下与线上打分是否一致：构造两个测试样本，以gRPC方式请求Triton获取online打分，再加载导出的pt模型对同样样本获取offline打分，比较offline和online打分是否一致。

https://github.com/akiragy/recsys_pipeline/blob/master/offline_to_online/rank/s3_check_offline_and_online.py

如图，线下与线下打分完全一致，测试通过。

![alt text](images/blog5_7.png)

注意：本地不装tritonclient也是可以的，因为可以用http调用的方式，但还是建议安装并使用grpc调用，速度会快很多。运行tritonclient如果报错了很有可能是缺grpc的包，conda安装即可。

## 3 在线

### 3.1 服务端启动

Python常用的web框架有Django, Flask, FastAPI, Tornado等，都可以实现REST API请求 -> url路由到某个函数 -> 处理的逻辑，各有优缺点，随机选取了Flask。这里使用Python作为后端仅仅是因为环境安装方便，工业界的推荐系统一般会使用Java + SpringBoot或者Go + Gin作为web后端。

下面是[s1_server.py](https://github.com/akiragy/recsys_pipeline/blob/master/online/main/s1_server.py)文件的内容，接收POST请求并依次执行Request解析、召回、排序、Response生成。因为请求item特征的QPS要远远高于请求user特征（每个user都要请求几百上千个item），所以在服务启动时预先将所有item的特征加载到了内存里（在FeastCaller里实现）。

```python
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from flask import Flask, request
import json
import logging
from data_exchange_center.constants import *
from online.utils.dto import *
from online.utils.redis_caller import RedisCaller
from online.utils.es_caller import ESCaller
from online.utils.feast_caller import FeastCaller
from online.utils.triton_caller import TritonCaller
logging.getLogger().setLevel(logging.INFO)


# init
redis_caller = RedisCaller()
es_caller = ESCaller()
triton_caller = TritonCaller()
feast_caller = FeastCaller()
app = Flask(__name__)


@app.route('/ml/rec', methods=['POST'])
def ml_rec():
    # parse request
    rec_data = build_rec_data(json.loads(request.data))

    # core
    recall(rec_data)
    rank(rec_data)

    # build response
    res = {"item_list": []}
    for i in range(min(rec_data.config.response_size, len(rec_data.item_list))):
        d = {
            "itemid": rec_data.item_list[i].itemid,
            "score": rec_data.item_list[i].score
        }
        res["item_list"].append(eval(str(d)))
    return res


def build_rec_data(rec_request: json) -> RecData:
    rec_data = RecData()
    rec_data.user_info.userid = rec_request["userid"]
    return rec_data


def recall(rec_data: RecData) -> None:
    redis_caller.call(rec_data)
    es_caller.call(rec_data)


def rank(rec_data: RecData) -> None:
    feast_caller.call(rec_data)
    triton_caller.call(rec_data)
```

新启动一个终端（称为终端4），输入以下命令启动flask web server，会有下图输出，服务运行在5000端口。server被调用时会输出一些日志，后续可以观察。

```bash
# 启动后端服务
flask --app s1_server.py run --host=0.0.0.0
```

![alt text](images/blog5_8.png)

### 3.2 召回

#### 3.2.1 userid请求Redis获取user信息

下面是[redis_caller.py](https://github.com/akiragy/recsys_pipeline/blob/master/online/utils/redis_caller.py)文件的内容，连接docker中6379端口的Redis，获取user的term（标签召回）、vector（向量召回）、filter（记录已经看过的item）。

```python
from redis import Redis, ConnectionPool
import logging
from data_exchange_center.constants import *
from online.utils.dto import *


class RedisCaller:
    def __init__(self):
        pool = ConnectionPool(host="localhost", port=6379, decode_responses=True)
        self.redis_obj = Redis(connection_pool=pool)

    def call(self, rec_data: RecData) -> None:
        userid = rec_data.user_info.userid
        test_user_key = RECALL_REDIS_PREFIX + str(userid)
        user_term = self.redis_obj.hget(test_user_key, RECALL_REDIS_TERM_KEY)
        user_vector = self.redis_obj.hget(test_user_key, RECALL_REDIS_VECTOR_KEY)
        user_filter = self.redis_obj.hget(test_user_key, RECALL_REDIS_FILTER_KEY)

        if user_term is None or len(user_term) == 0:
            rec_data.user_info.user_term = []
            logging.warning(f"user {userid} term empty")
        else:
            rec_data.user_info.user_term = [int(v) for v in user_term.split(",")]

        if user_vector is None or len(user_vector) == 0:
            rec_data.user_info.user_vector = [1] * RECALL_EMB_DIM
            logging.warning(f"user {userid} vector empty")
        else:
            rec_data.user_info.user_vector = [float(v) for v in user_vector.split(",")]

        if user_filter is None or len(user_filter) == 0:
            rec_data.user_info.user_filter = []
            logging.warning(f"user {userid} filter empty")
        else:
            rec_data.user_info.user_filter = [int(v) for v in user_filter.split(",")]

        logging.info(f"******user {userid}, redis done******")
```

#### 3.2.2 user信息请求ES获取召回itemid

下面是[es_caller.py](https://github.com/akiragy/recsys_pipeline/blob/master/online/utils/es_caller.py)文件的内容，连接docker中9200端口的ES，构建一个term请求和一个vector knn请求，过滤看过的item，获取召回itemid_list。

```python
from elasticsearch import Elasticsearch
import logging
from data_exchange_center.constants import *
from online.utils.dto import *
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class ESCaller:
    def __init__(self):
        self.es_obj = Elasticsearch(f"https://elastic:{ES_KEY}@localhost:9200", verify_certs=False)

    def call(self, rec_data: RecData) -> None:
        user_term = rec_data.user_info.user_term
        user_vector = rec_data.user_info.user_vector
        user_filter = rec_data.user_info.user_filter

        term_query = {
            "bool": {
                "must": {
                    "terms": {
                        GENRES: user_term,
                        "boost": 0.1
                    }
                },
                "filter": {
                    "bool": {
                        "must_not": [
                            {
                                "terms": {
                                    ITEMID: user_filter
                                }
                            }
                        ]
                    }
                }
            }
        }

        vector_query = {
            "field": ITEM_VECTOR,
            "query_vector": user_vector,
            "k": rec_data.config.recall_size,
            "num_candidates": MAX_ITEMID,
            "boost": 0.9,
            "filter": {
                "bool": {
                    "must_not": {
                        "terms": {
                            ITEMID: user_filter
                        }
                    }
                }
            }
        }

        cnt = 0
        res = self.es_obj.search(index=ITEM_ES_INDEX, knn=vector_query, query=term_query,
                                 size=rec_data.config.recall_size)
        for i in range(len(res["hits"]["hits"])):
            cnt += 1
            itemid = res["hits"]["hits"][i]["_source"]["itemid"]
            item_info = ItemInfo()
            item_info.itemid = itemid
            rec_data.item_list.append(item_info)
        logging.info(f"******user {rec_data.user_info.userid}, es done, recall_cnt {cnt}******")
```

### 3.3 排序

#### 3.3.1 userid请求Feast获取user特征

下面是[feast_caller.py](https://github.com/akiragy/recsys_pipeline/blob/master/online/utils/feast_caller.py)文件的内容，连接docker中6566端口的Feast，服务启动时先调用get_all_item_feature方法完成item特征的预拉取（放在内存中，降低IO开销），在每个请求时再获取user特征。

```python
import requests
import json
import pandas as pd
import logging
from data_exchange_center.constants import *
from online.utils.dto import *


class FeastCaller:
    def __init__(self):
        self.url = "http://localhost:6566/get-online-features"

    def call(self, rec_data: RecData) -> None:
        userid = rec_data.user_info.userid
        user_feature_request = {
            "feature_service": "ml_user",
            "entities": {
                USERID: [userid]
            }
        }
        response = json.loads(
            requests.post(self.url, data=json.dumps(user_feature_request)).text)

        user_feat_names = response["metadata"]["feature_names"]
        if len(user_feat_names) != len(response["results"]):
            logging.error(f"******feast user feature num error, user {userid}******")
            rec_data.user_info.user_feature = {}
            return

        user_feature = pd.DataFrame()
        for i, col in enumerate(user_feat_names):
            user_feature[col] = response["results"][i]["values"]
        rec_data.user_info.user_feature = user_feature.iloc[0].to_dict()
        logging.info(f"******user {userid}, feast done******")

    def get_all_item_feature(self):
        item_feature_request = {
            "feature_service": "ml_item",
            "entities": {
                ITEMID: [v for v in range(MAX_ITEMID+1)]
            }
        }
        response = json.loads(
            requests.post(self.url, data=json.dumps(item_feature_request)).text)

        item_feat_names = response["metadata"]["feature_names"]
        if len(item_feat_names) != len(response["results"]) or len(item_feat_names) == 0:
            logging.error(f"******feast item feature num error, cannot start server******")
            return None
        all_item_feature = pd.DataFrame()
        for i, col in enumerate(item_feat_names):
            all_item_feature[col] = response["results"][i]["values"]
        logging.info("****** feast item feature prefetch done******")
        return all_item_feature
```

#### 3.3.2 user和item特征请求Triron获取打分

下面是[triton_caller.py](https://github.com/akiragy/recsys_pipeline/blob/master/online/utils/triton_caller.py)文件的内容，连接docker中8001端口的Triton，分partition（防止batch_size过大对Triton造成压力）获取每个<user, item>对的打分，并按照score对item排序。

```python
import tritonclient.grpc as triton_client
import numpy as np
import logging
import pickle
from data_exchange_center.parse_feat_meta import get_feature_def
from data_exchange_center.paths import FEAT_META_PATH
from online.utils.dto import *
from online.utils.feast_caller import FeastCaller


class TritonCaller:
    def __init__(self):
        self.url = "localhost:8001"

        feat_meta = pickle.load(open(FEAT_META_PATH, "rb"))
        _, _, _, sparse_id_feat, sparse_side_feat, dense_feat = get_feature_def(feat_meta)

        self.sparse_id_feat = sparse_id_feat
        self.sparse_side_feat = sparse_side_feat
        self.dense_feat = dense_feat
        self.all_item_feature = FeastCaller().get_all_item_feature()
        assert self.all_item_feature is not None

    def call(self, rec_data: RecData) -> None:
        all_feature = rec_data.user_info.user_feature

        n_items = len(rec_data.item_list)
        for i in range(n_items):
            rec_data.item_list[i].item_feature = self.all_item_feature.iloc[rec_data.item_list[i].itemid].to_dict()

        has_none_col = set()

        partition_size = rec_data.config.rank_partition_size
        mb_iter = n_items // partition_size if n_items % partition_size == 0 else n_items // partition_size + 1
        for mb in range(mb_iter):
            samples = []
            # sample building
            for i in range(mb * partition_size, min((mb + 1) * partition_size, n_items)):
                sample = []
                item_feature = rec_data.item_list[i].item_feature
                all_feature.update(item_feature)
                for col in self.sparse_id_feat + self.sparse_side_feat:
                    if all_feature[col] is None:
                        has_none_col.add(col)
                        sample.append(0)
                    else:
                        sample.append(all_feature[col])
                for col in self.dense_feat:
                    if all_feature[col] is None:
                        has_none_col.add(col)
                        sample.append(0.0)
                    else:
                        sample.append(all_feature[col])
                samples.append(sample)

            # online prediction
            client = triton_client.InferenceServerClient(url=self.url)
            samples_np = np.array(samples)
            input_name = "ml_input"
            inputs = [triton_client.InferInput(input_name, samples_np.shape, "FP32")]
            inputs[0].set_data_from_numpy(samples_np.astype(np.float32))
            output_name = "ml_output"
            outputs = [triton_client.InferRequestedOutput(output_name)]

            response = client.infer(model_name="ml_rec",
                                    inputs=inputs,
                                    outputs=outputs)
            online_pred = response.as_numpy(output_name)
            for i in range(mb * partition_size, min((mb + 1) * partition_size, n_items)):
                rec_data.item_list[i].score = online_pred[i - mb * partition_size][0]

        if len(has_none_col) != 0:
            logging.warning(f"triton user {rec_data.user_info.userid} has none col!, {has_none_col}")

        from operator import attrgetter
        rec_data.item_list = sorted(rec_data.item_list, key=attrgetter('score'), reverse=True)
        logging.info(f"******user {rec_data.user_info.userid}, triton done******")
```

### 3.4 客户端请求

下面是[s2_client.py](https://github.com/akiragy/recsys_pipeline/blob/master/online/main/s2_client.py)文件的内容，调用5000端口的Flask服务，发送包含userid的POST请求获取推荐结果。

```python
import requests
import json

data = {
    "userid": 1
}

response = requests.post(
    "http://localhost:5000/ml/rec",
    data=json.dumps(data))
print(response.text)
```

回到终端1，测试客户端（这里的客户端不是指用户终端设备，而是指调用推荐服务的上游服务）调用，获取json格式的推荐结果，下图展示了top50的推荐列表，下游服务取这些item对应的字段并返回给客户端（用户终端），一个完整的推荐请求就结束了。

```bash
# 客户端调用（可以修改为其他userid测试）
python s2_client.py
```

![alt text](images/blog5_9.png)
