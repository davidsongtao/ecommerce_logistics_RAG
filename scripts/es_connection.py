from elasticsearch import Elasticsearch
import urllib3
import warnings

# 忽略警告
urllib3.disable_warnings()
warnings.filterwarnings("ignore")

# 连接配置
ES_URL = "https://es-3omy2u5o.public.tencentelasticsearch.com:9200"
ES_USER = "elastic"
ES_PASSWORD = "Dst881009."

# 创建客户端实例
es = Elasticsearch(
    [ES_URL],  # 7.x版本需要传入列表
    http_auth=(ES_USER, ES_PASSWORD),  # 7.x版本的认证方式
    verify_certs=False,
    timeout=30
)

# 测试连接
try:
    if es.ping():
        print("成功连接到ES!")
        # 打印集群信息
        print(es.info())
    else:
        print("连接失败!")

    # 获取更详细的信息
    health = es.cluster.health()
    print("集群健康状态:", health)

except Exception as e:
    print(f"连接出错: {str(e)}")
    print(f"错误类型: {type(e)}")
