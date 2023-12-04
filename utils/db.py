from redis import StrictRedis

def get_redis_client():
    """获取 Redis 客户端
    """
    return StrictRedis(host='127.0.0.1', port=38148, db=0, password="heu479")

redis_conn = get_redis_client()