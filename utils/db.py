from redis import StrictRedis

def get_redis_client():
    """获取 Redis 客户端
    """
    return StrictRedis(host='localhost', port=6379, db=0)
