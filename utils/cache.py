# import os
# import bmemcached
# from functools import lru_cache

# @lru_cache(maxsize=1)
# def get_memcache():
#     endpoint  = os.getenv("MEMCACHED_ENDPOINT")
#     port      = os.getenv("MEMCACHED_PORT")
#     username  = os.getenv("MEMCACHED_USERNAME")
#     password  = os.getenv("MEMCACHED_PASSWORD")

#     return bmemcached.Client(
#         (f"{endpoint}:{port}",),
#         username,
#         password
#     )