import time
from fastapi import HTTPException


async def verify_api_key_rate_limit(config, api_key):
    # NOTE: abit dangerous but very useful
    if not config.prod:
        if api_key == "test":
            return True

    rate_limit_key = f"rate_limit:{api_key}"
    rate_limit = await config.redis_db.get(rate_limit_key)
    if rate_limit is None:
        async with await config.psql_db.connection() as connection:
            # rate_limit = await get_api_key_rate_limit(connection, api_key)
            if rate_limit is None:
                raise HTTPException(status_code=403, detail="Invalid API key")
        await config.redis_db.set(rate_limit_key, rate_limit, ex=30)
    else:
        rate_limit = int(rate_limit)

    minute = time.time() // 60
    current_rate_limit_key = f"current_rate_limit:{api_key}:{minute}"
    current_rate_limit = await config.redis_db.get(current_rate_limit_key)
    if current_rate_limit is None:
        current_rate_limit = 0
        await config.redis_db.expire(current_rate_limit_key, 60)
    else:
        current_rate_limit = int(current_rate_limit)

    await config.redis_db.incr(current_rate_limit_key)
    if current_rate_limit >= rate_limit:
        raise HTTPException(status_code=429, detail="Too many requests")
