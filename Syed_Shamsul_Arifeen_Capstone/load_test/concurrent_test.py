from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
import requests
def request(_): return requests.post('http://localhost:8000/predict',json={'text':'I disagree respectfully.'},timeout=15).status_code
start=perf_counter()
with ThreadPoolExecutor(max_workers=20) as pool: codes=list(pool.map(request,range(200)))
print({'requests':200,'successes':codes.count(200),'seconds':round(perf_counter()-start,2)})
