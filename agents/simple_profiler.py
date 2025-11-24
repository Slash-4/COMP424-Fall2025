import time
import functools

class SimpleProfiler:
  def __init__(self):
    self.data = {}  # label -> {'time': float, 'count': int}

  def _record(self, label, elapsed):
    d = self.data.setdefault(label, {'time': 0.0, 'count': 0})
    d['time'] += elapsed
    d['count'] += 1

  def profile(self, label):
    def decorator(fn):
      @functools.wraps(fn)
      def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
          return fn(*args, **kwargs)
        finally:
          t1 = time.perf_counter()
          self._record(label, t1 - t0)
      return wrapper
    return decorator

  def report(self, top=10):
    items = sorted(self.data.items(), key=lambda kv: kv[1]['time'], reverse=True)[:top]
    out = ["Profiler report (label, total_time_s, calls, avg_s):"]
    for label, v in items:
      avg = v['time'] / v['count'] if v['count'] else 0.0
      out.append(f"{label:30} {v['time']:.6f}s  {v['count']:6d}  avg={avg:.6f}s")
    return "\n".join(out)