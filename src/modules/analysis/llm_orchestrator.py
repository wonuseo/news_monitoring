"""
llm_orchestrator.py - Shared Chunked Parallel LLM Runner
진행률, 청크 통계, 동기화 콜백을 공통 처리한다.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import traceback
from typing import Callable, Dict, List, Optional

from tqdm import tqdm


WorkerFn = Callable[[Dict], Dict]
SuccessFn = Callable[[Dict], None]
FailureFn = Callable[[Dict, int, int], None]
SyncFn = Callable[[int, int, int, int], None]


def run_chunked_parallel(
    tasks: List[Dict],
    worker_fn: WorkerFn,
    on_success: SuccessFn,
    on_failure: FailureFn,
    *,
    chunk_size: int = 50,
    max_workers: int = 3,
    progress_desc: str = "LLM 분석",
    unit: str = "기사",
    inter_chunk_sleep: float = 0.5,
    sync_callback: Optional[SyncFn] = None,
) -> Dict[str, int]:
    """
    공통 병렬 실행기.

    Returns:
        {"processed": int, "success": int, "failed": int, "chunks": int}
    """
    total = len(tasks)
    if total == 0:
        return {"processed": 0, "success": 0, "failed": 0, "chunks": 0}

    total_chunks = (total + chunk_size - 1) // chunk_size
    total_success = 0
    total_failed = 0

    pbar = tqdm(total=total, desc=progress_desc, unit=unit)

    for chunk_start in range(0, total, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total)
        chunk_tasks = tasks[chunk_start:chunk_end]
        chunk_num = (chunk_start // chunk_size) + 1

        chunk_success = 0
        chunk_failed = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(worker_fn, task): task
                for task in chunk_tasks
            }

            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    if not isinstance(result, dict):
                        raise TypeError("worker_fn must return dict")
                except Exception as e:
                    result = {
                        "success": False,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "error_trace": traceback.format_exc(),
                        "task_id": task.get("task_id"),
                    }

                if result.get("success"):
                    on_success(result)
                    chunk_success += 1
                    total_success += 1
                else:
                    chunk_failed += 1
                    total_failed += 1
                    on_failure(result, chunk_failed, total_failed)

                pbar.update(1)

        pbar.set_postfix({
            "청크": f"{chunk_num}/{total_chunks}",
            "성공": chunk_success,
            "실패": chunk_failed
        })

        chunk_total = chunk_success + chunk_failed
        success_rate = (chunk_success / chunk_total * 100) if chunk_total > 0 else 0.0
        print(
            f"\n  청크 {chunk_num}/{total_chunks} 완료: "
            f"성공 {chunk_success}/{chunk_total} ({success_rate:.1f}%)"
        )

        if sync_callback:
            try:
                sync_callback(chunk_num, total_chunks, chunk_success, chunk_failed)
            except Exception as e:
                print(f"    ⚠️  청크 동기화 실패: {e}")

        if chunk_num < total_chunks and inter_chunk_sleep > 0:
            time.sleep(inter_chunk_sleep)

    pbar.close()

    return {
        "processed": total_success + total_failed,
        "success": total_success,
        "failed": total_failed,
        "chunks": total_chunks,
    }
