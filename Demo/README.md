# ScienceWorld 双通道记忆更新演示

这个目录提供一个离线可运行的原型演示，用来展示一次完整的双通道学习与记忆更新过程：

```text
ScienceWorld task
-> retrieve memory
-> cheap channel proposal
-> risk gate
-> optional slow refinement
-> feedback evaluation
-> memory update
```

## 运行方式

从项目根目录运行：

```powershell
python .\演示\demo_scienceworld_memory_loop.py
```

可选参数：

```powershell
python .\演示\demo_scienceworld_memory_loop.py `
  --risk-threshold 0.55 `
  --top-k 2 `
  --task-ids scienceworld-0 scienceworld-2 scienceworld-3 scienceworld-6
```

## 输出文件

脚本会写入：

```text
演示/outputs/demo_trace.jsonl
演示/outputs/demo_summary.md
演示/outputs/memory_snapshot.json
```

- `demo_trace.jsonl`：逐任务事件日志，包含 retrieval、cheap channel、risk gate、slow refinement 和 memory update。
- `demo_summary.md`：可直接放进报告或 PPT 讲稿的过程总结。
- `memory_snapshot.json`：最终记忆库快照，能看到 cheap memory 和 refined-template 的写入结果。
