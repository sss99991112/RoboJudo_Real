# Current task

Goal: 冻结 UniLab G1 三专家学生，接通 RoboJudo 独立仿真部署链；完成源一致性及有界验证。
Do not: 不连接/控制真机、不训练、不改奖励或稳定阈值、不覆盖旧权重、不上传模型或本地原始证据。用户随后授权将本次迁移代码提交并推送到 GitHub 的独立迁移分支。

Current identity:
- Target: `/Users/sss9999/enbodied_intelligence/Roboju/RoboJudo_Real`，基线 `release/d0a0ec73686fdf21ff72ad20e79bef3553ada46a`；迁移分支 `codex/g1-recovery-sim-migration`；代码提交/推送仅限该分支，未合并 `release`，发布状态以 Git 为准。
- Source: `/Users/sss9999/enbodied_intelligence/unilab/unilab_stand`，`codex/g1-recovery-moe-integration/e1717b9462f74fcccb430889d36ecf121a1fe1e5`。源 dirty patch 与交接 `source-worktree.patch` 逐字节相同。
- Student: `/Users/sss9999/enbodied_intelligence/unilab/dagger_iteration_2.pt`，SHA256 `27d06b8d73734fb2a17c730aa63ff70b2ba4ffbef7d3d71373df43974ab9bf18`。
- 新导出 `assets/models/g1/unilab/g1_three_expert_dagger_iteration_2/policy.pt`，SHA256 `1435ecffdb457eb8d2cc41445c492c87322f4e318e2ed5186276d66c363e384d`；独立文件，未覆盖任何原模型，已从 Git 排除。
- 旧导出 SHA256 仍为 `c73c0bc17e20cc7644c54092ab131702e2ff05bbc3dc818f4715b5b9e940ab83`；交接三份 checkpoint SHA 均在结束时复验一致。
- 交接 lineage `lin-b21bf3764abdcfe7`，sequence 3；下一次交接 parent 为 `20261019-163142-codex-g1-robojudo-recovery-sim-migration-01a1219e-c47a-7412-a11f-`。

Confirmed decisions:
- 单一 opt-in owner 是配置 `g1_unilab_recovery_sim`：同时选择专用 policy、MuJoCo env 和 pipeline。旧 `g1_unilab_distill`、旧导出器、真机入口及默认倒地保护保留。混用硬件、IMU、控制周期/子步、动作缩放或模型契约会被拒绝。
- 流程：独立导出器 → TorchScript `(obs[N,99], expert_index[N]:int64)` → 专用物理路由 → 网络动作 `[N,29]` → float32 默认角偏置 → 关节映射 → 原 position actuators。网络没有新增观测维度或重复归一化，不使用 learned router 替代源角色选择。
- 角色 walk=0、stand_height=1、recovery=2。进入 h<.40m 或 tilt>60°；释放必须 h>=.65m、tilt<=10°、|v|<=.20m/s、|gyro|<=.30rad/s、双脚接触连续50个有效控制帧，然后100帧 settle。恢复/settle 期间输入命令为零速度/.754m，保留最新请求及其取消。
- 高度是 flat floor 相对骨盆实测高度；路由使用正向 torso upvector；99D 输入为其负值。局部速度读 pelvis_local_linvel，gyro/upvector 为 torso，脚接触为 XML 中每脚四传感器任一 >.5。没有硬件输入的常数替代。
- 源 combined 实测 gait_constraint disabled：正常站立、行走及 settle 继续推进 raw phase；恢复时清零。目标从合法固定 offset pair [0,π] 起步。旧双专家固定站立 π 相位保留。未复刻源 reset 随机分布或 domain randomization。
- 保持50Hz/3物理子步及源 float32 算术，包括 .65m 的 inclusive float32 边界。计数按 episode/control_frame，重复读帧/dry-run 不消耗时间；reset/reborn 清理状态。新仿真保留源 |v|>20m/s 或 max|joint_vel|>40rad/s 的终止界限，倒地进入恢复而非触发旧 tilt-reborn。没有全局关闭保护。

Progress:
- [x] 交接完整性、双仓库身份与冻结资产核验。
- [x] 独立配置、显式三专家导出、传感器通路、恢复路由/有效命令、reset/dry-run 生命周期及专用保护已实现。
- [x] 确定性源对比：340帧、340次重复读帧、12个阈值边界用例；观测/动作/最终PD目标最大差全部0。比较调用源 `_compute_obs`、`apply_action`、`update_recovery_route` 和角色选择函数。
- [x] nominal 物理参数与42个碰撞几何按名称一致；源 discardvisual 使几何下标不同，视觉网格不参与同下标比较。独立源 nominal 模型回放120控制步/360子步，qpos/qvel/sensordata 最大差均0。
- [x] 17项定向测试通过（旧路径7项、新路径10项），新增文件 Ruff check/format check 与 `git diff --check` 通过。没有运行会导入硬件 SDK 的全注册表测试，不声称全仓库测试通过。
- [x] 独立有界入口运行750步，未导入任何 `unilab` 模块；末态约 .741m/1.04°，恢复及 settle 结束。导出及部署运行本身无需源 Python 类定义。
- [x] 最终有界仿真完成7回合，共3700控制步。原始结果仅保存在本地 `.local-build/g1-recovery-sim/summary.json`，逐回合50Hz trace NPZ、测试及运行日志在同目录；这些原始证据不提交或上传。

  | 场景 | 预算 | 最终结果 |
  | --- | --- | --- |
  | nominal站立 | 150步/3s | 达到连续稳定；末高.736m |
  | 目标.65m站高 | 150步/3s | 末高.638m，低于恢复稳定高度；不记为稳定通过 |
  | .4m/s行走 | 150步/3s | 全程walk路由，前进.987m；恢复静止判据不适用行走成功率 |
  | 仰躺seed0 | 750步/15s | 第237帧释放，完成100帧settle，末态稳定 |
  | 仰躺seed1 | 750步/15s | 第192帧释放，完成settle，末态稳定 |
  | 仰躺seed2 | 750步/15s | 最长连续稳定.78s，未释放恢复路由；末高.752m不能替代完整判据 |
  | 不重置外力摔倒 | 1000步/20s | 第100..109帧施加骨盆-X 600N；episode未变；第906帧稳定释放，截止时仍余7帧settle |

  以上采用 nominal 物理、无观测噪声；仰躺为每关节±.05初态扰动，仅限本批描述性结果。不能替代 E132/E136 或历史成功率。全部回合逐帧的源/目标观测、动作、PD目标与路由一致；没有为结果延长回合或放宽阈值。早期 float64 默认角版本结果不是最终版本证据；接触闭环对微小数值差异敏感，最终只采用本表。

Blockers:
- 仿真适配链已验证；策略可靠性没有全面通过：本批仰躺2/3稳定交接，低站高跟踪有误差，扰动恢复在原预算内未走完缓冲。
- 本批50Hz采样仍观测到约.709rad的PD目标超关节范围及 actuator force 触顶。没有加入动作硬裁剪来掩盖问题。历史脚踝越界、腕饱和、自碰撞负荷问题仍有效；没有重做150Hz完整负荷审查，也没有硬件安全结论。
- 已有 RoboJudo Python3.10/Torch2.13/MuJoCo3.10 环境的 SciPy 二进制不能加载；独立 wheel 也出现相同错误，未继续修复该环境。有效验证运行时为 Python3.13.13/Torch2.7.0/MuJoCo3.8.0，外加 `/private/tmp/robojudo-g1-sim-extra` 的通用依赖；源 `.venv` 未改动。尚未验证图形 viewer 或真机。
- 聊天预设旧 cwd 已不存在；实际目标根由本次用户明确指定。外目录写入经沙箱 escalation 执行。lean-ctx MCP 在本会话不可用，未找到目标本地 LEAN-CTX.md，使用本地文件工具。

Next:
- 本次仿真适配闭环无剩余实现项。恢复可靠性、负荷约束与真实传感器/SDK保护须作为后续独立阶段处理；不自动进入真机。

复现（在目标根目录；以下为本机已验证环境，不会连接真机）：

```bash
export PYTHONPATH="/private/tmp/robojudo-g1-sim-extra:$PWD:$PWD/third_party/mujoco_viewer"
export UV_CACHE_DIR=/private/tmp/unilab-g1-recovery-uv-cache
UV=/Users/sss9999/.local/bin/uv
SIM_PY=/Users/sss9999/enbodied_intelligence/unilab/unilab_stand/.venv/bin/python

# 有界纯 RoboJudo 入口；不导入 UniLab，最多1500控制步。
"$UV" run --no-project --python "$SIM_PY" python scripts/run_unilab_recovery_sim.py --headless --supine --steps 750
# 站立或行走：去掉 --supine，行走加 --walk。
# 常规键盘 viewer 入口（本次未运行）：scripts/run_pipeline.py -c g1_unilab_recovery_sim
# 原键盘 w/s/a/d/q/e；新配置 [ 降高、] 升高；Ctrl+` reborn、Ctrl+Esc退出。

"$UV" run --no-project --python "$SIM_PY" python -m unittest discover -s tests -p 'test_unilab*.py' -v

# 输出目录必须是新目录；先执行离线门槛，通过后才运行物理回放和有界回合。
"$UV" run --no-project --python "$SIM_PY" python scripts/verify_unilab_recovery.py \
  --source-root /Users/sss9999/enbodied_intelligence/unilab/unilab_stand \
  --checkpoint /Users/sss9999/enbodied_intelligence/unilab/dagger_iteration_2.pt \
  --output /private/tmp/g1-recovery-check-new
```

`/private/tmp` 依赖若被系统清理，可在独立目录重新安装 `scipy==1.18.1 python-box==7.4.1 colorlog==6.12.0 msgpack-numpy==0.4.8 msgpack==1.2.3 glfw==2.11.0 imageio==2.38.1`，勿改源环境。源环境已提供 torch/numpy/mujoco/pydantic/yaml/tqdm。后续独立部署环境必须使用已核对的版本重新验证，不能默认原3.10环境可运行。

仅在新位置需要重新导出时，使用源 `uv run --offline --no-sync python` 执行 `scripts/export_unilab_recovery.py CHECKPOINT NEW_OUTPUT`，同时把目标根加入 PYTHONPATH。导出器校验源 SHA/角色/路由契约，拒绝覆盖已有文件；既有双专家导出器保持原样。

回退：选择旧 `g1_unilab_distill` 即回到原路径；本次代码通过独立迁移分支提交/推送，未合并 `release`。勿通过 reset/clean 删除源 dirty 或用户既有 `.DS_Store`。
