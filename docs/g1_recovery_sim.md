# Current task

Goal: 已实现默认关闭的20帧渐进站立协助及全过程负荷回退，并完成源七场景和目标seed2有界验证；冻结模型、奖励、PD和原稳定验收阈值。
Do not: 不连接/控制真机、不训练、不改奖励或稳定阈值、不覆盖旧权重、不上传模型或本地原始证据。既往旧迁移提交80c7b3c已按授权推送；用户现已授权将本轮仿真候选提交并推送到独立分支，实际发布状态以Git为准。

Current identity:
- Target: `/Users/sss9999/enbodied_intelligence/Roboju/RoboJudo_Real`，基线 `release/d0a0ec73686fdf21ff72ad20e79bef3553ada46a`；旧迁移提交80c7b3c已在 `codex/g1-recovery-sim-migration` 发布；当前候选发布分支为 `codex/g1-standing-assist-sim`，按用户授权提交/推送，未合并 `release`。
- Source: `/Users/sss9999/enbodied_intelligence/unilab/unilab_stand`，当前 `codex/g1-standing-assist-sim/e1717b9462f74fcccb430889d36ecf121a1fe1e5`，从原恢复分支创建。源既有 dirty patch 与交接 `source-worktree.patch` 逐字节相同。
- Student: `/Users/sss9999/enbodied_intelligence/unilab/dagger_iteration_2.pt`，SHA256 `27d06b8d73734fb2a17c730aa63ff70b2ba4ffbef7d3d71373df43974ab9bf18`。
- 新导出 `assets/models/g1/unilab/g1_three_expert_dagger_iteration_2/policy.pt`，SHA256 `1435ecffdb457eb8d2cc41445c492c87322f4e318e2ed5186276d66c363e384d`；独立文件，未覆盖任何原模型，已从 Git 排除。
- 旧导出 SHA256 仍为 `c73c0bc17e20cc7644c54092ab131702e2ff05bbc3dc818f4715b5b9e940ab83`；交接三份 checkpoint SHA 均在结束时复验一致。
- 交接 lineage `lin-b21bf3764abdcfe7`，sequence 3；下一次交接 parent 为 `20261019-163142-codex-g1-robojudo-recovery-sim-migration-01a1219e-c47a-7412-a11f-`。

Confirmed decisions:
- 新候选只通过独立 `g1_unilab_standing_assist_sim` 选择：五项原条件连续10帧后尝试20帧混合，过渡后重新累计原50帧，再100帧settle；每回合最多一次尝试。逐帧拒绝比同状态起身专家新增/加重目标范围越界或力矩超限请求的完整候选动作并回退，无逐关节裁剪。倒地/100帧协助超时回退。模型及旧路径不变。
- 源实验入口 `scripts/evaluate_g1_standing_assist.py --standing-assist`；不加标志为OFF。部署候选不依赖UniLab运行时。下列原ROUTING_CONTRACT描述仍是旧 `g1_unilab_recovery_sim` 的基线；新候选有显式独立入口，不改checkpoint原元数据。
- 单一 opt-in owner 是配置 `g1_unilab_recovery_sim`：同时选择专用 policy、MuJoCo env 和 pipeline。旧 `g1_unilab_distill`、旧导出器、真机入口及默认倒地保护保留。混用硬件、IMU、控制周期/子步、动作缩放或模型契约会被拒绝。
- 流程：独立导出器 → TorchScript `(obs[N,99], expert_index[N]:int64)` → 专用物理路由 → 网络动作 `[N,29]` → float32 默认角偏置 → 关节映射 → 原 position actuators。网络没有新增观测维度或重复归一化，不使用 learned router 替代源角色选择。
- 角色 walk=0、stand_height=1、recovery=2。进入 h<.40m 或 tilt>60°；释放必须 h>=.65m、tilt<=10°、|v|<=.20m/s、|gyro|<=.30rad/s、双脚接触连续50个有效控制帧，然后100帧 settle。恢复/settle 期间输入命令为零速度/.754m，保留最新请求及其取消。
- 高度是 flat floor 相对骨盆实测高度；路由使用正向 torso upvector；99D 输入为其负值。局部速度读 pelvis_local_linvel，gyro/upvector 为 torso，脚接触为 XML 中每脚四传感器任一 >.5。没有硬件输入的常数替代。
- 源 combined 实测 gait_constraint disabled：正常站立、行走及 settle 继续推进 raw phase；恢复时清零。目标从合法固定 offset pair [0,π] 起步。旧双专家固定站立 π 相位保留。未复刻源 reset 随机分布或 domain randomization。
- 保持50Hz/3物理子步及源 float32 算术，包括 .65m 的 inclusive float32 边界。计数按 episode/control_frame，重复读帧/dry-run 不消耗时间；reset/reborn 清理状态。新仿真保留源 |v|>20m/s 或 max|joint_vel|>40rad/s 的终止界限，倒地进入恢复而非触发旧 tilt-reborn。没有全局关闭保护。

Progress:
- [x] 用户选择20帧渐进过渡后，源新增独立默认关闭控制器/评测器/25项定向验证；目标新增独立仿真配置、环境/策略/管线和对比入口。源既有tracked dirty patch仍逐字节不变。
  - 最终seed2：1.78s开始协助，2.02s拒绝左踝俯仰越界.007868rad的提案并回退；实际执行到混合权重.60，没有完整走完20帧。回退后原起身专家在3.82s满足原50帧并释放，5.82s结束settle，末态稳定。初次无全过程负荷回退版本虽3.30s释放，但新增踝俯仰越界.207335rad，已否决。
  - 源七场景共3700步：三个仰躺均完成交接/settle，seed0/1释放分别5.40/4.74s，比旧版晚.66/.90s；外力不reset案例15.96s释放、17.96s完成settle。站立/行走/低站高8列轨迹与旧版逐元素相同；低站高既有误差保留。
  - 目标3700帧观测/动作/最终PD目标/路由和3700次重复读取与源一致，控制器类AST相同；目标seed2实际750步/2250子步的状态、传感器、力矩与源一致，独立重算原稳定条件，未导入UniLab。目标旧17项+新2项测试通过。
  - seed2 4–15s最大目标越界.102053→0rad，但早期.696769rad目标越界及135.84Nm/644.04N峰值保留；左腕俯仰/偏航各多2个150Hz饱和采样，后段自碰撞峰值15.17→20.21N，不能声称负荷全面改善或真机通过。
  - [源修复报告与曲线](/Users/sss9999/enbodied_intelligence/unilab/unilab_stand/.local-build/g1_standing_assist/REPORT.md)；目标验证 `.local-build/g1-recovery-sim/standing-assist-verification-final/summary.json`。本轮未训练、未连接硬件、未上传权重或原始证据，候选默认关闭。用户随后授权发布当前目标代码；源实验代码仍在本地独立分支且未提交。
- [x] 用户批准提前接手设计后完成第一关：原版seed2重跑750步与历史逐元素一致；第89帧直接切站立专家的目标跳变由.038716增至.758482rad，左腕俯仰/偏航PD请求超过仿真±5Nm限幅，按约定否决直接切换。未修改生产代码，未执行候选闭环。证据：`.local-build/g1-recovery-sim/diagnostics/standing_assist_preflight/REPORT.md`。
- [x] supine_seed2 诊断完成：当前HEAD `80c7b3c` 与原评测代码/权重身份一致；仅复现750控制步/2250物理子步一次，新旧750×8轨迹逐元素差0。原代码、源dirty、权重、奖励、PD和稳定阈值保持不变。
  - 40次稳定段中断：仅角速度25、仅双脚接触9、仅线速度3、角速度+接触3；最长39/50帧。8.86s角速.309172rad/s使39帧清零，其余条件全达标。
  - 7.14s/9.20s脚球形传感器全0但胶囊仍承载，属于接触覆盖边界；其余7次接触单独中断时该脚无承载法向力记录。4–15s无执行器饱和，仍有以局部z轴为主的转动及右腕—右髋接触；因果贡献尚未隔离。
  - 本地证据：`.local-build/g1-recovery-sim/diagnostics/supine_seed2/REPORT.md`，含3组PNG/PDF同时间轴曲线、条件表、40事件表、29关节分段负荷、原始2250子步记录和复现脚本。仅补充本地诊断材料及本文件；未连接真机、训练、修改策略或新增推送。
- [x] 交接完整性、双仓库身份与冻结资产核验。
- [x] 独立配置、显式三专家导出、传感器通路、恢复路由/有效命令、reset/dry-run 生命周期及专用保护已实现。
- [x] 确定性源对比：340帧、340次重复读帧、12个阈值边界用例；观测/动作/最终PD目标最大差全部0。比较调用源 `_compute_obs`、`apply_action`、`update_recovery_route` 和角色选择函数。
- [x] nominal 物理参数与42个碰撞几何按名称一致；源 discardvisual 使几何下标不同，视觉网格不参与同下标比较。独立源 nominal 模型回放120控制步/360子步，qpos/qvel/sensordata 最大差均0。
- [x] 17项定向测试通过（旧路径7项、新路径10项），新增文件 Ruff check/format check 与 `git diff --check` 通过。没有运行会导入硬件 SDK 的全注册表测试，不声称全仓库测试通过。
- [x] 独立有界入口运行750步，未导入任何 `unilab` 模块；末态约 .741m/1.04°，恢复及 settle 结束。导出及部署运行本身无需源 Python 类定义。
- [x] 旧三专家迁移基线的有界仿真完成7回合，共3700控制步。原始结果仅保存在本地 `.local-build/g1-recovery-sim/summary.json`，逐回合50Hz trace NPZ、测试及运行日志在同目录；这些原始证据不提交或上传。

  | 场景 | 预算 | 最终结果 |
  | --- | --- | --- |
  | nominal站立 | 150步/3s | 达到连续稳定；末高.736m |
  | 目标.65m站高 | 150步/3s | 末高.638m，低于恢复稳定高度；不记为稳定通过 |
  | .4m/s行走 | 150步/3s | 全程walk路由，前进.987m；恢复静止判据不适用行走成功率 |
  | 仰躺seed0 | 750步/15s | 第237帧释放，完成100帧settle，末态稳定 |
  | 仰躺seed1 | 750步/15s | 第192帧释放，完成settle，末态稳定 |
  | 仰躺seed2 | 750步/15s | 最长连续稳定.78s，未释放恢复路由；末高.752m不能替代完整判据 |
  | 不重置外力摔倒 | 1000步/20s | 第100..109帧施加骨盆-X 600N；episode未变；第906帧稳定释放，截止时仍余7帧settle |

  上表为旧配置的历史基线，采用nominal物理、无观测噪声；仰躺为每关节±.05初态扰动，仅限该批描述性结果。不能替代 E132/E136 或历史成功率。全部回合逐帧的源/目标观测、动作、PD目标与路由一致；没有为结果延长回合或放宽阈值。早期float64默认角版本不属于该基线最终证据；接触闭环对微小数值差异敏感。新候选结果见本节首项及修复报告。

Blockers:
- 旧基线本批仰躺2/3；新候选本批3/3并完成缓冲，但仅限nominal/无噪声的小样本。低站高误差、seed0/1交接延迟增加、域随机化/广泛初姿可靠性仍未解决。
- 新候选已记录150Hz负荷并对seed2完整对照；既有早期踝越界、腕饱和、自碰撞保留，还新增短暂腕饱和/交回跳变及后段自碰撞增加。模型限幅不是真机能力，尚无硬件安全结论；未用裁剪或放宽判据掩盖。
- 已有 RoboJudo Python3.10/Torch2.13/MuJoCo3.10 环境的 SciPy 二进制不能加载；独立 wheel 也出现相同错误，未继续修复该环境。有效验证运行时为 Python3.13.13/Torch2.7.0/MuJoCo3.8.0，外加 `/private/tmp/robojudo-g1-sim-extra` 的通用依赖；源 `.venv` 未改动。尚未验证图形 viewer 或真机。
- 聊天预设旧 cwd 已不存在；实际目标根由本次用户明确指定。外目录写入经沙箱 escalation 执行。lean-ctx MCP 在本会话不可用，未找到目标本地 LEAN-CTX.md，使用本地文件工具。

Next:
- 按用户要求发布目标分支，并提供回到UniLab的接续prompt；UniLab已有3个新增实验文件，应先复核再接入默认关闭的正式仿真播放路径，保留负荷代价，不重复实现、不自动训练或进入真机。

新候选复查：沿用下文已验证环境，在目标根执行以下命令，输出目录须全新（会先验证3700帧再有界复现750步）：

```bash
"$UV" run --no-project --python "$SIM_PY" python scripts/verify_standing_assist.py \
  --source-root /Users/sss9999/enbodied_intelligence/unilab/unilab_stand \
  --source-evidence /Users/sss9999/enbodied_intelligence/unilab/unilab_stand/.local-build/g1_standing_assist \
  --output /private/tmp/g1-assist-check-new
# 新配置仅供仿真；图形viewer未验证：scripts/run_pipeline.py -c g1_unilab_standing_assist_sim
```

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
