#!/usr/bin/env bash
# =============================================================================
# bg_run.sh —— 服务器后台任务工具（断开 SSH / 关闭 VS Code 后任务继续运行）
#
# 用法：
#   bash tools/bg_run.sh start  <任务名> <工作目录> <命令...>
#   bash tools/bg_run.sh status <任务名>
#   bash tools/bg_run.sh log    <任务名> [行数]      # tail -f，实时看进度
#   bash tools/bg_run.sh stop   <任务名> [信号]      # 默认 TERM，15s 后 KILL
#   bash tools/bg_run.sh list
#
# 原理：setsid + nohup 让任务脱离当前终端会话（独立 session / 进程组），
#       SSH 断开时终端发出的 SIGHUP 传不到它；stdin 指向 /dev/null，
#       终端消失也不会因写标准输入而挂住。
#
# 产物：
#   <工作目录>/log/<任务名>_<时间戳>.log   完整 stdout + stderr（推荐直接看这个）
#   <工作目录>/log/<任务名>_<时间戳>.pid   进程组 PID
#   ~/.bg_run/<任务名>.{pid,log,info}      注册信息；<任务名>.log 是指向日志的软链
#
# 环境变量：
#   BG_CONDA_ENV  启动前 conda activate 的环境名，默认 sedir；设成 - 则跳过激活
#   BG_CONDA_SH   conda.sh 的路径，默认 /home/liyumeng/miniconda3/etc/profile.d/conda.sh
#   BG_HOME       注册信息目录，默认 ~/.bg_run
#   BG_LOGDIR     日志目录名（相对工作目录），默认 log
# =============================================================================
set -u

CONDA_SH="${BG_CONDA_SH:-/home/liyumeng/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${BG_CONDA_ENV:-sedir}"
BG_HOME="${BG_HOME:-$HOME/.bg_run}"
LOGDIR_NAME="${BG_LOGDIR:-log}"

die()  { echo "[bg_run] $*" >&2; exit 1; }
alive(){ [ -n "${1:-}" ] && kill -0 "$1" 2>/dev/null; }
help() { sed -n '2,22p' "$0"; exit 1; }

mkdir -p "$BG_HOME"
act="${1:-}"

case "$act" in
  start)
    name="${2:-}"; work="${3:-}"
    [ -n "$name" ] && [ -n "$work" ] || help
    shift 3 2>/dev/null || true
    [ "$#" -gt 0 ] || die "缺少要执行的命令"
    [ -d "$work" ] || die "工作目录不存在: $work"
    work="$(cd "$work" && pwd)"
    reg_pid="$BG_HOME/$name.pid"
    if [ -f "$reg_pid" ] && alive "$(cat "$reg_pid")"; then
      die "任务 '$name' 已在运行 (PID $(cat "$reg_pid"))；请先 stop，或换个任务名"
    fi

    logdir="$work/$LOGDIR_NAME"; mkdir -p "$logdir"
    log="$logdir/${name}_$(date +%Y%m%d_%H%M%S).log"

    cmdline=""
    for a in "$@"; do cmdline="$cmdline $(printf '%q' "$a")"; done

    conda_part=":"
    if [ "$CONDA_ENV" != "-" ] && [ -f "$CONDA_SH" ]; then
      conda_part="source '$CONDA_SH' && conda activate '$CONDA_ENV' || exit 1"
    fi

    setsid nohup bash -c "
      cd '$work' || exit 1
      $conda_part
      export PYTHONUNBUFFERED=1
      exec$cmdline
    " > "$log" 2>&1 < /dev/null &
    pid=$!
    sleep 2
    if ! alive "$pid"; then
      echo "[bg_run] 启动失败，日志末尾：" >&2
      tail -n 25 "$log" >&2
      exit 1
    fi

    echo "$pid" > "$reg_pid"
    ln -sfn "$log" "$BG_HOME/$name.log"
    { echo "task=$name"; echo "workdir=$work"; echo "cmd=$(echo "$cmdline" | sed 's/^ //')";
      echo "start=$(date '+%F %T')"; echo "pid=$pid"; } > "$BG_HOME/$name.info"

    echo "[bg_run] 已后台启动，可安全断开 SSH / 关闭 VS Code"
    echo "  任务名   : $name"
    echo "  PID/进程组: $pid"
    echo "  工作目录 : $work"
    echo "  命令     :$(echo "$cmdline" | sed 's/^ //')"
    echo "  日志     : $log"
    echo
    echo "  实时看进度: bash $0 log $name"
    echo "  查看状态  : bash $0 status $name"
    echo "  停止      : bash $0 stop $name"
    ;;

  status)
    name="${2:-}"; [ -n "$name" ] || help
    [ -f "$BG_HOME/$name.info" ] || die "没有名为 '$name' 的任务记录（可用 list 查看）"
    cat "$BG_HOME/$name.info"
    pid="$(cat "$BG_HOME/$name.pid" 2>/dev/null || true)"
    if alive "$pid"; then
      echo "state=RUNNING"
      ps -o pid,pgid,sid,etime,pcpu,pmem,rss,cmd --sid "$pid" 2>/dev/null \
        || ps -o pid,etime,pcpu,pmem,cmd -p "$pid"
    else
      echo "state=EXITED"
      echo "--- 日志末尾 5 行 ---"
      tail -n 5 "$(readlink -f "$BG_HOME/$name.log")" 2>/dev/null
    fi
    ;;

  log)
    name="${2:-}"; n="${3:-200}"; [ -n "$name" ] || help
    [ -f "$BG_HOME/$name.log" ] || die "找不到任务 '$name' 的日志"
    exec tail -n "$n" -f "$(readlink -f "$BG_HOME/$name.log")"
    ;;

  stop)
    name="${2:-}"; sig="${3:-TERM}"; [ -n "$name" ] || help
    pid="$(cat "$BG_HOME/$name.pid" 2>/dev/null || true)"
    [ -n "$pid" ] || die "找不到任务 '$name' 的 PID 记录"
    if ! alive "$pid"; then echo "[bg_run] 任务 '$name' 已经退出"; exit 0; fi
    echo "[bg_run] 向进程组 $pid 发送 SIG$sig ..."
    kill -"$sig" -- "-$pid" 2>/dev/null || kill -"$sig" "$pid" 2>/dev/null
    for _ in $(seq 1 15); do alive "$pid" || break; sleep 1; done
    if alive "$pid"; then
      echo "[bg_run] 仍未退出，发送 SIGKILL"
      kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null
      sleep 2
    fi
    if alive "$pid"; then echo "[bg_run] '$name' 仍在运行，请手动检查"; else echo "[bg_run] 已停止 '$name'"; fi
    echo "  日志: $(readlink -f "$BG_HOME/$name.log")"
    ;;

  list)
    shopt -s nullglob
    found=0
    for f in "$BG_HOME"/*.pid; do
      found=1
      n="$(basename "$f" .pid)"; p="$(cat "$f" 2>/dev/null)"
      if alive "$p"; then st="RUNNING"; else st="EXITED "; fi
      wd="$(sed -n 's/^workdir=//p' "$BG_HOME/$n.info" 2>/dev/null)"
      printf '%-22s %s pid=%-8s %s\n' "$n" "$st" "$p" "$wd"
    done
    [ "$found" = 1 ] || echo "（暂无后台任务记录）"
    ;;

  *) help ;;
esac
