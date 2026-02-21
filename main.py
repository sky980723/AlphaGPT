"""
AlphaGPT - Meme Coin Alpha Mining System
统一入口，支持 IDE 直接运行
"""
import argparse
import asyncio


def run_pipeline():
    """数据管道：拉取链上数据入库"""
    from data_pipeline.run_pipeline import main
    asyncio.run(main())


def run_train():
    """模型训练：AI 挖掘最优交易公式"""
    from model_core.engine import AlphaEngine
    eng = AlphaEngine(use_lord_regularization=True)
    eng.train()


def run_strategy():
    """实盘策略：自动交易（需配置钱包私钥）"""
    from strategy_manager.runner import StrategyRunner
    runner = StrategyRunner()
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(runner.initialize())
        loop.run_until_complete(runner.run_loop())
    except KeyboardInterrupt:
        loop.run_until_complete(runner.shutdown())


def run_dashboard():
    """监控面板：启动 Streamlit Web UI"""
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard/app.py"])


COMMANDS = {
    "pipeline": ("数据采集入库", run_pipeline),
    "train": ("模型训练", run_train),
    "strategy": ("实盘交易", run_strategy),
    "dashboard": ("监控面板", run_dashboard),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaGPT - Meme Coin Alpha Mining System")
    parser.add_argument("command", nargs="?", choices=COMMANDS.keys(),
                        help="要执行的命令")
    args = parser.parse_args()

    if not args.command:
        print("\nAlphaGPT 可用命令:\n")
        for cmd, (desc, _) in COMMANDS.items():
            print(f"  python main.py {cmd:12s} # {desc}")
        print()
    else:
        desc, func = COMMANDS[args.command]
        print(f">>> {desc}")
        func()
