"""批量生成长尾驾驶场景."""

import argparse

from driveworld.pipeline import DriveWorldPipeline
from driveworld.utils import load_config, get_scenario_names, get_generation_seed


def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="config/default.yaml")
    pre_args, _ = pre_parser.parse_known_args()
    default_scenarios = get_scenario_names(config)

    parser = argparse.ArgumentParser(description="批量生成长尾驾驶场景")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=default_scenarios,
        help="要生成的场景列表（提示词见 config scenarios）",
    )
    config = load_config(pre_args.config)
    parser.add_argument(
        "--seed",
        type=int,
        default=get_generation_seed(config),
        help="基础随机种子（默认读取 config generation.seed）",
    )
    parser.add_argument("--config", type=str, default=pre_args.config)
    args = parser.parse_args()

    pipeline = DriveWorldPipeline(config_path=args.config)
    results = []

    for i, scenario in enumerate(args.scenarios):
        print(f"\n{'='*60}")
        print(f"生成场景 [{i+1}/{len(args.scenarios)}]: {scenario}")
        print(f"{'='*60}")

        result = pipeline.run_text_to_video(
            scenario=scenario,
            seed=args.seed + i,
        )
        results.append(result)

    print(f"\n全部完成，共生成 {len(results)} 个场景。")
    for r in results:
        print(f"  - {r['scenario']}: risk={r['risk_assessment']['level']} "
              f"({r['risk_assessment']['overall_score']:.3f})")


if __name__ == "__main__":
    main()
